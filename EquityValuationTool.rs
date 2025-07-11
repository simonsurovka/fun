// Equity Valuation Tool  

// Analyze any publicly traded equity by providing ticker symbol.
// Example: cargo run -- --ticker <TICKER>

mod data_sources;
mod scraping;
mod parsing;
mod financials;
mod valuation;
mod reporting;
mod utils;

use clap::{Parser, CommandFactory};
use reporting::report_company_analysis;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::task;
use utils::cache::{get_or_fetch_filings, CacheConfig};
use utils::config::AppConfig;
use utils::health::{health_check_server, HealthStatus};
use utils::metrics::{init_metrics, METRICS};
use utils::rate_limit::{RateLimiter, RateLimitConfig};
use utils::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use tracing::{error, info, instrument, warn, debug, span, Level};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use anyhow::{Result, Context};
use once_cell::sync::Lazy;
use std::process;

static GLOBAL_RATE_LIMITER: Lazy<RateLimiter> = Lazy::new(|| {
    RateLimiter::new(RateLimitConfig {
        max_requests_per_minute: 120,
        burst: 20,
    })
});

static GLOBAL_CIRCUIT_BREAKER: Lazy<CircuitBreaker> = Lazy::new(|| {
    CircuitBreaker::new(CircuitBreakerConfig {
        failure_threshold: 5,
        recovery_timeout: Duration::from_secs(60),
    })
});


#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long)]
    ticker: String,

    #[arg(short, long, default_value = "./cache")]
    cache_dir: PathBuf,

    #[arg(long, default_value_t = false)]
    force_refresh: bool,

    #[arg(short = 'c', long, default_value = "config.yaml")]
    config: PathBuf,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer())
        .init();
    init_metrics();

    let cli = Cli::parse();
    let config = AppConfig::from_file(&cli.config)
        .unwrap_or_else(|e| {
            error!(error = %e, "Failed to load configuration");
            process::exit(1);
        });

    let health_status = Arc::new(Mutex::new(HealthStatus::Healthy));
    let health_status_clone = health_status.clone();
    tokio::spawn(async move {
        if let Err(e) = health_check_server(health_status_clone, config.health_check_port).await {
            error!(error = %e, "Health check server failed");
        }
    });

    #[cfg(unix)]
    if let Some(user) = &config.security.run_as_user {
        utils::security::drop_privileges(user).unwrap_or_else(|e| {
            error!(error = %e, "Failed to drop privileges");
            process::exit(1);
        });
    }

    if let Err(e) = run(cli, config, health_status).await {
        error!(error = %e, "Fatal error in main execution");
        process::exit(1);
    }
}

#[instrument(skip(config, health_status))]
async fn run(cli: Cli, config: AppConfig, health_status: Arc<Mutex<HealthStatus>>) -> Result<()> {
    let ticker = cli.ticker.to_uppercase();
    let cache_config = CacheConfig {
        dir: cli.cache_dir.clone(),
        force_refresh: cli.force_refresh,
    };

    let filings = {
        let mut retries = 0u32;
        let max_retries = config.retries.max_sec_filing_retries;
        let mut backoff = Duration::from_secs(1);
        loop {
            if !GLOBAL_RATE_LIMITER.check() {
                warn!("Rate limit exceeded for SEC filings fetch");
                METRICS.rate_limit_exceeded.inc();
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(Duration::from_secs(32));
                continue;
            }
            if !GLOBAL_CIRCUIT_BREAKER.allow() {
                error!("Circuit breaker open for SEC filings fetch");
                METRICS.circuit_breaker_open.inc();
                *health_status.lock().await = HealthStatus::Degraded;
                return Err(anyhow::anyhow!("Circuit breaker open for SEC filings fetch"));
            }
            match get_or_fetch_filings(&ticker, &cache_config).await {
                Ok(filings) => {
                    GLOBAL_CIRCUIT_BREAKER.success();
                    METRICS.filings_fetch_success.inc();
                    debug!(ticker = %ticker, "SEC filings fetched successfully");
                    break filings;
                }
                Err(e) => {
                    GLOBAL_CIRCUIT_BREAKER.failure();
                    METRICS.filings_fetch_failure.inc();
                    error!(error = %e, ticker = %ticker, attempt = retries + 1, "Error fetching SEC filings");
                    retries += 1;
                    if retries > max_retries {
                        *health_status.lock().await = HealthStatus::Degraded;
                        error!(ticker = %ticker, "Exceeded max retries for SEC filings");
                        return Err(e).context("Exceeded max retries for SEC filings");
                    }
                    let sleep_dur = Duration::from_secs(2u64.pow(retries.min(5)));
                    tokio::time::sleep(sleep_dur).await;
                }
            }
        }
    };

    let filings = Arc::new(filings);

    let income_f = {
        let filings = Arc::clone(&filings);
        task::spawn_blocking(move || parsing::extract_income_statement(&filings))
    };
    let balance_f = {
        let filings = Arc::clone(&filings);
        task::spawn_blocking(move || parsing::extract_balance_sheet(&filings))
    };
    let cashflow_f = {
        let filings = Arc::clone(&filings);
        task::spawn_blocking(move || parsing::extract_cashflow_statement(&filings))
    };

    let (income_statement, balance_sheet, cashflow_statement) = match tokio::try_join!(income_f, balance_f, cashflow_f) {
        Ok((Ok(income), Ok(balance), Ok(cashflow))) => {
            METRICS.financials_parse_success.inc();
            debug!(ticker = %ticker, "Financial statements parsed successfully");
            (income, balance, cashflow)
        }
        Ok((Err(e), _, _)) | Ok((_, Err(e), _)) | Ok((_, _, Err(e))) | Err(e) => {
            METRICS.financials_parse_failure.inc();
            *health_status.lock().await = HealthStatus::Degraded;
            error!(error = %e, ticker = %ticker, "Error parsing financial statements");
            return Err(anyhow::anyhow!("Error parsing financial statements: {}", e));
        }
    };

    let adjusted_cashflows = match financials::analyze_and_adjust_cashflows(&cashflow_statement) {
        Ok(cf) => {
            METRICS.cashflow_adjust_success.inc();
            debug!(ticker = %ticker, "Cash flows adjusted successfully");
            cf
        }
        Err(e) => {
            METRICS.cashflow_adjust_failure.inc();
            *health_status.lock().await = HealthStatus::Degraded;
            error!(error = %e, ticker = %ticker, "Error adjusting cash flows");
            return Err(anyhow::anyhow!("Error adjusting cash flows: {}", e));
        }
    };

    let adj_cf = adjusted_cashflows.clone();
    let inc_st = income_statement.clone();
    let bal_sh = balance_sheet.clone();

    let dcf_handle = task::spawn_blocking(move || valuation::discounted_cash_flow(&adj_cf, &inc_st, &bal_sh));
    let tbv_handle = {
        let bal_sh = balance_sheet.clone();
        task::spawn_blocking(move || valuation::tangible_book_value(&bal_sh))
    };
    let lv_handle = {
        let bal_sh = balance_sheet.clone();
        task::spawn_blocking(move || valuation::liquidation_value(&bal_sh))
    };

    let (dcf_valuation, tbv_valuation, lv_valuation) = match tokio::try_join!(dcf_handle, tbv_handle, lv_handle) {
        Ok((Ok(dcf), Ok(tbv), Ok(lv))) => {
            METRICS.valuation_success.inc();
            debug!(ticker = %ticker, "Valuations completed successfully");
            (dcf, tbv, lv)
        }
        Ok((Err(e), _, _)) | Ok((_, Err(e), _)) | Ok((_, _, Err(e))) | Err(e) => {
            METRICS.valuation_failure.inc();
            *health_status.lock().await = HealthStatus::Degraded;
            error!(error = %e, ticker = %ticker, "Error during valuation");
            return Err(anyhow::anyhow!("Error during valuation: {}", e));
        }
    };

    let shares_outstanding = match financials::get_shares_outstanding(&balance_sheet) {
        Ok(shares) => shares,
        Err(e) => {
            METRICS.shares_outstanding_failure.inc();
            *health_status.lock().await = HealthStatus::Degraded;
            error!(error = %e, ticker = %ticker, "Error getting shares outstanding");
            return Err(anyhow::anyhow!("Error getting shares outstanding: {}", e));
        }
    };

    let per_share_valuations = match valuation::per_share_valuations(
        dcf_valuation,
        tbv_valuation,
        lv_valuation,
        shares_outstanding,
    ) {
        Ok(val) => {
            METRICS.per_share_valuation_success.inc();
            debug!(ticker = %ticker, "Per-share valuations calculated successfully");
            val
        }
        Err(e) => {
            METRICS.per_share_valuation_failure.inc();
            *health_status.lock().await = HealthStatus::Degraded;
            error!(error = %e, ticker = %ticker, "Error calculating per-share valuations");
            return Err(anyhow::anyhow!("Error calculating per-share valuations: {}", e));
        }
    };

    match report_company_analysis(
        &ticker,
        &income_statement,
        &balance_sheet,
        &cashflow_statement,
        dcf_valuation,
        tbv_valuation,
        lv_valuation,
        per_share_valuations,
    ) {
        Ok(_) => {
            METRICS.reporting_success.inc();
            info!(ticker = %ticker, "Report generated successfully");
        }
        Err(e) => {
            METRICS.reporting_failure.inc();
            *health_status.lock().await = HealthStatus::Degraded;
            error!(error = %e, ticker = %ticker, "Error generating report");
            return Err(anyhow::anyhow!("Error generating report: {}", e));
        }
    }

    Ok(())
}
