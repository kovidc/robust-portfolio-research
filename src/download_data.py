from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
TRAINING_WINDOW_DAYS = 504
MAX_MISSING_FRACTION = 0.10
FORWARD_FILL_LIMIT = 5

ETF_UNIVERSE = {
    "us_broad_equity": [
        "SPY",
        "IVV",
        "VOO",
        "VTI",
        "ITOT",
        "DIA",
        "QQQ",
        "IWM",
        "IJH",
        "IJR",
        "MDY",
        "RSP",
        "VV",
        "VB",
        "VO",
        "SCHB",
        "SCHA",
        "SCHX",
    ],
    "us_sectors": [
        "XLB",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLU",
        "XLV",
        "XLY",
        "XHB",
        "XBI",
        "XME",
        "XOP",
        "XRT",
        "KBE",
        "KRE",
        "SOXX",
        "SMH",
    ],
    "international_equity": [
        "EFA",
        "VEA",
        "IEFA",
        "ACWI",
        "VT",
        "VEU",
        "VXUS",
        "EWG",
        "EWJ",
        "EWU",
        "EWC",
        "EWA",
        "EWQ",
        "EWI",
        "EWP",
        "EWL",
        "EWN",
        "EWD",
        "EWO",
        "EWK",
        "EWT",
        "EWH",
        "EWS",
        "EZA",
    ],
    "emerging_markets": [
        "EEM",
        "VWO",
        "IEMG",
        "SCHE",
        "FXI",
        "EWZ",
        "EPI",
        "INDA",
        "KWEB",
        "EWW",
        "THD",
        "TUR",
    ],
    "bonds_treasuries": [
        "AGG",
        "BND",
        "GOVT",
        "IEF",
        "IEI",
        "SHY",
        "VGIT",
        "VGLT",
        "TLT",
        "TLH",
        "TIP",
        "STIP",
        "SCHR",
        "SCHZ",
        "BSV",
        "BIV",
        "MUB",
    ],
    "corporate_bonds": [
        "LQD",
        "VCIT",
        "VCSH",
        "HYG",
        "JNK",
        "IGSB",
        "SJNK",
        "EMB",
        "PFF",
        "CWB",
    ],
    "real_estate": [
        "VNQ",
        "IYR",
        "RWR",
        "SCHH",
        "RWX",
        "ICF",
        "REM",
        "VNQI",
    ],
    "commodities": [
        "GLD",
        "IAU",
        "SGOL",
        "SLV",
        "DBC",
        "GSG",
        "USO",
        "UNG",
        "DBA",
        "CORN",
        "JO",
        "WEAT",
        "PDBC",
        "BNO",
    ],
    "dividend_factor": [
        "VIG",
        "VYM",
        "SCHD",
        "DGRO",
        "DVY",
        "SDY",
        "HDV",
        "USMV",
        "SPLV",
        "MTUM",
        "QUAL",
        "VLUE",
        "SIZE",
        "EFAV",
        "EEMV",
    ],
    "growth_value": [
        "VUG",
        "VTV",
        "IWF",
        "IWD",
        "IWO",
        "IWN",
        "VOOG",
        "VOOV",
        "RPG",
        "RPV",
        "IUSG",
        "IUSV",
    ],
}


def get_universe_metadata():
    """Build a metadata table for the requested ETF universe."""
    rows = []

    for asset_class, tickers in ETF_UNIVERSE.items():
        for ticker in tickers:
            rows.append({"ticker": ticker, "asset_class": asset_class})

    metadata = pd.DataFrame(rows).drop_duplicates(subset="ticker")
    metadata = metadata.sort_values("ticker").reset_index(drop=True)
    return metadata


def get_universe_tickers():
    """Return the sorted list of unique ETF tickers."""
    metadata = get_universe_metadata()
    return metadata["ticker"].tolist()


def _extract_adjusted_close(downloaded_data):
    """Handle the different column layouts returned by yfinance."""
    if downloaded_data.empty:
        raise ValueError("No price data was downloaded.")

    if isinstance(downloaded_data.columns, pd.MultiIndex):
        level_zero = downloaded_data.columns.get_level_values(0)
        level_one = downloaded_data.columns.get_level_values(1)

        if "Adj Close" in level_zero:
            prices = downloaded_data["Adj Close"].copy()
        elif "Close" in level_zero:
            prices = downloaded_data["Close"].copy()
        elif "Adj Close" in level_one:
            prices = downloaded_data.xs("Adj Close", axis=1, level=1).copy()
        elif "Close" in level_one:
            prices = downloaded_data.xs("Close", axis=1, level=1).copy()
        else:
            raise ValueError("Could not find adjusted close or close prices in download.")
    else:
        series_name = downloaded_data.name or "price"
        prices = downloaded_data.to_frame(name=series_name)

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().sort_index(axis=1)
    return prices


def download_raw_prices(tickers, start_date=START_DATE, end_date=END_DATE):
    """Download raw ETF prices from yfinance."""
    print(f"Downloading prices for {len(tickers)} ETFs from {start_date} to {end_date}...")

    downloaded_data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    prices = _extract_adjusted_close(downloaded_data)
    available_tickers = [ticker for ticker in tickers if ticker in prices.columns]
    prices = prices[available_tickers]

    if prices.empty:
        raise ValueError("Downloaded price table is empty after selecting ETF columns.")

    return prices


def load_cached_raw_prices(raw_prices_path):
    """Load raw prices from the cached CSV file."""
    print(f"Loading cached raw prices from {raw_prices_path}...")
    prices = pd.read_csv(raw_prices_path, index_col=0, parse_dates=True)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().sort_index(axis=1)
    return prices


def clean_price_data(
    prices_raw,
    max_missing_fraction=MAX_MISSING_FRACTION,
    forward_fill_limit=FORWARD_FILL_LIMIT,
):
    """
    Clean price data by:
    1. Dropping ETFs with too much missing data.
    2. Forward filling only short missing gaps.
    3. Dropping any remaining rows with missing values.
    4. Converting clean prices to daily returns.
    """
    if prices_raw.empty:
        raise ValueError("Raw price data is empty.")

    missing_fraction = prices_raw.isna().mean().sort_index()
    first_valid_dates = prices_raw.apply(lambda column: column.first_valid_index())
    last_valid_dates = prices_raw.apply(lambda column: column.last_valid_index())

    kept_tickers = missing_fraction[missing_fraction <= max_missing_fraction].index.tolist()
    dropped_tickers = missing_fraction[missing_fraction > max_missing_fraction].index.tolist()

    print(f"Keeping {len(kept_tickers)} ETFs and dropping {len(dropped_tickers)} for missing data.")

    prices_clean = prices_raw[kept_tickers].copy()
    prices_clean = prices_clean.ffill(limit=forward_fill_limit)
    prices_clean = prices_clean.dropna(axis=0, how="any")

    if prices_clean.empty:
        raise ValueError("Clean price data is empty after removing missing observations.")

    returns_clean = prices_clean.pct_change(fill_method=None).dropna()

    if returns_clean.empty:
        raise ValueError("Return data is empty after converting prices to returns.")

    metadata = get_universe_metadata().copy()
    metadata["downloaded"] = metadata["ticker"].isin(prices_raw.columns)
    metadata["missing_fraction"] = metadata["ticker"].map(missing_fraction).fillna(1.0)
    metadata["first_valid_date"] = metadata["ticker"].map(first_valid_dates)
    metadata["last_valid_date"] = metadata["ticker"].map(last_valid_dates)
    metadata["kept_after_cleaning"] = metadata["ticker"].isin(kept_tickers)
    metadata["dropped_for_missing_data"] = metadata["ticker"].isin(dropped_tickers)

    for column in ["first_valid_date", "last_valid_date"]:
        metadata[column] = pd.to_datetime(metadata[column]).dt.strftime("%Y-%m-%d")
        metadata[column] = metadata[column].fillna("")

    return prices_clean, returns_clean, metadata


def compute_quarterly_rebalance_dates(returns_clean, training_window_days=TRAINING_WINDOW_DAYS):
    """Use the first trading day of each quarter after enough training data exists."""
    if returns_clean.empty:
        raise ValueError("Return data is empty, so rebalance dates cannot be created.")

    quarterly_starts = returns_clean.groupby(returns_clean.index.to_period("Q")).apply(
        lambda frame: frame.index.min()
    )
    quarterly_starts = pd.DatetimeIndex(quarterly_starts.tolist())

    valid_rebalance_dates = []
    for rebalance_date in quarterly_starts:
        row_position = returns_clean.index.get_loc(rebalance_date)
        if row_position >= training_window_days:
            valid_rebalance_dates.append(rebalance_date)

    rebalance_dates = pd.DataFrame({"rebalance_date": pd.DatetimeIndex(valid_rebalance_dates)})
    return rebalance_dates


def prepare_data(force_download=False, allow_cached=True):
    """Download, clean, and save all data needed by the backtest."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_prices_path = DATA_DIR / "prices_raw.csv"
    clean_prices_path = DATA_DIR / "prices_clean.csv"
    clean_returns_path = DATA_DIR / "returns_clean.csv"
    metadata_path = DATA_DIR / "universe_metadata.csv"
    rebalance_path = DATA_DIR / "quarterly_rebalance_dates.csv"

    tickers = get_universe_tickers()
    prices_raw = None

    try:
        if force_download or not raw_prices_path.exists():
            prices_raw = download_raw_prices(tickers=tickers)
        else:
            prices_raw = load_cached_raw_prices(raw_prices_path)
    except Exception as error:
        print(f"Fresh download failed: {error}")
        if allow_cached and raw_prices_path.exists():
            prices_raw = load_cached_raw_prices(raw_prices_path)
        else:
            raise

    prices_raw = prices_raw.copy()
    available_tickers = [ticker for ticker in tickers if ticker in prices_raw.columns]
    prices_raw = prices_raw[available_tickers]
    prices_raw.to_csv(raw_prices_path)

    prices_clean, returns_clean, metadata = clean_price_data(prices_raw)
    rebalance_dates = compute_quarterly_rebalance_dates(returns_clean)

    prices_clean.to_csv(clean_prices_path)
    returns_clean.to_csv(clean_returns_path)
    metadata.to_csv(metadata_path, index=False)
    rebalance_dates.to_csv(rebalance_path, index=False)

    print("Saved cleaned data files:")
    print(f"  {raw_prices_path}")
    print(f"  {clean_prices_path}")
    print(f"  {clean_returns_path}")
    print(f"  {metadata_path}")
    print(f"  {rebalance_path}")
    print()
    print("Data summary:")
    print(f"  Raw prices shape: {prices_raw.shape}")
    print(f"  Clean prices shape: {prices_clean.shape}")
    print(f"  Clean returns shape: {returns_clean.shape}")
    print(f"  Clean data start: {prices_clean.index.min().date()}")
    print(f"  Clean data end: {prices_clean.index.max().date()}")
    print(f"  Number of rebalance dates: {len(rebalance_dates)}")

    return {
        "prices_raw": prices_raw,
        "prices_clean": prices_clean,
        "returns_clean": returns_clean,
        "universe_metadata": metadata,
        "quarterly_rebalance_dates": rebalance_dates,
    }


if __name__ == "__main__":
    prepare_data(force_download=True, allow_cached=True)
