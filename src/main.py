import argparse

from backtest import run_backtest
from download_data import prepare_data
from evaluate import evaluate_performance
from plots import create_plots


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the ETF portfolio optimization project end to end."
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached raw prices instead of forcing a fresh yfinance download.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=5.0,
        help="Common risk-aversion parameter. Used for Classical Markowitz and as the fallback for Robust Markowitz unless --robust-gamma is set.",
    )
    parser.add_argument(
        "--robust-gamma",
        type=float,
        default=20.0,
        help="Risk-aversion parameter for the robust Markowitz optimizer.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.25,
        help="Strength of the robust expected-return haircut in the robust Markowitz optimizer.",
    )
    parser.add_argument(
        "--cov-uncertainty",
        type=float,
        default=0.20,
        help="Diagonal covariance uncertainty bump used by the robust Markowitz optimizer.",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.10,
        help="Maximum allowed portfolio weight per ETF.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Step 1/4: Downloading and cleaning ETF data...")
    prepare_data(force_download=not args.use_cache, allow_cached=True)

    print()
    print("Step 2/4: Running the quarterly backtest...")
    run_backtest(
        classical_gamma=args.gamma,
        robust_gamma=args.robust_gamma,
        rho=args.rho,
        cov_uncertainty=args.cov_uncertainty,
        max_weight=args.max_weight,
    )

    print()
    print("Step 3/4: Evaluating strategy performance...")
    evaluate_performance()

    print()
    print("Step 4/4: Generating plots...")
    create_plots()

    print()
    print("Pipeline complete. Check the data/ and outputs/ folders for results.")


if __name__ == "__main__":
    main()
