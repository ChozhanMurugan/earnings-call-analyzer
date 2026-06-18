"""Command-line entry point. Run ``python -m eca.cli --help``."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import typer

from eca.config import settings
from eca.features import build_features_df
from eca.ingest import Transcript, load_hf_earnings_calls
from eca.utils import logger

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Earnings Call Analyzer")


# ----- ingest + features -----

@app.command("build-dataset")
def build_dataset(
    source: str = typer.Option("hf", help="hf | edgar | motley_fool"),
    tickers: str = typer.Option("", help="comma-separated tickers (edgar/motley_fool only)"),
    limit: int = typer.Option(200, help="hf row cap / per-ticker filing cap"),
    output: Path | None = typer.Option(None, help="output parquet (default: data/processed/features_labelled.parquet)"),
    skip_labels: bool = typer.Option(False, help="skip yfinance label download (offline mode)"),
) -> None:
    """Pull transcripts, extract features, attach T+1 excess-return labels."""
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    output = output or (settings.processed_dir / "features_labelled.parquet")

    transcripts: Iterator[Transcript]
    if source == "hf":
        transcripts = load_hf_earnings_calls(limit=limit)
    elif source == "edgar":
        from eca.ingest.edgar import fetch_many

        if not tickers:
            raise typer.BadParameter("--tickers required for edgar")
        transcripts = fetch_many([t.strip().upper() for t in tickers.split(",") if t.strip()], per_ticker=limit)
    elif source == "motley_fool":
        from eca.ingest.motley_fool import fetch_many

        if not tickers:
            raise typer.BadParameter("--tickers required for motley_fool")
        typer.secho(
            "Motley Fool ToS prohibits scraping. Use only for personal research.",
            fg=typer.colors.YELLOW,
        )
        transcripts = fetch_many([t.strip().upper() for t in tickers.split(",") if t.strip()], per_ticker=limit)
    else:
        raise typer.BadParameter(f"unknown source: {source}")

    feats = build_features_df(transcripts)
    if feats.empty:
        typer.secho("no transcripts ingested", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    logger.info(f"built features for {len(feats)} transcripts")

    if not skip_labels:
        from eca.prices import attach_labels

        feats = attach_labels(feats)

    feats.to_parquet(output, index=False)
    typer.secho(f"wrote {len(feats)} rows -> {output}", fg=typer.colors.GREEN)


# ----- train -----

@app.command("train")
def train_cmd(
    input: Path = typer.Option(
        None,
        help="features parquet with labels (default: data/processed/features_labelled.parquet)",
    ),
    n_splits: int = typer.Option(5, help="walk-forward CV folds"),
) -> None:
    """Train the LightGBM directional classifier."""
    from eca.model.train import train

    input = input or (settings.processed_dir / "features_labelled.parquet")
    if not input.exists():
        typer.secho(f"missing {input}; run `build-dataset` first", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    df = pd.read_parquet(input)
    result = train(df, n_splits=n_splits)
    typer.secho(
        f"cv_mean_accuracy={result.mean_accuracy:.3f} cv_mean_auc={result.mean_auc:.3f}",
        fg=typer.colors.GREEN,
    )
    top = list(result.feature_importances.items())[:10]
    typer.echo("top features:")
    for k, v in top:
        typer.echo(f"  {k:30s} {v:.1f}")


# ----- predict (cache predictions for the backtest endpoint) -----

@app.command("predict")
def predict_cmd(
    input: Path | None = typer.Option(None),
    output: Path | None = typer.Option(None),
) -> None:
    """Score every labelled row and cache predictions for backtesting."""
    from eca.model.predict import load_model

    input = input or (settings.processed_dir / "features_labelled.parquet")
    output = output or (settings.processed_dir / "predictions.parquet")
    if not input.exists():
        typer.secho(f"missing {input}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    df = pd.read_parquet(input)
    predictor = load_model()
    df["prob_up"] = predictor.predict_frame(df)
    df.to_parquet(output, index=False)
    typer.secho(f"wrote predictions ({len(df)} rows) -> {output}", fg=typer.colors.GREEN)


# ----- backtest -----

@app.command("backtest")
def backtest_cmd(
    ticker: str | None = typer.Option(None, help="restrict to one ticker"),
    threshold: float = typer.Option(0.0, help="trade only when |prob - 0.5| > threshold"),
    input: Path | None = typer.Option(None),
) -> None:
    """Run the vectorized backtest and print summary."""
    from eca.backtest import run_backtest

    input = input or (settings.processed_dir / "predictions.parquet")
    if not input.exists():
        typer.secho(f"missing {input}; run `predict` first", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    df = pd.read_parquet(input)
    if ticker:
        df = df[df["ticker"].str.upper() == ticker.upper()]
        if df.empty:
            typer.secho(f"no predictions for {ticker}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
    result = run_backtest(df, threshold=threshold)
    for k, v in result.summary().items():
        typer.echo(f"{k:24s} {v}")


# ----- serve -----

@app.command("serve")
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the FastAPI app via uvicorn."""
    import uvicorn

    uvicorn.run("eca.api.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
