"""
Shared fixtures for the mozaic test suite.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

import pytest

import mozaic
import mozaic.utils
from mozaic.core import Mozaic
from mozaic.tile import Tile
from mozaic.models import desktop_forecast_model


# ---------------------------------------------------------------------------
# Mock forecast model
# ---------------------------------------------------------------------------

def _make_mock_forecast_model(seed=42):
    """Return a deterministic mock forecast model callable.

    The model produces 1000 samples centred on the historical median, with 1%
    noise. The returned triple matches what real models return:
      (predictive_samples, prophet_model, prophet_forecast_df)
    """
    def mock_model(historical_data, historical_dates, forecast_dates):
        n = len(forecast_dates)
        vals = historical_data.dropna()
        base_value = float(vals.median()) if len(vals) > 0 else 1000.0

        rng = np.random.RandomState(seed)
        samples = pd.DataFrame(
            np.abs(rng.randn(n, 1000) * (base_value * 0.01) + base_value),
            columns=range(1000),
        )

        mock_forecast = pd.DataFrame({
            "ds": list(forecast_dates),
            "trend": np.full(n, base_value),
            "yhat": np.full(n, base_value),
            "yhat_lower": np.full(n, base_value * 0.9),
            "yhat_upper": np.full(n, base_value * 1.1),
        })

        class MockProphet:
            pass

        return samples, MockProphet(), mock_forecast

    return mock_model


@pytest.fixture
def mock_forecast_model():
    """Pytest fixture that provides a deterministic mock forecast model."""
    return _make_mock_forecast_model()


# ---------------------------------------------------------------------------
# Tile factory (also usable standalone, not just as a fixture)
# ---------------------------------------------------------------------------

def _make_tile(
    metric="DAU",
    country="US",
    population="win10",
    forecast_start_date="2025-01-15",
    forecast_end_date="2025-01-28",
    base_value=1000.0,
    seed=42,
    historical_dates=None,
    n_historical=60,
):
    """Create a Tile backed by a mock forecast model.

    If *historical_dates* is given it overrides *n_historical*.
    """
    if historical_dates is None:
        # End just before forecast_start_date so there is no overlap
        historical_dates = pd.date_range("2024-11-16", periods=n_historical, freq="D")

    n = len(historical_dates)
    rng = np.random.RandomState(seed)
    raw_data = pd.Series(
        np.abs(rng.randn(n) * (base_value * 0.05) + base_value)
    )

    return Tile(
        metric=metric,
        country=country,
        population=population,
        forecast_start_date=forecast_start_date,
        forecast_end_date=forecast_end_date,
        forecast_model=_make_mock_forecast_model(seed),
        historical_dates=historical_dates,
        raw_historical_data=raw_data,
    )


@pytest.fixture
def make_tile():
    """Fixture that exposes :func:`_make_tile` as a factory."""
    return _make_tile


# ---------------------------------------------------------------------------
# Mozaic factory
# ---------------------------------------------------------------------------

@pytest.fixture
def make_mozaic():
    """Fixture that provides a Mozaic factory backed by mock tiles.

    All tiles share the same *historical_dates* and *forecast_dates* (required
    by Mozaic's constructor assertions).
    """
    def factory(
        metric="DAU",
        country="US",
        populations=("win10", "win11"),
        forecast_start_date="2025-01-15",
        forecast_end_date="2025-01-28",
        n_historical=60,
        base_value=1000.0,
        is_country_level=False,
    ):
        # Shared date range so Mozaic's equality checks pass
        historical_dates = pd.date_range("2024-11-16", periods=n_historical, freq="D")

        tiles = []
        for i, pop in enumerate(populations):
            tile = _make_tile(
                metric=metric,
                country=country,
                population=pop,
                forecast_start_date=forecast_start_date,
                forecast_end_date=forecast_end_date,
                base_value=base_value * (1 + i * 0.1),
                seed=42 + i,
                historical_dates=historical_dates,
            )
            tiles.append(tile)

        return Mozaic(
            tiles=tiles,
            forecast_model=_make_mock_forecast_model(99),
            is_country_level=is_country_level,
        )

    return factory


# ---------------------------------------------------------------------------
# Paths to test data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def short_csv_path():
    """Path to the short desktop DAU CSV used in integration tests."""
    return Path(__file__).parent / "data" / "desktop_dau.short.csv"


@pytest.fixture(scope="session")
def parquet_path():
    """Path to the full legacy parquet dataset used in E2E tests."""
    return Path(__file__).parent / "data" / "legacy_desktop_dau.parquet"


# ---------------------------------------------------------------------------
# Session-scoped full-pipeline fixture (short CSV)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cur_mozaic(short_csv_path) -> Mozaic:
    """Session-scoped fixture: full pipeline on the short desktop DAU CSV.

    Used by integration tests (marked *slow*) and by the existing golden-data
    regression tests in test_core_additions.py.
    """
    desktop_df = pd.read_csv(short_csv_path)
    desktop_tileset = mozaic.TileSet()

    mozaic.populate_tiles(
        {"DAU": desktop_df},
        desktop_tileset,
        desktop_forecast_model,
        "2025-11-01",
        "2026-01-01",
    )

    desktop_mozaics: dict[str, Mozaic] = {}
    _ctry = defaultdict(lambda: defaultdict(mozaic.Mozaic))
    _pop = defaultdict(lambda: defaultdict(mozaic.Mozaic))

    mozaic.utils.curate_mozaics(
        {"DAU": desktop_df},
        desktop_tileset,
        desktop_forecast_model,
        desktop_mozaics,
        _ctry,
        _pop,
    )

    return desktop_mozaics["DAU"]
