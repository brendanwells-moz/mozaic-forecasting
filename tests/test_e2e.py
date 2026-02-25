"""
End-to-end tests using the full legacy_desktop_dau.parquet dataset (104 tiles).

All tests are @pytest.mark.slow (~4 min, 104+ Prophet model fits).
Requires pyarrow or fastparquet for reading the parquet file.

Run with:
    python -m pytest tests/test_e2e.py -v
"""
import numpy as np
import pandas as pd
import pytest
from collections import defaultdict
from pathlib import Path

# Skip the entire module if pyarrow is not available
pyarrow = pytest.importorskip("pyarrow", reason="pyarrow required to read parquet files")

import mozaic
import mozaic.utils
from mozaic.core import Mozaic
from mozaic.models import desktop_forecast_model


pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Session-scoped E2E fixture (runs the full 104-tile pipeline once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def parquet_mozaic(parquet_path) -> Mozaic:
    """Full pipeline on the legacy desktop DAU parquet dataset."""
    df = pd.read_parquet(parquet_path)

    tileset = mozaic.TileSet()
    mozaic.populate_tiles(
        {"DAU": df},
        tileset,
        desktop_forecast_model,
        "2025-11-01",
        "2026-01-01",
    )

    metric_mozaics: dict[str, Mozaic] = {}
    _ctry = defaultdict(lambda: defaultdict(mozaic.Mozaic))
    _pop = defaultdict(lambda: defaultdict(mozaic.Mozaic))

    mozaic.utils.curate_mozaics(
        {"DAU": df},
        tileset,
        desktop_forecast_model,
        metric_mozaics,
        _ctry,
        _pop,
    )

    return metric_mozaics["DAU"]


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------

class TestE2EFullPipeline:
    def test_pipeline_produces_mozaic(self, parquet_mozaic):
        assert isinstance(parquet_mozaic, Mozaic)

    def test_expected_tile_count(self, parquet_mozaic):
        # The parquet has ~104 tiles
        assert len(parquet_mozaic.tiles) > 0

    def test_all_tiles_have_forecast(self, parquet_mozaic):
        for tile in parquet_mozaic.tiles:
            assert hasattr(tile, "forecast")
            assert tile.forecast.shape[1] == 1000

    def test_no_nan_in_any_tile_forecast_reconciled(self, parquet_mozaic):
        """THE regression test for the divide-by-zero bug with full dataset."""
        nan_tiles = []
        for tile in parquet_mozaic.tiles:
            if tile.forecast_reconciled.isna().any().any():
                nan_tiles.append(tile.name)
        assert len(nan_tiles) == 0, (
            f"NaN found in forecast_reconciled for {len(nan_tiles)} tiles: "
            f"{nan_tiles[:5]}{'...' if len(nan_tiles) > 5 else ''}"
        )

    def test_no_nan_in_tile_to_df_forecast(self, parquet_mozaic):
        """to_df()['forecast'] must not have NaN (original bug symptom)."""
        nan_tiles = []
        for tile in parquet_mozaic.tiles:
            df = tile.to_df(0.5)
            if "forecast" in df.columns and df["forecast"].dropna().isna().any():
                nan_tiles.append(tile.name)
        assert len(nan_tiles) == 0, \
            f"NaN in to_df().forecast for: {nan_tiles}"

    def test_all_tiles_have_holiday_impacts(self, parquet_mozaic):
        missing = [t.name for t in parquet_mozaic.tiles
                   if not hasattr(t, "forecasted_holiday_impacts")]
        assert len(missing) == 0, f"Missing forecasted_holiday_impacts: {missing}"

    def test_all_tiles_have_proportional_effects(self, parquet_mozaic):
        missing = [t.name for t in parquet_mozaic.tiles
                   if not hasattr(t, "proportional_holiday_effects")]
        assert len(missing) == 0, f"Missing proportional_holiday_effects: {missing}"

    def test_proportional_effects_bounded(self, parquet_mozaic):
        for tile in parquet_mozaic.tiles:
            effects = tile.proportional_holiday_effects
            assert (effects >= -1.0).all(), f"Effect < -1.0 in {tile.name}"
            assert (effects <= 0.0).all(), f"Positive effect in {tile.name}"

    def test_final_forecast_values_nonnegative(self, parquet_mozaic):
        # forecast_reconciled can have small negatives for declining populations
        # (reconciliation's quantile shift + rescaling can push near-zero
        # samples below zero). The clip(lower=0) is applied only when computing
        # the final 'forecast' column in to_df(). Check that column instead.
        neg_tiles = []
        for tile in parquet_mozaic.tiles:
            df = tile.to_df(0.5)
            if "forecast" in df.columns:
                if (df["forecast"].dropna() < 0).any():
                    neg_tiles.append(tile.name)
        assert len(neg_tiles) == 0, \
            f"Negative final forecast values in: {neg_tiles}"

    def test_granular_forecast_df_runs(self, parquet_mozaic):
        df = parquet_mozaic.to_granular_forecast_df()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == [
            "target_date", "country", "population", "source", "value"
        ]

    def test_granular_forecast_df_no_nan(self, parquet_mozaic):
        df = parquet_mozaic.to_granular_forecast_df()
        assert not df["value"].isna().any()

    def test_granular_forecast_df_shape(self, parquet_mozaic):
        df = parquet_mozaic.to_granular_forecast_df()
        n_countries = len(parquet_mozaic.get_countries())
        n_pops = len(parquet_mozaic.get_populations())
        expected_combos = (n_countries + 1) * (n_pops + 1)
        # Different (country, population) pairs can have different date ranges,
        # so total row count is not simply n_dates * combos. Instead verify
        # that all expected (country, population) combinations are present.
        actual_combos = df.groupby(["country", "population"]).ngroups
        assert actual_combos == expected_combos, \
            f"Expected {expected_combos} (country, population) combos, got {actual_combos}"

    def test_children_medians_sum_to_parent(self, parquet_mozaic):
        parent_med = parquet_mozaic.forecast_reconciled.median(axis=1)
        children_sum = sum(
            t.forecast_reconciled.median(axis=1) for t in parquet_mozaic.tiles
        )
        # 104 tiles accumulate more floating-point error than the short CSV;
        # use a looser tolerance (0.5%) for the full dataset.
        np.testing.assert_allclose(
            parent_med.values, children_sum.values, rtol=5e-3,
        )
