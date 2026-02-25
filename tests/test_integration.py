"""
Integration tests using the real Prophet pipeline (short CSV dataset).

All tests are marked @pytest.mark.slow because they depend on the session-
scoped cur_mozaic fixture which runs ~8 Prophet model fits.

Run with:
    python -m pytest tests/test_integration.py -v
    python -m pytest -m slow -v
"""
import numpy as np
import pandas as pd
import pytest

from mozaic.core import Mozaic


pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Structural invariants — no NaN, correct shapes
# ---------------------------------------------------------------------------

class TestNoNaN:
    def test_no_nan_in_any_tile_forecast_reconciled(self, cur_mozaic):
        """THE regression test: zero-mean tiles must not propagate NaN."""
        for tile in cur_mozaic.tiles:
            bad = tile.forecast_reconciled.isna().any(axis=1)
            assert not bad.any(), (
                f"NaN in forecast_reconciled for {tile.name} on "
                f"{tile.forecast_dates[bad].tolist()}"
            )

    def test_no_nan_in_tile_to_df_forecast_column(self, cur_mozaic):
        """to_df()['forecast'] was the symptom of the divide-by-zero bug."""
        for tile in cur_mozaic.tiles:
            df = tile.to_df(0.5)
            assert "forecast" in df.columns, f"'forecast' column missing for {tile.name}"
            forecast_vals = df["forecast"].dropna()
            assert not forecast_vals.isna().any(), f"NaN in {tile.name}.to_df().forecast"

    def test_no_nan_in_mozaic_to_df_forecast_column(self, cur_mozaic):
        df = cur_mozaic.to_df(0.5)
        assert "forecast" in df.columns
        assert not df["forecast"].dropna().isna().any()


# ---------------------------------------------------------------------------
# Shape and type checks
# ---------------------------------------------------------------------------

class TestForecastShapes:
    def test_each_tile_has_1000_sample_columns(self, cur_mozaic):
        for tile in cur_mozaic.tiles:
            assert tile.forecast.shape[1] == 1000, \
                f"{tile.name} has {tile.forecast.shape[1]} sample columns"

    def test_each_tile_forecast_rows_match_forecast_dates(self, cur_mozaic):
        for tile in cur_mozaic.tiles:
            n_fc = len(tile.forecast_dates)
            assert tile.forecast.shape[0] == n_fc, \
                f"{tile.name}: forecast has {tile.forecast.shape[0]} rows, expected {n_fc}"

    def test_forecast_reconciled_shape_matches_forecast(self, cur_mozaic):
        for tile in cur_mozaic.tiles:
            assert tile.forecast_reconciled.shape == tile.forecast.shape


# ---------------------------------------------------------------------------
# Non-negativity
# ---------------------------------------------------------------------------

class TestNonNegativity:
    def test_all_tile_forecast_reconciled_nonnegative(self, cur_mozaic):
        for tile in cur_mozaic.tiles:
            assert (tile.forecast_reconciled >= 0).all().all(), \
                f"Negative values in {tile.name} forecast_reconciled"

    def test_mozaic_forecast_reconciled_nonnegative(self, cur_mozaic):
        assert (cur_mozaic.forecast_reconciled >= 0).all().all()


# ---------------------------------------------------------------------------
# Holiday impacts
# ---------------------------------------------------------------------------

class TestHolidayImpacts:
    def test_all_tiles_have_forecasted_holiday_impacts(self, cur_mozaic):
        for tile in cur_mozaic.tiles:
            assert hasattr(tile, "forecasted_holiday_impacts"), \
                f"{tile.name} missing forecasted_holiday_impacts"

    def test_all_tiles_have_proportional_holiday_effects(self, cur_mozaic):
        for tile in cur_mozaic.tiles:
            assert hasattr(tile, "proportional_holiday_effects"), \
                f"{tile.name} missing proportional_holiday_effects"

    def test_proportional_holiday_effects_bounded(self, cur_mozaic):
        for tile in cur_mozaic.tiles:
            effects = tile.proportional_holiday_effects
            assert (effects >= -1.0).all(), f"Effect < -1.0 in {tile.name}"
            assert (effects <= 0.0).all(), f"Positive effect in {tile.name}"

    def test_holiday_impacts_shape_matches_forecast(self, cur_mozaic):
        for tile in cur_mozaic.tiles:
            impacts = tile.forecasted_holiday_impacts
            assert impacts.shape == tile.forecast_reconciled.shape, \
                f"Impact shape mismatch for {tile.name}"


# ---------------------------------------------------------------------------
# Reconciliation consistency
# ---------------------------------------------------------------------------

class TestReconciliationConsistency:
    def test_children_medians_sum_to_parent(self, cur_mozaic):
        parent_med = cur_mozaic.forecast_reconciled.median(axis=1)
        children_sum = sum(
            t.forecast_reconciled.median(axis=1) for t in cur_mozaic.tiles
        )
        np.testing.assert_allclose(
            parent_med.values, children_sum.values, rtol=1e-3,
            err_msg="Sum of tile medians should match parent median after reconciliation",
        )

    def test_forecast_shape_consistent_across_tiles(self, cur_mozaic):
        shapes = {tile.forecast.shape for tile in cur_mozaic.tiles}
        n_dates = len(cur_mozaic.tiles[0].forecast_dates)
        assert all(s == (n_dates, 1000) for s in shapes)


# ---------------------------------------------------------------------------
# Output method smoke tests
# ---------------------------------------------------------------------------

class TestOutputMethods:
    def test_to_df_runs_without_error(self, cur_mozaic):
        df = cur_mozaic.to_df(0.5)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_to_df_columns(self, cur_mozaic):
        df = cur_mozaic.to_df(0.5)
        assert "actuals" in df.columns
        assert "forecast" in df.columns

    def test_to_forecast_df_no_country_no_population(self, cur_mozaic):
        df = cur_mozaic.to_forecast_df()
        assert list(df.columns) == ["target_date", "source", "value"]
        assert not df["value"].isna().any()

    def test_to_forecast_df_with_country(self, cur_mozaic):
        country = next(iter(cur_mozaic.get_countries()))
        df = cur_mozaic.to_forecast_df(country=country)
        assert list(df.columns) == ["target_date", "source", "value"]

    def test_to_forecast_df_with_population(self, cur_mozaic):
        population = next(iter(cur_mozaic.get_populations()))
        df = cur_mozaic.to_forecast_df(population=population)
        assert list(df.columns) == ["target_date", "source", "value"]

    def test_to_granular_forecast_df_runs(self, cur_mozaic):
        df = cur_mozaic.to_granular_forecast_df()
        assert isinstance(df, pd.DataFrame)

    def test_to_granular_forecast_df_columns(self, cur_mozaic):
        df = cur_mozaic.to_granular_forecast_df()
        assert list(df.columns) == [
            "target_date", "country", "population", "source", "value"
        ]

    def test_to_granular_forecast_df_no_nan(self, cur_mozaic):
        df = cur_mozaic.to_granular_forecast_df()
        assert not df["value"].isna().any()

    def test_to_granular_forecast_df_shape(self, cur_mozaic):
        df = cur_mozaic.to_granular_forecast_df()
        n_countries = len(cur_mozaic.get_countries())
        n_pops = len(cur_mozaic.get_populations())
        expected_combos = (n_countries + 1) * (n_pops + 1)
        actual_combos = df.groupby(["country", "population"]).ngroups
        assert actual_combos == expected_combos

    def test_quantile_parameter_works(self, cur_mozaic):
        df_50 = cur_mozaic.to_forecast_df(quantile=0.5)
        df_20 = cur_mozaic.to_forecast_df(quantile=0.2)
        fc_50 = df_50[df_50["source"] == "forecast"]["value"]
        fc_20 = df_20[df_20["source"] == "forecast"]["value"]
        # Different quantiles should produce different values
        assert not fc_50.reset_index(drop=True).equals(fc_20.reset_index(drop=True))
