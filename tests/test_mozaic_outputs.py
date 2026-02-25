"""
Unit tests for Mozaic output methods:
  _standard_df_to_forecast_df, _add_indicator_columns,
  to_forecast_df, to_granular_forecast_df, sum_tile_dfs.

Most of these are fast (mock model). The golden-data regression tests for
exact values live in test_core_additions.py.
"""
import numpy as np
import pandas as pd
import pytest

from mozaic.core import Mozaic
from mozaic.tile import sum_tile_dfs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_standard_df(n_actuals=10, n_forecast=5):
    """Build a minimal standard DataFrame with both actuals and forecast rows."""
    n = n_actuals + n_forecast
    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    actuals_vals = np.concatenate([np.ones(n_actuals) * 1000.0, np.full(n_forecast, np.nan)])
    forecast_vals = np.concatenate([np.full(n_actuals, np.nan), np.ones(n_forecast) * 1100.0])
    forecast_28ma = np.concatenate([np.full(n_actuals, np.nan), np.ones(n_forecast) * 1050.0])
    actuals_28ma = np.concatenate([np.ones(n_actuals) * 1000.0, np.full(n_forecast, np.nan)])

    return pd.DataFrame({
        "submission_date": dates,
        "actuals": actuals_vals,
        "actuals_detrended": actuals_vals,
        "forecast": forecast_vals,
        "forecast_28ma": forecast_28ma,
        "actuals_28ma": actuals_28ma,
    })


# ---------------------------------------------------------------------------
# _standard_df_to_forecast_df
# ---------------------------------------------------------------------------

class TestStandardDfToForecastDf:
    def test_output_columns(self):
        df = _make_standard_df()
        out = Mozaic._standard_df_to_forecast_df(df)
        assert list(out.columns) == ["target_date", "source", "value"]

    def test_source_values_are_actual_or_forecast(self):
        df = _make_standard_df()
        out = Mozaic._standard_df_to_forecast_df(df)
        assert set(out["source"].unique()).issubset({"actual", "forecast"})

    def test_actuals_come_before_forecasts(self):
        df = _make_standard_df(n_actuals=5, n_forecast=3)
        out = Mozaic._standard_df_to_forecast_df(df)
        actual_dates = out[out["source"] == "actual"]["target_date"]
        forecast_dates = out[out["source"] == "forecast"]["target_date"]
        assert actual_dates.max() < forecast_dates.min()

    def test_no_nan_values(self):
        df = _make_standard_df()
        out = Mozaic._standard_df_to_forecast_df(df)
        assert not out["value"].isna().any()

    def test_total_rows(self):
        df = _make_standard_df(n_actuals=10, n_forecast=5)
        out = Mozaic._standard_df_to_forecast_df(df)
        assert len(out) == 15

    def test_ma_true_uses_28ma_columns(self):
        df = _make_standard_df(n_actuals=10, n_forecast=5)
        out_no_ma = Mozaic._standard_df_to_forecast_df(df, ma=False)
        out_ma = Mozaic._standard_df_to_forecast_df(df, ma=True)
        # forecast values should differ (1100 vs 1050)
        fc_no_ma = out_no_ma[out_no_ma["source"] == "forecast"]["value"].iloc[0]
        fc_ma = out_ma[out_ma["source"] == "forecast"]["value"].iloc[0]
        assert fc_no_ma != fc_ma

    def test_ma_true_drops_nan_actuals(self):
        # The 28ma actuals column has NaN for early rows; those should be dropped
        df = _make_standard_df(n_actuals=10, n_forecast=5)
        out = Mozaic._standard_df_to_forecast_df(df, ma=True)
        assert not out["value"].isna().any()

    def test_returns_reset_index(self):
        df = _make_standard_df()
        out = Mozaic._standard_df_to_forecast_df(df)
        assert list(out.index) == list(range(len(out)))


# ---------------------------------------------------------------------------
# _add_indicator_columns
# ---------------------------------------------------------------------------

class TestAddIndicatorColumns:
    def test_output_columns(self):
        df = pd.DataFrame({
            "target_date": pd.date_range("2024-01-01", periods=3),
            "source": ["actual"] * 3,
            "value": [1.0, 2.0, 3.0],
        })
        out = Mozaic._add_indicator_columns("US", "win10", df)
        assert list(out.columns) == ["target_date", "country", "population", "source", "value"]

    def test_country_column_value(self):
        df = pd.DataFrame({
            "target_date": pd.date_range("2024-01-01", periods=2),
            "source": ["actual", "forecast"],
            "value": [1.0, 2.0],
        })
        out = Mozaic._add_indicator_columns("JP", "other", df)
        assert (out["country"] == "JP").all()

    def test_population_column_value(self):
        df = pd.DataFrame({
            "target_date": pd.date_range("2024-01-01", periods=2),
            "source": ["actual", "forecast"],
            "value": [1.0, 2.0],
        })
        out = Mozaic._add_indicator_columns("US", "win11", df)
        assert (out["population"] == "win11").all()

    def test_none_country(self):
        df = pd.DataFrame({
            "target_date": pd.date_range("2024-01-01", periods=2),
            "source": ["actual", "forecast"],
            "value": [1.0, 2.0],
        })
        out = Mozaic._add_indicator_columns("None", "None", df)
        assert (out["country"] == "None").all()


# ---------------------------------------------------------------------------
# to_df() on a Mozaic (mock model)
# ---------------------------------------------------------------------------

class TestMozaicToDF:
    @pytest.fixture(autouse=True)
    def _mozaic(self, make_mozaic):
        self.m = make_mozaic()

    def test_returns_dataframe(self):
        df = self.m.to_df()
        assert isinstance(df, pd.DataFrame)

    def test_has_actuals_and_forecast_columns(self):
        df = self.m.to_df()
        assert "actuals" in df.columns
        assert "forecast_detrended_raw" in df.columns

    def test_row_count(self):
        df = self.m.to_df()
        n_hist = len(self.m.historical_dates)
        n_fc = len(self.m.forecast_dates)
        assert len(df) == n_hist + n_fc


# ---------------------------------------------------------------------------
# to_forecast_df() on a Mozaic (mock model)
# ---------------------------------------------------------------------------

class TestToForecastDF:
    @pytest.fixture(autouse=True)
    def _mozaic(self, make_mozaic):
        self.m = make_mozaic(populations=("win10", "win11"))
        # Attach holiday impacts so 'forecast' column exists
        n = len(self.m.tiles[0].forecast_dates)
        for tile in self.m.tiles:
            tile.proportional_holiday_effects = pd.Series(
                np.zeros(n), index=tile.forecast_dates
            )
        self.m.aggregate_holiday_impacts_upward()

    def test_output_columns(self):
        out = self.m.to_forecast_df()
        assert list(out.columns) == ["target_date", "source", "value"]

    def test_source_values(self):
        out = self.m.to_forecast_df()
        assert set(out["source"].unique()).issubset({"actual", "forecast"})

    def test_no_nan_values(self):
        out = self.m.to_forecast_df()
        assert not out["value"].isna().any()

    def test_country_filter(self):
        out = self.m.to_forecast_df(country="US")
        assert len(out) > 0

    def test_population_filter(self):
        out_win10 = self.m.to_forecast_df(population="win10")
        out_win11 = self.m.to_forecast_df(population="win11")
        # Different populations → different values
        merged = out_win10.merge(out_win11, on=["target_date", "source"], suffixes=("_w10", "_w11"))
        assert not (merged["value_w10"] == merged["value_w11"]).all()

    def test_country_and_population_filter(self):
        out = self.m.to_forecast_df(country="US", population="win10")
        assert len(out) > 0
        assert list(out.columns) == ["target_date", "source", "value"]


# ---------------------------------------------------------------------------
# to_granular_forecast_df() on a Mozaic (mock model)
# ---------------------------------------------------------------------------

class TestToGranularForecastDF:
    @pytest.fixture(autouse=True)
    def _mozaic(self, make_mozaic):
        self.m = make_mozaic(populations=("win10", "win11"))
        n = len(self.m.tiles[0].forecast_dates)
        for tile in self.m.tiles:
            tile.proportional_holiday_effects = pd.Series(
                np.zeros(n), index=tile.forecast_dates
            )
        self.m.aggregate_holiday_impacts_upward()

    def test_output_columns(self):
        df = self.m.to_granular_forecast_df()
        assert list(df.columns) == ["target_date", "country", "population", "source", "value"]

    def test_contains_none_country_aggregate(self):
        df = self.m.to_granular_forecast_df()
        assert "None" in df["country"].values

    def test_contains_none_population_aggregate(self):
        df = self.m.to_granular_forecast_df()
        assert "None" in df["population"].values

    def test_sorted_by_target_date_country_population(self):
        df = self.m.to_granular_forecast_df()
        sorted_df = df.sort_values(["target_date", "country", "population"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(df, sorted_df)

    def test_row_count(self):
        df = self.m.to_granular_forecast_df()
        n_countries = len(self.m.get_countries())  # 1 country (US)
        n_pops = len(self.m.get_populations())      # 2 populations
        expected_combos = (n_countries + 1) * (n_pops + 1)
        actual_combos = df.groupby(["country", "population"]).ngroups
        assert actual_combos == expected_combos

    def test_no_nan_values(self):
        df = self.m.to_granular_forecast_df()
        assert not df["value"].isna().any()
