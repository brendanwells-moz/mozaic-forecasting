"""
Unit tests for mozaic/tile.py: Tile construction and to_df().
Uses the mock forecast model from conftest.py (fast, no Prophet).
"""
import numpy as np
import pandas as pd
import pytest

from mozaic.tile import Tile, sum_tile_dfs


# ---------------------------------------------------------------------------
# Tile construction
# ---------------------------------------------------------------------------

class TestTileConstruction:
    @pytest.fixture(autouse=True)
    def _tile(self, make_tile):
        self.tile = make_tile()

    def test_name_format(self):
        assert self.tile.name == "DAU | US | win10"

    def test_forecast_dates_is_daterange(self):
        assert isinstance(self.tile.forecast_dates, pd.DatetimeIndex)

    def test_forecast_dates_start(self):
        assert self.tile.forecast_dates[0] == pd.Timestamp("2025-01-15")

    def test_forecast_dates_end(self):
        assert self.tile.forecast_dates[-1] == pd.Timestamp("2025-01-28")

    def test_calendar_years_covers_historical(self):
        # Historical data starts 2024-11-16, so 2024 must be in calendar_years
        assert 2024 in self.tile.calendar_years

    def test_calendar_years_covers_forecast(self):
        # Forecast end is 2025-01-28, so 2025 must be in calendar_years
        assert 2025 in self.tile.calendar_years

    def test_holiday_calendar_is_dataframe(self):
        assert isinstance(self.tile.holiday_calendar, pd.DataFrame)

    def test_holiday_calendar_has_required_columns(self):
        assert {"submission_date", "holiday", "country"}.issubset(
            self.tile.holiday_calendar.columns
        )

    def test_historical_dates_is_datetime_index(self):
        assert pd.api.types.is_datetime64_any_dtype(self.tile.historical_dates)

    def test_raw_historical_data_length(self):
        assert len(self.tile.raw_historical_data) == len(self.tile.historical_dates)

    def test_holiday_detrended_length_matches_historical(self):
        assert len(self.tile.holiday_detrended_historical_data) == len(
            self.tile.historical_dates
        )

    def test_forecast_shape(self):
        n_forecast = len(self.tile.forecast_dates)
        assert self.tile.forecast.shape == (n_forecast, 1000)

    def test_forecast_reconciled_initial_equals_forecast(self):
        pd.testing.assert_frame_equal(
            self.tile.forecast_reconciled, self.tile.forecast
        )

    def test_forecast_reconciled_is_deep_copy(self):
        # Modifying forecast_reconciled should not affect forecast
        original_val = self.tile.forecast.iloc[0, 0]
        self.tile.forecast_reconciled.iloc[0, 0] = -9999.0
        assert self.tile.forecast.iloc[0, 0] == original_val

    def test_mozaic_initially_none(self):
        assert self.tile.mozaic is None

    def test_forecast_nonnegative(self):
        assert (self.tile.forecast >= 0).all().all()


class TestTileWithDifferentCountry:
    def test_japan_tile_builds_successfully(self, make_tile):
        tile = make_tile(country="JP")
        assert tile.country == "JP"

    def test_canada_tile_builds_successfully(self, make_tile):
        tile = make_tile(country="CA")
        assert tile.country == "CA"


# ---------------------------------------------------------------------------
# to_df()
# ---------------------------------------------------------------------------

class TestTileToDF:
    @pytest.fixture(autouse=True)
    def _tile(self, make_tile):
        self.tile = make_tile()

    def test_returns_dataframe(self):
        df = self.tile.to_df()
        assert isinstance(df, pd.DataFrame)

    def test_has_submission_date_column(self):
        df = self.tile.to_df()
        assert "submission_date" in df.columns

    def test_has_actuals_columns(self):
        df = self.tile.to_df()
        assert "actuals" in df.columns
        assert "actuals_detrended" in df.columns

    def test_has_forecast_detrended_raw_column(self):
        df = self.tile.to_df()
        assert "forecast_detrended_raw" in df.columns

    def test_has_forecast_detrended_column(self):
        # forecast_reconciled is set at construction → present
        df = self.tile.to_df()
        assert "forecast_detrended" in df.columns

    def test_no_forecast_column_before_holiday_impacts(self):
        # 'forecast' column requires forecasted_holiday_impacts
        df = self.tile.to_df()
        assert "forecast" not in df.columns

    def test_has_28ma_forecast_columns(self):
        df = self.tile.to_df()
        assert "forecast_detrended_28ma" in df.columns

    def test_has_28ma_actuals_columns(self):
        df = self.tile.to_df()
        assert "actuals_28ma" in df.columns

    def test_actuals_not_nan(self):
        df = self.tile.to_df()
        # Actuals for historical period should not be NaN
        actuals = df.dropna(subset=["actuals"])
        assert len(actuals) == len(self.tile.historical_dates)

    def test_forecast_detrended_raw_not_nan_in_forecast_period(self):
        df = self.tile.to_df()
        forecast_rows = df[~df["forecast_detrended_raw"].isna()]
        assert len(forecast_rows) == len(self.tile.forecast_dates)

    def test_actuals_and_forecasts_do_not_overlap(self):
        df = self.tile.to_df()
        # Rows with actuals should not have forecasts (outer merge)
        both = df.dropna(subset=["actuals", "forecast_detrended_raw"])
        assert len(both) == 0

    def test_row_count_is_historical_plus_forecast(self):
        df = self.tile.to_df()
        expected = len(self.tile.historical_dates) + len(self.tile.forecast_dates)
        assert len(df) == expected

    def test_quantile_parameter_accepted(self):
        df_50 = self.tile.to_df(quantile=0.5)
        df_20 = self.tile.to_df(quantile=0.2)
        # Different quantiles → different forecast values
        forecast_col = "forecast_detrended_raw"
        assert not df_50[forecast_col].equals(df_20[forecast_col])

    def test_forecast_column_present_after_holiday_impacts(self, make_tile):
        tile = make_tile()
        n = len(tile.forecast_dates)
        # Manually attach holiday impacts (all zeros → no effect)
        tile.forecasted_holiday_impacts = tile.forecast_reconciled * 0.0
        df = tile.to_df()
        assert "forecast" in df.columns


# ---------------------------------------------------------------------------
# sum_tile_dfs()
# ---------------------------------------------------------------------------

class TestSumTileDFs:
    def _make_simple_df(self, offset=0.0, n=5):
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        return pd.DataFrame({
            "submission_date": dates,
            "actuals": np.ones(n) * (100.0 + offset),
            "actuals_detrended": np.ones(n) * (100.0 + offset),
        })

    def test_returns_dataframe(self):
        result = sum_tile_dfs([self._make_simple_df(), self._make_simple_df()])
        assert isinstance(result, pd.DataFrame)

    def test_sums_numeric_columns(self):
        df1 = self._make_simple_df(0)
        df2 = self._make_simple_df(0)
        result = sum_tile_dfs([df1, df2])
        np.testing.assert_allclose(result["actuals"].values, np.ones(5) * 200.0)

    def test_groups_by_submission_date(self):
        df1 = self._make_simple_df(0)
        df2 = self._make_simple_df(0)
        result = sum_tile_dfs([df1, df2])
        assert "submission_date" in result.columns
        assert len(result) == 5

    def test_all_nan_group_produces_nan_not_zero(self):
        # min_count=1 means all-NaN groups → NaN (not 0)
        df1 = pd.DataFrame({
            "submission_date": pd.date_range("2024-01-01", periods=2),
            "actuals": [np.nan, 100.0],
        })
        df2 = pd.DataFrame({
            "submission_date": pd.date_range("2024-01-01", periods=2),
            "actuals": [np.nan, 100.0],
        })
        result = sum_tile_dfs([df1, df2])
        assert np.isnan(result.loc[result["submission_date"] == "2024-01-01", "actuals"].values[0])
