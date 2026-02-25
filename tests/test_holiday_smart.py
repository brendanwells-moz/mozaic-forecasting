"""
Unit tests for mozaic/holiday_smart.py: get_calendar() and detrend().
"""
import numpy as np
import pandas as pd
import pytest

from mozaic.holiday_smart import get_calendar, detrend


# ---------------------------------------------------------------------------
# get_calendar() tests
# ---------------------------------------------------------------------------

class TestGetCalendarColumns:
    def test_returns_dataframe(self):
        df = get_calendar("US", [2024])
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        df = get_calendar("US", [2024])
        assert set(df.columns) == {"submission_date", "holiday", "country"}

    def test_submission_date_is_datetime(self):
        df = get_calendar("US", [2024])
        assert pd.api.types.is_datetime64_any_dtype(df["submission_date"])

    def test_sorted_by_submission_date(self):
        df = get_calendar("US", [2024])
        assert df["submission_date"].is_monotonic_increasing

    def test_country_column_matches_input(self):
        df = get_calendar("JP", [2024])
        assert (df["country"] == "JP").all()


class TestUSHolidays:
    def setup_method(self):
        self.df = get_calendar("US", [2024])
        self.names = self.df["holiday"].str.lower()

    def test_new_years_day_present(self):
        assert self.names.str.contains("new year").any()

    def test_independence_day_present(self):
        assert self.names.str.contains("independence").any()

    def test_thanksgiving_present(self):
        assert self.names.str.contains("thanksgiving").any()

    def test_christmas_present(self):
        assert self.names.str.contains("christmas").any()

    def test_holiday_names_prefixed_with_country(self):
        # All holiday names should start with "US "
        assert self.df["holiday"].str.startswith("US ").all()


class TestGlobalHolidays:
    def test_christmas_eve_present(self):
        df = get_calendar("US", [2024])
        assert df["holiday"].str.contains("Christmas Eve").any()

    def test_new_years_eve_present(self):
        df = get_calendar("FR", [2024])
        assert df["holiday"].str.contains("New Year's Eve").any()

    def test_new_years_day_global_present(self):
        df = get_calendar("DE", [2024])
        assert df["holiday"].str.contains("New Year's Day").any()

    def test_global_holidays_appear_for_multiple_countries(self):
        for country in ["US", "JP", "FR"]:
            df = get_calendar(country, [2024])
            assert df["holiday"].str.contains("Christmas Eve").any(), \
                f"Christmas Eve missing for {country}"


class TestMozillaHolidays:
    def test_mozilla_holidays_present_when_2019_in_years(self):
        df = get_calendar("US", [2019])
        assert df["holiday"].str.contains("Data Loss").any()

    def test_mozilla_holidays_absent_when_2019_not_in_years(self):
        df = get_calendar("US", [2024])
        assert not df["holiday"].str.contains("Data Loss").any()

    def test_mozilla_holidays_present_with_range_including_2019(self):
        df = get_calendar("US", [2018, 2019, 2020])
        assert df["holiday"].str.contains("Data Loss").any()


class TestPaschalCycleExclusions:
    def test_jp_excludes_easter(self):
        df = get_calendar("JP", [2024])
        # Easter-related holidays should not be in JP calendar
        assert not df["holiday"].str.contains("Easter Sunday").any()

    def test_jp_excludes_good_friday(self):
        df = get_calendar("JP", [2024])
        assert not df["holiday"].str.contains("Good Friday").any()

    def test_in_excludes_custom_paschal_cycle(self):
        # India's base holidays library includes Good Friday natively.
        # What's excluded is the *custom* ChristianHolidays additions (Mardi
        # Gras, Corpus Christi, Ascension Day, etc.).
        df = get_calendar("IN", [2024])
        assert not df["holiday"].str.contains("Mardi Gras").any()
        assert not df["holiday"].str.contains("Corpus Christi").any()

    def test_cn_excludes_paschal_cycle(self):
        df = get_calendar("CN", [2024])
        assert not df["holiday"].str.contains("Easter Sunday").any()
        assert not df["holiday"].str.contains("Mardi Gras").any()

    def test_us_includes_paschal_cycle(self):
        df = get_calendar("US", [2024])
        assert df["holiday"].str.contains("Good Friday").any()


class TestRussiaOrthodoxEaster:
    def test_ru_includes_christian_holidays(self):
        df = get_calendar("RU", [2024])
        # Russia uses Orthodox calendar - should have Easter-cycle holidays
        assert df["holiday"].str.contains("Easter Sunday").any()

    def test_ru_orthodox_easter_differs_from_western(self):
        # Orthodox Easter is typically different from Western Easter
        ru_df = get_calendar("RU", [2024])
        us_df = get_calendar("US", [2024])

        ru_easter = ru_df[ru_df["holiday"].str.contains("Easter Sunday")]["submission_date"]
        us_easter = us_df[us_df["holiday"].str.contains("Easter Sunday")]["submission_date"]

        assert not ru_easter.equals(us_easter), "RU and US should have different Easter dates"


class TestCountrySpecificHolidays:
    def test_argentina_dia_del_maestro(self):
        df = get_calendar("AR", [2024])
        assert df["holiday"].str.contains("Maestro").any()

    def test_brazil_dia_do_professor(self):
        df = get_calendar("BR", [2024])
        assert df["holiday"].str.contains("Professor").any()

    def test_canada_thanksgiving(self):
        df = get_calendar("CA", [2024])
        assert df["holiday"].str.contains("Thanksgiving").any()

    def test_iran_blackout_2025(self):
        df = get_calendar("IR", [2025])
        blackout = df[df["holiday"].str.contains("Blackout", case=False)]
        assert len(blackout) > 0

    def test_iran_blackout_2026(self):
        df = get_calendar("IR", [2026])
        blackout = df[df["holiday"].str.contains("Blackout", case=False)]
        assert len(blackout) > 0

    def test_iran_blackout_not_in_2024(self):
        df = get_calendar("IR", [2024])
        blackout = df[df["holiday"].str.contains("Blackout", case=False)]
        assert len(blackout) == 0

    def test_mexico_dia_del_maestro(self):
        df = get_calendar("MX", [2024])
        assert df["holiday"].str.contains("Maestro").any()


class TestROWHolidays:
    def test_row_uses_us_holidays(self):
        row_df = get_calendar("ROW", [2024])
        us_df = get_calendar("US", [2024])
        # Independence Day should be in ROW (via US holidays)
        assert row_df["holiday"].str.contains("Independence").any()

    def test_row_has_workers_day(self):
        df = get_calendar("ROW", [2024])
        assert df["holiday"].str.contains("Workers").any()


class TestSplitConcurrentHolidays:
    def test_split_false_default(self):
        # Default is False: concurrent holidays stay as a single row with semicolons
        df_nosplit = get_calendar("US", [2024], split_concurrent_holidays=False)
        df_split = get_calendar("US", [2024], split_concurrent_holidays=True)
        # Split should have >= as many rows (splitting always produces same or more)
        assert len(df_split) >= len(df_nosplit)

    def test_split_true_no_semicolons_in_holiday_names(self):
        df = get_calendar("US", [2024], split_concurrent_holidays=True)
        # No semicolons should remain in holiday column when split
        assert not df["holiday"].str.contains(";").any()


class TestAdditionalHolidays:
    def test_additional_holidays_parameter_works(self):
        import holidays as hol_lib
        from mozaic.holiday_smart import DesktopBugs

        df_without = get_calendar("US", [2025])
        df_with = get_calendar("US", [2025], additional_holidays=[DesktopBugs])

        # DesktopBugs adds telemetry drop in 2025
        assert len(df_with) >= len(df_without)
        assert df_with["holiday"].str.contains("Telemetry").any()

    def test_desktop_bugs_not_in_base_us(self):
        df = get_calendar("US", [2025])
        assert not df["holiday"].str.contains("Telemetry").any()


# ---------------------------------------------------------------------------
# detrend() tests
# ---------------------------------------------------------------------------

def _make_detrend_inputs(n=60, start="2024-06-01", base=1000.0, seed=42):
    """Helper: generate dates, values and a non-empty holiday calendar."""
    dates = pd.date_range(start, periods=n, freq="D")
    years = list(dates.year.unique())
    rng = np.random.RandomState(seed)
    y = pd.Series(base + rng.randn(n) * (base * 0.02))
    holiday_df = get_calendar("US", years)
    return dates, y, holiday_df


class TestDetrend:
    def test_returns_series(self):
        dates, y, hdf = _make_detrend_inputs()
        result = detrend(dates=dates, y=y, holiday_df=hdf)
        assert isinstance(result, pd.Series)

    def test_returns_correct_length(self):
        dates, y, hdf = _make_detrend_inputs(n=60)
        result = detrend(dates=dates, y=y, holiday_df=hdf)
        assert len(result) == 60

    def test_short_series_returns_observed_values(self):
        # With n < 21, all values fall into the early-point branch → observed
        dates, y, hdf = _make_detrend_inputs(n=15)
        result = detrend(dates=dates, y=y, holiday_df=hdf)
        np.testing.assert_array_equal(result.values, y.values)

    def test_early_dates_unchanged(self):
        # First 21 rows (idx 0-20) should always be the observed values
        dates, y, hdf = _make_detrend_inputs(n=60)
        result = detrend(dates=dates, y=y, holiday_df=hdf)
        np.testing.assert_array_equal(result.values[:21], y.values[:21])

    def test_non_holiday_data_mostly_unchanged(self):
        # Generate flat data with no holiday dips — detrend should not alter it
        dates = pd.date_range("2024-06-01", periods=60, freq="D")
        y = pd.Series(np.ones(60) * 1000.0)
        hdf = get_calendar("US", [2024])
        result = detrend(dates=dates, y=y, holiday_df=hdf)
        # Values should be very close to 1000 (no dips to correct)
        assert np.allclose(result.values, y.values, rtol=0.05)

    def test_all_nan_input_does_not_crash(self):
        dates = pd.date_range("2024-06-01", periods=30, freq="D")
        y = pd.Series([np.nan] * 30)
        hdf = get_calendar("US", [2024])
        result = detrend(dates=dates, y=y, holiday_df=hdf)
        # Should return 30 values, all NaN
        assert len(result) == 30
        assert result.isna().all()

    def test_numeric_output(self):
        dates, y, hdf = _make_detrend_inputs()
        result = detrend(dates=dates, y=y, holiday_df=hdf)
        assert pd.api.types.is_numeric_dtype(result)

    def test_holiday_dip_smoothed_upward(self):
        # Create synthetic data with a large dip on July 4, 2024
        dates = pd.date_range("2024-05-01", periods=90, freq="D")
        y = pd.Series(np.ones(90) * 1000.0)
        # Insert a significant dip on July 4 (day 64)
        july4_idx = (dates == "2024-07-04").argmax()
        y.iloc[july4_idx] = 600.0  # 40% dip

        hdf = get_calendar("US", [2024])
        result = detrend(dates=dates, y=y, holiday_df=hdf)

        # Detrended value on July 4 should be higher than observed (smoothed)
        assert result.iloc[july4_idx] >= y.iloc[july4_idx]
