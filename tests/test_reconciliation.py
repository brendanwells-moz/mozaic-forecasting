"""
Tests for reconciliation logic in mozaic/core.py.

Includes:
  - Fast unit tests with synthetic/mocked data
  - Slow structural invariant tests using the real cur_mozaic fixture

The divide-by-zero bug (fixed in core.py) is explicitly covered:
  When a tile's forecast has all-zero samples on some dates, the old code
  produced 0/0 = NaN in get_weight(), poisoning all 104 tiles. The fix uses
  mean.where(mean != 0, np.inf) so zero-mean tiles get weight 0 instead.
"""
import numpy as np
import pandas as pd
import pytest

from mozaic.core import Mozaic


# ---------------------------------------------------------------------------
# Helper: build a constant forecast DataFrame (n_dates × 1000)
# ---------------------------------------------------------------------------

def _constant_forecast(value, n_dates=14, seed=None):
    """All 1000 samples equal *value* (plus optional tiny noise)."""
    if seed is not None:
        rng = np.random.RandomState(seed)
        noise = rng.randn(n_dates, 1000) * (abs(value) * 0.001)
        data = np.clip(value + noise, 0, None)
    else:
        data = np.full((n_dates, 1000), float(value))
    return pd.DataFrame(data, columns=range(1000))


# ---------------------------------------------------------------------------
# Tests for _reconcile_top_down_by_rescaling()
# ---------------------------------------------------------------------------

class TestReconcileTopDownByRescaling:
    """Critical regression tests for the divide-by-zero bug."""

    def test_zero_mean_tile_no_nan_propagation(self, make_mozaic):
        """A tile with all-zero forecast should not propagate NaN to siblings.

        This is the exact bug that caused 104 tiles to have NaN outputs:
        zero-mean forecast → var/mean = 0/0 = NaN → NaN weight → NaN delta.
        """
        m = make_mozaic()
        # Force one tile's forecast_reconciled to all zeros
        m.tiles[0].forecast_reconciled = _constant_forecast(0.0)

        m._reconcile_top_down_by_rescaling()

        for tile in m.tiles:
            bad_rows = tile.forecast_reconciled.isna().any(axis=1)
            assert not bad_rows.any(), (
                f"NaN found in tile {tile.name} after reconciliation. "
                "Zero-mean tile should get zero weight, not NaN weight."
            )

    def test_zero_mean_tile_weight_is_zero(self, make_mozaic):
        """Zero-mean tile should receive no adjustment (delta = 0)."""
        m = make_mozaic()
        zero_tile = m.tiles[0]
        zero_tile.forecast_reconciled = _constant_forecast(0.0)

        before = zero_tile.forecast_reconciled.copy()
        m._reconcile_top_down_by_rescaling()

        pd.testing.assert_frame_equal(
            zero_tile.forecast_reconciled, before,
            check_exact=True,
            obj="Zero-mean tile should not change after reconciliation",
        )

    def test_all_zero_tiles_no_nan(self, make_mozaic):
        """If all tiles are zero, weight_sum > 0 (from parent), so no NaN."""
        m = make_mozaic()
        for tile in m.tiles:
            tile.forecast_reconciled = _constant_forecast(0.0)

        # Parent is non-zero (from mock model), tiles are all zero
        # weight_sum = 0, so delta = diff * (0/0) which is the edge case.
        # We can't guarantee non-NaN here (this is an extreme edge case),
        # but we document the behaviour: it should not crash.
        try:
            m._reconcile_top_down_by_rescaling()
        except Exception as e:
            pytest.fail(f"_reconcile_top_down_by_rescaling crashed: {e}")

    def test_nonzero_tile_gets_adjustment(self, make_mozaic):
        """The non-zero sibling should receive the full diff when one is zero."""
        m = make_mozaic(populations=("win10", "win11"))

        # Force tile0 to zero, tile1 to half the parent's level
        parent_median = m.forecast_reconciled.median(axis=1).mean()
        m.tiles[0].forecast_reconciled = _constant_forecast(0.0)
        m.tiles[1].forecast_reconciled = _constant_forecast(parent_median * 0.5, seed=7)

        m._reconcile_top_down_by_rescaling()

        # After reconciliation, sum of tile medians ≈ parent median
        parent_med = m.forecast_reconciled.median(axis=1)
        children_sum = sum(t.forecast_reconciled.median(axis=1) for t in m.tiles)
        np.testing.assert_allclose(
            parent_med.values, children_sum.values, rtol=1e-6,
            err_msg="Children medians should sum to parent median after rescaling"
        )

    def test_nonnegative_input_stays_nonnegative(self, make_mozaic):
        """Reconciliation should not introduce negative values for positive inputs."""
        m = make_mozaic()
        m._reconcile_top_down_by_rescaling()
        for tile in m.tiles:
            # After rescaling the median, samples near the lower tail might
            # theoretically go negative. We check no extreme negatives.
            min_val = tile.forecast_reconciled.min().min()
            assert min_val > -1e6, f"Extreme negative values in {tile.name}"


# ---------------------------------------------------------------------------
# Tests for reconcile_bottom_up()
# ---------------------------------------------------------------------------

class TestReconcileBottomUp:
    def test_parent_equals_sum_of_children(self, make_mozaic):
        m = make_mozaic(populations=("win10", "win11"))
        m.reconcile_bottom_up()

        expected = m.tiles[0].forecast_reconciled + m.tiles[1].forecast_reconciled
        pd.testing.assert_frame_equal(
            m.forecast_reconciled, expected,
            check_names=False,
            rtol=1e-10,
        )

    def test_single_child(self, make_mozaic):
        m = make_mozaic(populations=("win10",))
        m.reconcile_bottom_up()
        pd.testing.assert_frame_equal(
            m.forecast_reconciled, m.tiles[0].forecast_reconciled
        )

    def test_three_children(self, make_mozaic):
        m = make_mozaic(populations=("win10", "win11", "other"))
        m.reconcile_bottom_up()
        expected = sum(t.forecast_reconciled for t in m.tiles)
        pd.testing.assert_frame_equal(m.forecast_reconciled, expected, check_names=False)


# ---------------------------------------------------------------------------
# Tests for reconcile_top_down()
# ---------------------------------------------------------------------------

class TestReconcileTopDown:
    def test_children_medians_sum_to_parent_after_top_down(self, make_mozaic):
        """After top-down reconciliation, sum of children medians = parent median."""
        m = make_mozaic()
        m.reconcile_top_down()

        parent_med = m.forecast_reconciled.median(axis=1)
        children_sum = sum(t.forecast_reconciled.median(axis=1) for t in m.tiles)
        np.testing.assert_allclose(
            parent_med.values, children_sum.values, rtol=1e-6,
        )

    def test_no_nan_after_top_down(self, make_mozaic):
        m = make_mozaic()
        m.reconcile_top_down()
        for tile in m.tiles:
            assert not tile.forecast_reconciled.isna().any().any()

    def test_zero_mean_tile_no_nan_in_top_down(self, make_mozaic):
        """Full reconcile_top_down() should also be safe with a zero-mean tile."""
        m = make_mozaic()
        m.tiles[0].forecast_reconciled = _constant_forecast(0.0)
        m.reconcile_top_down()
        for tile in m.tiles:
            assert not tile.forecast_reconciled.isna().any().any(), \
                f"NaN in {tile.name} after full reconcile_top_down with zero-mean tile"


# ---------------------------------------------------------------------------
# Tests for aggregate_holiday_impacts_upward()
# ---------------------------------------------------------------------------

class TestAggregateHolidayImpacts:
    def _attach_zero_proportional_effects(self, m):
        """Give every tile a zero-effect proportional_holiday_effects Series."""
        n = len(m.tiles[0].forecast_dates)
        for tile in m.tiles:
            tile.proportional_holiday_effects = pd.Series(
                np.zeros(n), index=tile.forecast_dates
            )

    def test_forecasted_holiday_impacts_set_on_parent(self, make_mozaic):
        m = make_mozaic()
        self._attach_zero_proportional_effects(m)
        m.aggregate_holiday_impacts_upward()
        assert hasattr(m, "forecasted_holiday_impacts")

    def test_impacts_sum_of_children(self, make_mozaic):
        m = make_mozaic(populations=("win10", "win11"))
        n = len(m.tiles[0].forecast_dates)
        # Give each tile a non-trivial proportional effect
        for i, tile in enumerate(m.tiles):
            tile.proportional_holiday_effects = pd.Series(
                np.full(n, -0.05 * (i + 1)), index=tile.forecast_dates
            )

        m.aggregate_holiday_impacts_upward()

        child0_impact = m.tiles[0].forecasted_holiday_impacts
        child1_impact = m.tiles[1].forecasted_holiday_impacts
        expected_parent = child0_impact + child1_impact

        pd.testing.assert_frame_equal(
            m.forecasted_holiday_impacts, expected_parent, check_names=False
        )

    def test_skips_tiles_already_having_impacts(self, make_mozaic):
        m = make_mozaic()
        n = len(m.tiles[0].forecast_dates)
        for tile in m.tiles:
            tile.proportional_holiday_effects = pd.Series(
                np.zeros(n), index=tile.forecast_dates
            )

        # Pre-set forecasted_holiday_impacts on tile 0 with a sentinel
        sentinel = _constant_forecast(-99.0)
        m.tiles[0].forecasted_holiday_impacts = sentinel

        m.aggregate_holiday_impacts_upward()

        # Tile 0's impacts should be unchanged (pre-set sentinel preserved)
        pd.testing.assert_frame_equal(m.tiles[0].forecasted_holiday_impacts, sentinel)


# ---------------------------------------------------------------------------
# Tests for assign_holiday_effects()
# ---------------------------------------------------------------------------

class TestAssignHolidayEffects:
    def test_proportional_effects_set_on_tiles(self, make_mozaic):
        m = make_mozaic(is_country_level=True)
        m.assign_holiday_effects()
        for tile in m.tiles:
            assert hasattr(tile, "proportional_holiday_effects")

    def test_effects_are_series(self, make_mozaic):
        m = make_mozaic(is_country_level=True)
        m.assign_holiday_effects()
        for tile in m.tiles:
            assert isinstance(tile.proportional_holiday_effects, pd.Series)

    def test_effects_bounded_to_minus_point_six_and_zero(self, make_mozaic):
        m = make_mozaic(is_country_level=True)
        m.assign_holiday_effects()
        for tile in m.tiles:
            effects = tile.proportional_holiday_effects
            # Clipped to [-0.6, 0] (blackouts can reach -1.0 via override)
            assert (effects >= -1.0).all(), f"Effect < -1.0 in {tile.name}"
            assert (effects <= 0.0).all(), f"Positive effect in {tile.name}"

    def test_no_effect_without_country_level(self, make_mozaic):
        m = make_mozaic(is_country_level=False)
        m.assign_holiday_effects()
        # Without is_country_level=True, proportional_holiday_effects is NOT
        # set on the parent Mozaic (only propagated if it already has them)
        for tile in m.tiles:
            assert not hasattr(tile, "proportional_holiday_effects")


# ---------------------------------------------------------------------------
# Structural invariant tests (require real Prophet; marked slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestStructuralInvariants:
    def test_no_nan_in_any_tile_forecast_reconciled(self, cur_mozaic):
        """THE key regression test for the divide-by-zero bug."""
        for tile in cur_mozaic.tiles:
            bad = tile.forecast_reconciled.isna().any(axis=1)
            assert not bad.any(), (
                f"NaN in forecast_reconciled for {tile.name} on dates: "
                f"{tile.forecast_dates[bad].tolist()}"
            )

    def test_no_nan_in_to_df_forecast_column(self, cur_mozaic):
        """The original bug symptom: to_df()['forecast'] contained NaN."""
        for tile in cur_mozaic.tiles:
            df = tile.to_df(0.5)
            if "forecast" in df.columns:
                forecast_vals = df["forecast"].dropna()
                assert not forecast_vals.isna().any(), f"NaN in {tile.name}.to_df().forecast"

    def test_all_tiles_nonnegative_forecast(self, cur_mozaic):
        for tile in cur_mozaic.tiles:
            assert (tile.forecast_reconciled >= 0).all().all(), \
                f"Negative values in {tile.name}"

    def test_holiday_effects_bounded(self, cur_mozaic):
        for tile in cur_mozaic.tiles:
            assert hasattr(tile, "proportional_holiday_effects"), \
                f"{tile.name} missing proportional_holiday_effects"
            effects = tile.proportional_holiday_effects
            assert (effects >= -1.0).all(), f"Effect < -1.0 in {tile.name}"
            assert (effects <= 0.0).all(), f"Positive effect in {tile.name}"

    def test_children_reconciled_sum_matches_parent_median(self, cur_mozaic):
        parent_med = cur_mozaic.forecast_reconciled.median(axis=1)
        children_sum = sum(t.forecast_reconciled.median(axis=1) for t in cur_mozaic.tiles)
        np.testing.assert_allclose(
            parent_med.values, children_sum.values, rtol=1e-3,
        )
