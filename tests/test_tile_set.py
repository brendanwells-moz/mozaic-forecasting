"""
Unit tests for mozaic/tile_set.py: TileSet.add(), fetch(), levels().
"""
import pytest
from unittest.mock import MagicMock

from mozaic.tile_set import TileSet


def _mock_tile(metric, country, population):
    """Create a simple mock object with tile-like attributes."""
    t = MagicMock()
    t.metric = metric
    t.country = country
    t.population = population
    return t


# ---------------------------------------------------------------------------
# add() and basic retrieval
# ---------------------------------------------------------------------------

class TestAdd:
    def test_add_stores_tile(self):
        ts = TileSet()
        tile = _mock_tile("DAU", "US", "win10")
        ts.add(tile)
        assert ts.tiles["DAU"]["US"]["win10"] is tile

    def test_add_multiple_tiles(self):
        ts = TileSet()
        t1 = _mock_tile("DAU", "US", "win10")
        t2 = _mock_tile("DAU", "US", "win11")
        t3 = _mock_tile("DAU", "JP", "win10")
        ts.add(t1)
        ts.add(t2)
        ts.add(t3)
        assert ts.tiles["DAU"]["US"]["win10"] is t1
        assert ts.tiles["DAU"]["US"]["win11"] is t2
        assert ts.tiles["DAU"]["JP"]["win10"] is t3

    def test_add_overwrites_existing(self):
        ts = TileSet()
        t1 = _mock_tile("DAU", "US", "win10")
        t2 = _mock_tile("DAU", "US", "win10")
        ts.add(t1)
        ts.add(t2)
        assert ts.tiles["DAU"]["US"]["win10"] is t2


# ---------------------------------------------------------------------------
# fetch()
# ---------------------------------------------------------------------------

class TestFetch:
    def setup_method(self):
        self.ts = TileSet()
        self.t1 = _mock_tile("DAU", "US", "win10")
        self.t2 = _mock_tile("DAU", "US", "win11")
        self.t3 = _mock_tile("DAU", "JP", "win10")
        self.t4 = _mock_tile("MAU", "US", "win10")
        for t in [self.t1, self.t2, self.t3, self.t4]:
            self.ts.add(t)

    def test_fetch_exact_returns_single_tile(self):
        result = self.ts.fetch("DAU", "US", "win10")
        assert result == [self.t1]

    def test_fetch_metric_only_returns_all_dau(self):
        result = self.ts.fetch("DAU")
        assert set(result) == {self.t1, self.t2, self.t3}

    def test_fetch_metric_country_returns_country_tiles(self):
        result = self.ts.fetch("DAU", "US")
        assert set(result) == {self.t1, self.t2}

    def test_fetch_metric_population_returns_population_tiles(self):
        result = self.ts.fetch("DAU", population="win10")
        assert set(result) == {self.t1, self.t3}

    def test_fetch_wrong_metric_returns_empty(self):
        result = self.ts.fetch("FAKE")
        assert result == []

    def test_fetch_wrong_country_returns_empty(self):
        result = self.ts.fetch("DAU", "FR")
        assert result == []

    def test_fetch_wrong_population_returns_empty(self):
        result = self.ts.fetch("DAU", population="other")
        assert result == []

    def test_fetch_different_metric_isolates(self):
        # MAU tiles should not appear in DAU fetch
        result = self.ts.fetch("DAU")
        assert self.t4 not in result

    def test_fetch_mau_returns_mau_tiles(self):
        result = self.ts.fetch("MAU")
        assert result == [self.t4]


class TestFetchSubstringBehavior:
    """Document the (population or p) in p behavior.

    The fetch() condition is: (population or p) in p
    This means:
      - population=None  → (None or p) = p → p in p → always True
      - population="win" → "win" in "win10" → True (substring match!)
      - population="win10" → "win10" in "win10" → True
      - population="win10" → "win10" in "win10_win11" → True (surprise!)
    """
    def setup_method(self):
        self.ts = TileSet()
        self.t_win10 = _mock_tile("DAU", "US", "win10")
        self.t_win11 = _mock_tile("DAU", "US", "win11")
        self.t_combined = _mock_tile("DAU", "US", "win10_win11")
        for t in [self.t_win10, self.t_win11, self.t_combined]:
            self.ts.add(t)

    def test_exact_win10_also_matches_win10_win11(self):
        # "win10" is a substring of "win10_win11", so both are returned
        result = self.ts.fetch("DAU", population="win10")
        assert self.t_win10 in result
        assert self.t_combined in result

    def test_win11_also_matches_win10_win11(self):
        result = self.ts.fetch("DAU", population="win11")
        assert self.t_win11 in result
        assert self.t_combined in result

    def test_prefix_matches_multiple(self):
        # "win" is a prefix of all three → all three returned
        result = self.ts.fetch("DAU", population="win")
        assert len(result) == 3


# ---------------------------------------------------------------------------
# levels()
# ---------------------------------------------------------------------------

class TestLevels:
    def setup_method(self):
        self.ts = TileSet()
        for metric, country, population in [
            ("DAU", "US", "win10"),
            ("DAU", "US", "win11"),
            ("DAU", "JP", "win10"),
            ("MAU", "US", "win10"),
        ]:
            self.ts.add(_mock_tile(metric, country, population))

    def test_levels_metrics(self):
        levels = self.ts.levels("DAU")
        assert set(levels.metrics) == {"DAU"}

    def test_levels_countries(self):
        levels = self.ts.levels("DAU")
        assert set(levels.countries) == {"US", "JP"}

    def test_levels_populations(self):
        levels = self.ts.levels("DAU")
        assert set(levels.populations) == {"win10", "win11"}

    def test_levels_with_country_filter(self):
        levels = self.ts.levels("DAU", country="US")
        assert set(levels.countries) == {"US"}
        assert set(levels.populations) == {"win10", "win11"}

    def test_levels_with_population_filter(self):
        levels = self.ts.levels("DAU", population="win10")
        assert set(levels.countries) == {"US", "JP"}

    def test_levels_returns_namedtuple(self):
        from collections import namedtuple
        levels = self.ts.levels("DAU")
        assert hasattr(levels, "metrics")
        assert hasattr(levels, "countries")
        assert hasattr(levels, "populations")
