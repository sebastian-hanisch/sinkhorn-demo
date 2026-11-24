"""Gemeinsame Fixtures: die Handbeispiele, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from snk_scenario import handbeispiel_h1, handbeispiel_h3  # noqa: E402


@pytest.fixture
def h1():
    """H1: Angebote und Nachfragen (1/2, 1/2), Kosten Zeilen (1/2, 1), (1, 1/2); OPT = 1/2, Produktmaß kostet 3/4. Mit eps = 1/(2 ln 3) ist der weiche Plan nach einem Durchlauf [[3/8, 1/8], [1/8, 3/8]]."""
    return handbeispiel_h1()


@pytest.fixture
def h3():
    """H3: Angebote (3/4, 1/4), Nachfragen (1/4, 3/4), Kosten Zeilen (0, 1), (1, 0); OPT = 1/2. Mit eps = 1/ln 3 ist der Plan nach einem Durchlauf [[9/40, 3/8], [1/40, 3/8]]."""
    return handbeispiel_h3()
