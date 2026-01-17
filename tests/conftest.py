import os
import sys
from unittest.mock import MagicMock

import pytest

# Add src to sys.path to allow importing main
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(autouse=True)
def mock_ml_components(monkeypatch):
    """
    Mock the heavy ML components to prevent them from loading during tests.
    This fixture runs automatically before every test.
    """
    # Mock the modules before they are imported by main
    sys.modules["src.ml.maml_scheduler"] = MagicMock()
    sys.modules["src.nlp.deliverable_mapper"] = MagicMock()

    # We also need to mock the classes that main.py tries to instantiate from those modules
    # But since main.py does "from .ml.maml_scheduler import SchedulingMAML", we need to ensure
    # that the mocked module returns a mocked class?

    # Actually, if we mock the imports in main.py it's safer.
    # But `src` isn't a package yet in the standard sense.
    pass
