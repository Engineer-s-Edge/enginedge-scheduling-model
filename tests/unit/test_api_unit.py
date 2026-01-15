import sys
from unittest.mock import MagicMock

import pytest

# 1. Setup Mocks for heavy ML dependencies
# We must do this BEFORE importing src.main
mock_maml_module = MagicMock()
mock_nlp_module = MagicMock()

# Define the classes within the mocked modules
mock_maml_class = MagicMock()
mock_nlp_class = MagicMock()

mock_maml_module.SchedulingMAML = mock_maml_class
mock_nlp_module.DeliverableMapper = mock_nlp_class

# Inject into sys.modules
# Note: The keys depend on how the modules are resolved.
# Since we will likely import src.main, the relative import resolves to src.ml.maml_scheduler
sys.modules["src.ml.maml_scheduler"] = mock_maml_module
sys.modules["src.nlp.deliverable_mapper"] = mock_nlp_module

from fastapi.testclient import TestClient  # noqa: E402

# 2. Import the app
# This import will trigger the top-level code in main.py, which uses the mocks
from src.main import app  # noqa: E402

client = TestClient(app)


@pytest.mark.unit
def test_health_check_ok():
    """
    Test the health check endpoint.
    Since we mocked the models to succeed initialization, it should return
    models_initialized=True (or similar).
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    # In src.main: deliverable_mapper is instantiated from the class we mocked.
    # It should be truthy.
    assert data["models_initialized"] is True


@pytest.mark.unit
def test_map_deliverable_success():
    """
    Test /map-deliverable endpoint mocking the NLP mapper response.
    """
    # Configure the mock instance
    # deliverable_mapper = DeliverableMapper() <- this happened at import time
    # We need to access that instance.

    # In src.main, 'deliverable_mapper' is a global variable holding the instance.
    # But checking main.py, it's not easily accessible unless we import it.
    from src.main import deliverable_mapper

    # Async mock for map_deliverable
    async def mock_map(*args, **kwargs):
        return {
            "deliverable": "mocked",
            "embedding": [0.1, 0.2],
            "category": "coding",
            "category_confidence": 0.95,
            "urgency": "high",
            "priority": "P1",
            "estimated_duration_hours": 2.5,
            "semantic_features": {"keywords": ["report"]},
        }

    deliverable_mapper.map_deliverable.side_effect = mock_map

    payload = {"deliverable_text": "Finish the report"}
    response = client.post("/map-deliverable", json=payload)
    assert response.status_code == 200
    assert response.json()["category"] == "coding"
