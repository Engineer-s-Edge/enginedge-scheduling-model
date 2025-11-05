import json
import logging
import os
import uuid
from datetime import datetime

try:
    from kafka import KafkaProducer
except Exception:
    KafkaProducer = None
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Adjust imports to be relative for package structure
from .ml.maml_scheduler import SchedulingMAML
from .nlp.deliverable_mapper import DeliverableMapper

# --- Basic Setup ---
SERVICE_NAME = os.environ.get("SERVICE_NAME", "enginedge-scheduling-model")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
KAFKA_BROKERS = os.environ.get("KAFKA_BROKERS", "localhost:9092").split(",")


class KafkaLogHandler(logging.Handler):
    def __init__(self, service_name: str, level=logging.INFO):
        super().__init__(level)
        self.service_name = service_name
        self.buffer_path = os.path.join(
            os.getcwd(), os.environ.get("LOG_BUFFER_DIR", "logs")
        )
        os.makedirs(self.buffer_path, exist_ok=True)
        self.buffer_file = os.path.join(
            self.buffer_path, f"{self.service_name}-buffer.log"
        )
        self._producer = None
        if KafkaProducer:
            try:
                self._producer = KafkaProducer(
                    bootstrap_servers=KAFKA_BROKERS,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    key_serializer=lambda k: k.encode("utf-8") if k else None,
                    acks="all",
                    retries=3,
                    max_in_flight_requests_per_connection=1,
                )
            except Exception:
                self._producer = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname.lower(),
                "message": msg,
                "service": self.service_name,
            }
            topic = f"enginedge.logs.worker.{self.service_name}"
            if self._producer:
                try:
                    self._producer.send(topic, value=entry, key=str(uuid.uuid4()))
                except Exception:
                    self._buffer(entry)
            else:
                self._buffer(entry)
        except Exception:
            pass

    def _buffer(self, entry: dict):
        try:
            with open(self.buffer_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass


root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
root_console = logging.StreamHandler()
root_console.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
root_logger.addHandler(root_console)
root_logger.addHandler(
    KafkaLogHandler(SERVICE_NAME, level=getattr(logging, LOG_LEVEL, logging.INFO))
)
logger = logging.getLogger(__name__)

# --- Global Service Instances ---
# In a production environment, you might manage these models more carefully
# (e.g., lazy loading, singleton pattern).
try:
    deliverable_mapper = DeliverableMapper()
    maml_scheduler = SchedulingMAML()
    logger.info("✅ ML and NLP models initialized successfully.")
except Exception as e:
    logger.error(f"🔥 Failed to initialize models: {e}", exc_info=True)
    # Depending on the desired behavior, you might want the app to fail startup
    # or run in a degraded state. For now, we'll let it run but log the error.
    deliverable_mapper = None
    maml_scheduler = None

# --- Pydantic Models for API Data Validation ---


class DeliverableMapRequest(BaseModel):
    deliverable_text: str = Field(
        ..., json_schema_extra={"example": "Review Q3 performance report"}
    )
    context: Dict[str, Any] = Field(
        default_factory=dict, json_schema_extra={"example": {"priority": "high"}}
    )


class DeliverableMapResponse(BaseModel):
    embedding: List[float]
    category: str
    category_confidence: float
    urgency: str
    priority: str
    estimated_duration_hours: float
    semantic_features: Dict[str, Any]


class PredictSlotsRequest(BaseModel):
    user_id: str = Field(..., json_schema_extra={"example": "user-42"})
    deliverable: Dict[str, Any] = Field(
        ..., json_schema_extra={"example": {"title": "Code new feature"}}
    )
    context: Dict[str, Any] = Field(default_factory=dict)


class SlotRecommendation(BaseModel):
    time_slot: int
    hour: int
    probability: float
    confidence: float
    recommended: bool


class PredictSlotsResponse(BaseModel):
    recommendations: List[SlotRecommendation]


# --- Lifespan handler (startup/shutdown) ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 API service is starting up.")
    if deliverable_mapper is None or maml_scheduler is None:
        logger.warning(
            "⚠️ Service is starting in a degraded state. Models are not available."
        )
    try:
        yield
    finally:
        logger.info("🛑 API service is shutting down.")


# Create the FastAPI app with lifespan
app = FastAPI(
    title="Calendar Scheduling ML Service",
    description="A service for personalized calendar scheduling using MAML and NLP.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Simple health check endpoint to confirm the service is running.
    """
    return {
        "status": "ok",
        "models_initialized": deliverable_mapper is not None
        and maml_scheduler is not None,
    }


@app.post("/map-deliverable", response_model=DeliverableMapResponse, tags=["NLP"])
async def map_deliverable_endpoint(request: DeliverableMapRequest):
    """
    Maps a natural language deliverable to a semantic embedding and structured data.
    """
    if not deliverable_mapper:
        raise HTTPException(status_code=503, detail="NLP model is not available.")

    try:
        result = await deliverable_mapper.map_deliverable(
            deliverable_text=request.deliverable_text, context=request.context
        )
        return result
    except Exception as e:
        logger.error(f"Error in /map-deliverable: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error during NLP processing."
        )


@app.post("/predict-slots", response_model=PredictSlotsResponse, tags=["ML"])
async def predict_slots_endpoint(request: PredictSlotsRequest):
    """
    Predicts and recommends optimal time slots for a given deliverable.
    """
    if not maml_scheduler:
        raise HTTPException(
            status_code=503, detail="ML scheduling model is not available."
        )

    try:
        # The MAML scheduler expects embeddings; for simplicity, this endpoint could
        # call the NLP mapper internally or expect the embedding to be passed in.
        # Here, we assume the prediction features are derived without the embedding step
        # for a simplified endpoint. A more complex implementation would chain these calls.

        recommendations = await maml_scheduler.predict_optimal_slots(
            user_id=request.user_id,
            deliverable=request.deliverable,
            context=request.context,
        )
        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error in /predict-slots: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error during slot prediction."
        )


# To run this application:
# uvicorn calendar-model.src.main:app --reload --port 8000
# Note the path `calendar-model.src.main` assumes you run uvicorn from the project root.
