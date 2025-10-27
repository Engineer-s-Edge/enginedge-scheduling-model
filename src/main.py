import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# Adjust imports to be relative for package structure
from .ml.maml_scheduler import SchedulingMAML
from .nlp.deliverable_mapper import DeliverableMapper

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO)
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
    deliverable_text: str = Field(..., json_schema_extra={"example": "Review Q3 performance report"})
    context: Dict[str, Any] = Field(default_factory=dict, json_schema_extra={"example": {"priority": "high"}})

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
    deliverable: Dict[str, Any] = Field(..., json_schema_extra={"example": {"title": "Code new feature"}})
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
        logger.warning("⚠️ Service is starting in a degraded state. Models are not available.")
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
        "models_initialized": deliverable_mapper is not None and maml_scheduler is not None
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
            deliverable_text=request.deliverable_text,
            context=request.context
        )
        return result
    except Exception as e:
        logger.error(f"Error in /map-deliverable: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during NLP processing.")

@app.post("/predict-slots", response_model=PredictSlotsResponse, tags=["ML"])
async def predict_slots_endpoint(request: PredictSlotsRequest):
    """
    Predicts and recommends optimal time slots for a given deliverable.
    """
    if not maml_scheduler:
        raise HTTPException(status_code=503, detail="ML scheduling model is not available.")

    try:
        # The MAML scheduler expects embeddings; for simplicity, this endpoint could
        # call the NLP mapper internally or expect the embedding to be passed in.
        # Here, we assume the prediction features are derived without the embedding step
        # for a simplified endpoint. A more complex implementation would chain these calls.

        recommendations = await maml_scheduler.predict_optimal_slots(
            user_id=request.user_id,
            deliverable=request.deliverable,
            context=request.context
        )
        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error in /predict-slots: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during slot prediction.")

# To run this application:
# uvicorn calendar-model.src.main:app --reload --port 8000
# Note the path `calendar-model.src.main` assumes you run uvicorn from the project root.
