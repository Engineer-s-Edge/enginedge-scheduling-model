try:
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None  # type: ignore
    _HAS_TORCH = False
import asyncio
import hashlib
import logging
import re
from typing import Dict, List, Optional

import numpy as np

try:
    # Import lazily to allow fallback when unavailable/offline
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    SentenceTransformer = None  # type: ignore


class DeliverableMapper:
    """
    NLP component for mapping deliverables to semantic embeddings
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if _HAS_TORCH:
            # type: ignore[attr-defined]
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        # Defer heavy model load; allow offline fallback
        self.model_name = model_name
        self.model = None
        self.use_fallback = False

        # Cache for embeddings
        self.embedding_cache: Dict[str, Dict] = {}

        # Predefined deliverable categories (lazy)
        self.category_embeddings: Dict[str, np.ndarray] = {}
        # Initialize categories on first use to avoid heavy init on import

    def _initialize_categories(self) -> Dict[str, np.ndarray]:
        """Initialize predefined deliverable categories with embeddings"""
        categories = {
            "meeting": "business meeting discussion collaboration",
            "coding": "programming development coding implementation",
            "planning": "project planning strategy roadmap",
            "research": "research analysis investigation study",
            "documentation": "documentation writing report notes",
            "review": "code review evaluation assessment",
            "learning": "learning training education skill development",
            "creative": "creative design brainstorming ideation",
            "administrative": "administrative tasks paperwork processing",
            "client_work": "client work customer service support",
        }

        embeddings: Dict[str, np.ndarray] = {}
        for category, description in categories.items():
            embeddings[category] = self._encode_text(description)

        return embeddings

    def _ensure_model(self):
        """Attempt to load SentenceTransformer; if unavailable, use hashing fallback."""
        if self.model is not None or self.use_fallback:
            return
        try:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers not available")
            self.model = SentenceTransformer(self.model_name)
            self.model.to(self.device)
            logging.info(f"Loaded SentenceTransformer model: {self.model_name}")
        except Exception as e:
            logging.warning(
                f"Falling back to hashing embeddings for NLP due to initialization failure: {e}"
            )
            self.model = None
            self.use_fallback = True

    def _hashing_embed(self, text: str, dim: int = 384) -> np.ndarray:
        """Deterministic bag-of-words style hashing embedding with fixed dimension."""
        vec = np.zeros(dim, dtype=np.float32)
        tokens = re.findall(r"\w+", text.lower())
        for tok in tokens:
            # Use stable hash via md5 to avoid Python's randomized hash seed
            h = hashlib.md5(tok.encode("utf-8")).hexdigest()
            # Take first 8 hex digits -> 32-bit int
            idx = int(h[:8], 16) % dim
            vec[idx] += 1.0
        # Normalize to unit length if non-zero
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to a 384-dim embedding, using model if available, else fallback."""
        # Ensure model or fallback is ready
        self._ensure_model()
        if self.model is not None:
            emb = self.model.encode(text, convert_to_tensor=False)
            # Ensure numpy array
            if isinstance(emb, list):
                emb = np.array(emb, dtype=np.float32)
            return emb.astype(np.float32)
        # Fallback
        return self._hashing_embed(text, dim=384)

    async def map_deliverable(
        self, deliverable_text: str, context: Optional[Dict] = None
    ) -> Dict:
        """
        Map deliverable text to semantic embedding and category
        """
        try:
            if deliverable_text is None:
                deliverable_text = ""
            # Handle empty or whitespace-only strings explicitly per tests
            if deliverable_text.strip() == "":
                zero_embedding = np.zeros(384, dtype=np.float32).tolist()
                return {
                    "embedding": zero_embedding,
                    "category": "unknown",
                    "category_confidence": 0.0,
                    "urgency": "medium",
                    "priority": "medium",
                    "estimated_duration_hours": 1.0,
                    "semantic_features": {
                        "word_count": 0,
                        "is_question": False,
                        "action_verb_present": False,
                    },
                }
            # Check cache first
            cache_key = f"{deliverable_text}_{hash(str(context))}"
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]

            # Generate embedding (model or fallback)
            embedding = self._encode_text(deliverable_text)

            # Ensure category embeddings initialized lazily
            if not self.category_embeddings:
                self.category_embeddings = self._initialize_categories()

            # Find most similar category
            best_category = self._find_best_category(embedding)

            # Extract urgency and priority
            urgency = self._extract_urgency(deliverable_text, context)
            priority = self._extract_priority(deliverable_text, context)

            # Estimate duration
            estimated_duration = self._estimate_duration(
                deliverable_text, best_category
            )

            result = {
                "embedding": embedding.tolist(),
                "category": best_category["category"],
                "category_confidence": best_category["confidence"],
                "urgency": urgency,
                "priority": priority,
                "estimated_duration_hours": estimated_duration,
                "semantic_features": self._extract_semantic_features(deliverable_text),
            }

            # Cache result
            self.embedding_cache[cache_key] = result

            return result

        except Exception as e:
            logging.error(f"Deliverable mapping failed: {str(e)}")
            return {
                "embedding": np.zeros(384).tolist(),
                "category": "unknown",
                "category_confidence": 0.0,
                "urgency": "medium",
                "priority": "medium",
                "estimated_duration_hours": 1.0,
                "semantic_features": {},
            }

    def _find_best_category(self, embedding: np.ndarray) -> Dict:
        """Find the most similar predefined category"""
        best_similarity = -1
        best_category = "unknown"

        for category, cat_embedding in self.category_embeddings.items():
            denom = np.linalg.norm(embedding) * np.linalg.norm(cat_embedding)
            if denom == 0:
                similarity = 0.0
            else:
                similarity = float(np.dot(embedding, cat_embedding) / denom)

            if similarity > best_similarity:
                best_similarity = similarity
                best_category = category

        return {"category": best_category, "confidence": float(best_similarity)}

    def _extract_urgency(self, text: str, context: Optional[Dict] = None) -> str:
        """Extract urgency level from text and context"""
        urgent_keywords = ["urgent", "asap", "immediately", "deadline", "critical"]
        low_urgency_keywords = ["eventually", "when possible", "low priority"]

        text_lower = text.lower()
        # Text keyword precedence
        if any(keyword in text_lower for keyword in urgent_keywords):
            return "high"
        if any(keyword in text_lower for keyword in low_urgency_keywords):
            return "low"

        # Context explicit override 'urgency'
        if (
            context
            and "urgency" in context
            and context["urgency"] in ["low", "medium", "high"]
        ):
            return context["urgency"]

        # Deadline-based heuristic only if no keywords and no explicit urgency
        if context:
            days_until_deadline = context.get("days_until_deadline")
            if days_until_deadline is not None:
                if days_until_deadline <= 1:
                    return "high"
                elif days_until_deadline <= 3:
                    return "medium"
                else:
                    return "low"

        return "medium"

    def _extract_priority(self, text: str, context: Optional[Dict] = None) -> str:
        """Extract priority level from text and context."""
        text_lower = text.lower()
        if context and "priority" in context:
            return context["priority"]

        if "high priority" in text_lower or "critical" in text_lower:
            return "high"
        if "low priority" in text_lower:
            return "low"

        return "medium"

    def _estimate_duration(self, text: str, category: dict) -> float:
        """Estimate task duration from text and category."""
        text_lower = text.lower()

        # Check for explicit duration mentions, e.g., "2-hour", "30 mins"
        duration_match = re.search(
            r"(\d+(\.\d+)?)\s*[-]?\s*(hour|hr|h|minute|min|m)", text_lower
        )
        if duration_match:
            value = float(duration_match.group(1))
            unit = duration_match.group(3)
            if unit.startswith("h"):
                return value
            if unit.startswith("m"):
                return value / 60.0

        # Default durations based on category
        category_durations = {
            "meeting": 1.0,
            "coding": 2.5,
            "planning": 1.5,
            "research": 2.0,
            "documentation": 1.0,
            "review": 0.75,
            "learning": 1.0,
            "creative": 2.0,
            "administrative": 0.5,
            "client_work": 1.5,
            "unknown": 1.0,
        }
        return category_durations.get(category.get("category", "unknown"), 1.0)

    def _extract_semantic_features(self, text: str) -> dict:
        """Extract simple semantic features from the text."""
        words = text.split()
        return {
            "word_count": len(words),
            "is_question": text.strip().endswith("?"),
            "action_verb_present": any(
                verb in text.lower()
                for verb in ["create", "review", "plan", "implement", "discuss"]
            ),
        }

    async def batch_map_deliverables(
        self, deliverables: List[str], contexts: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Process multiple deliverables in batch for efficiency"""
        if contexts is None:
            contexts = [{}] * len(deliverables)

        tasks = [
            self.map_deliverable(deliverable, context)
            for deliverable, context in zip(deliverables, contexts)
        ]

        return await asyncio.gather(*tasks)
