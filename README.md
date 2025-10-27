# EnginEdge - Calendar ML Model Service

This service provides advanced, real-time personalized calendar scheduling recommendations. It is built as a Python microservice using PyTorch, Transformers, and FastAPI, and is designed to replace the existing in-process TensorFlow.js model in the `main-node` backend.

The architecture is based on Model-Agnostic Meta-Learning (MAML) to handle cold-start problems and provide rapid personalization for each user.

## Core Components

- **FastAPI Application (`src/main.py`):** Exposes the ML/NLP functionality via a REST API.
- **MAML Scheduler (`src/ml/maml_scheduler.py`):** The core PyTorch model for learning user scheduling preferences and predicting optimal time slots.
- **Deliverable Mapper (`src/nlp/deliverable_mapper.py`):** A Transformer-based NLP component to map natural language tasks into semantic vector embeddings.
- **Training Pipeline (`src/ml/train_pipeline.py`):** A script to perform meta-training on the MAML model.

---

## 1. Getting Started

This service is fully containerized with Docker and Docker Compose for easy setup and deployment.

### Prerequisites

- Docker
- Docker Compose

### Running the Service

1.  **Navigate to the directory:**
    ```bash
    cd calendar-model
    ```

2.  **Build and run the services:**
    This command will build the Docker image for the API, pull the Redis image, and start both containers in the background.
    ```bash
    docker-compose up --build -d
    ```

3.  **Verify the services are running:**
    You can check the status of the containers:
    ```bash
    docker-compose ps
    ```
    You should see `calendar-model-api` and `calendar-model-redis` with a status of `Up`.

4.  **Check the logs (optional):**
    To see the logs from the API service:
    ```bash
    docker-compose logs -f model-api
    ```
    You should see a message indicating that the Uvicorn server has started and the models have been initialized.

---

## 2. API Endpoints

The service exposes the following endpoints.

### Health Check

- **Endpoint:** `GET /health`
- **Description:** Checks if the service is running and if the models are initialized.
- **Example:**
  ```bash
  curl http://localhost:8000/health
  ```
- **Expected Response:**
  ```json
  {
    "status": "ok",
    "models_initialized": true
  }
  ```

### Map Deliverable

- **Endpoint:** `POST /map-deliverable`
- **Description:** Maps a natural language deliverable to a semantic embedding and other structured data.
- **Example:**
  ```bash
  curl -X POST http://localhost:8000/map-deliverable \
  -H "Content-Type: application/json" \
  -d '{
    "deliverable_text": "Prepare the Q4 marketing presentation",
    "context": {"priority": "high"}
  }'
  ```

### Predict Optimal Slots

- **Endpoint:** `POST /predict-slots`
- **Description:** Predicts optimal time slots for a given user and deliverable.
- **Example:**
  ```bash
  curl -X POST http://localhost:8000/predict-slots \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-123",
    "deliverable": {"title": "Code review for new feature"},
    "context": {}
  }'
  ```

---

## 3. Integration with `main-node` Backend

To integrate this new ML service, you need to modify the existing `CalendarActivityModelService` in the `main-node` project. The goal is to replace the local TensorFlow.js model calls with HTTP requests to this new service running at `http://localhost:8000`.

**File to Modify:** `main-node/src/core/infrastructure/calendar/ml/calendar-activity-model.service.ts`

### Step-by-Step Integration Guide

1.  **Inject HttpService:**
    First, you'll need NestJS's `HttpModule` to make HTTP requests. Ensure `HttpModule` is registered in the corresponding module file (`calendar-ml.module.ts` or a global module). Then, inject `HttpService` into the `CalendarActivityModelService`.

    ```typescript
    // In calendar-activity-model.service.ts
    import { HttpService } from '@nestjs/axios';
    import { firstValueFrom } from 'rxjs';

    @Injectable()
    export class CalendarActivityModelService {
      private readonly logger = new Logger(CalendarActivityModelService.name);
      private readonly mlServiceBaseUrl = 'http://localhost:8000'; // Or from config

      constructor(
        // ... other injections
        private readonly httpService: HttpService,
      ) {
        // Remove the call to this.initializeModel() that loads TensorFlow.js
      }

      // ...
    }
    ```

2.  **Refactor the `predict` Method:**
    Modify the `predict` method to call the `/predict-slots` endpoint of this new service instead of using the local TensorFlow model.

    ```typescript
    // In CalendarActivityModelService
    async predict(userId: string, proposedEvent: Partial<CalendarActivityInput>): Promise<any> {
      this.logger.log(`Calling external ML service for prediction for user ${userId}`);
      try {
        const requestBody = {
          user_id: userId,
          deliverable: {
            title: proposedEvent.eventData.title,
            // Add other relevant deliverable fields
          },
          context: proposedEvent.userContext,
        };

        const response = await firstValueFrom(
          this.httpService.post(`${this.mlServiceBaseUrl}/predict-slots`, requestBody)
        );

        const recommendations = response.data.recommendations;

        // You will need to adapt the response from the Python service
        // to the format expected by the rest of the NestJS application.
        // This is a placeholder for that transformation logic.
        const topRecommendation = recommendations[0] || { probability: 0.5, confidence: 0.5 };

        return {
          eventSuccess: topRecommendation.probability,
          userSatisfaction: topRecommendation.confidence,
          scheduleEfficiency: (topRecommendation.probability + topRecommendation.confidence) / 2,
          recommendation: topRecommendation.recommended ? 'approve' : 'modify',
          suggestions: ['Suggestion from new ML model.'],
        };

      } catch (error) {
        this.logger.error(`Failed to get prediction from ML service: ${error.message}`, error.stack);
        // Fallback to a default response
        return {
          eventSuccess: 0.5,
          userSatisfaction: 0.5,
          scheduleEfficiency: 0.5,
          recommendation: 'modify',
        };
      }
    }
    ```

3.  **Refactor `getSchedulingRecommendations`:**
    Similarly, update this method to call the new service. You can use the same `/predict-slots` endpoint or create a new, more specific one in the Python service if needed.

4.  **Remove Old Code:**
    Once the integration is complete and verified, you can safely remove the TensorFlow.js model logic:
    - The `initializeModel`, `createModel`, `extractFeatures`, `trainModel`, and `saveModel` methods.
    - The `@tensorflow/tfjs` import.
    - Any related private properties like `this.model`, `this.isTraining`, etc.

By following these steps, you will have successfully migrated from the embedded JS-based model to the more powerful, external Python-based ML microservice.
