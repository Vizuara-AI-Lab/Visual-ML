# Visual-ML Server

Production-ready ML platform for Linear Regression and Logistic Regression.

## Features

✅ **Algorithms Implemented**

- Linear Regression (scikit-learn backend)
- Logistic Regression (binary & multi-class)

✅ **Node-Based ML Pipeline**

- Upload File Node - Dataset upload with validation
- Preprocess Node - Missing values, TF-IDF, scaling
- Split Node - Train/val/test splitting
- Train Node - Model training with hyperparameters
- Evaluate Node - Comprehensive metrics

✅ **Production-Ready Features**

- Rate limiting (100 req/min, 1000 req/hour)
- Model caching (LRU cache for 10 models)
- Request validation (Pydantic schemas)
- Structured logging (JSON format)
- Error handling with detailed messages
- Model versioning & management
- Admin-protected training endpoints
- Async/await support
- CORS middleware

✅ **API Endpoints**

- `POST /api/v1/ml/nodes/run` - Execute single node
- `POST /api/v1/ml/pipeline/run` - Execute pipeline
- `POST /api/v1/ml/train/regression` - Train regression model (admin)
- `POST /api/v1/ml/train/classification` - Train classification model (admin)
- `POST /api/v1/ml/predict/regression` - Predict (regression)
- `POST /api/v1/ml/predict/classification` - Predict (classification)
- `GET /api/v1/ml/models` - List models
- `POST /api/v1/ml/models/reload` - Hot reload model (admin)

## Installation

```bash
# Install dependencies
pip install -e .

# Or with uv
uv pip install -e .
```
## 🔑 Default Admin

- Email: `admin@visualml.com`
- Password: `Admin123`

## Configuration

Create a `.env` file:

```env
# Application
ENVIRONMENT=development
DEBUG=True
SECRET_KEY=your-secret-key-min-32-chars

# Database
DATABASE_URL=sqlite:///./visual_ml.db

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_PER_HOUR=1000

# Storage
MAX_UPLOAD_SIZE_MB=100
UPLOAD_DIR=./uploads
MODEL_ARTIFACTS_DIR=./models

# Caching (optional - production)
ENABLE_CACHE=False
REDIS_HOST=localhost
REDIS_PORT=6379

# ML Settings
MODEL_CACHE_SIZE=10
DEFAULT_RANDOM_SEED=42
MAX_MODEL_VERSIONS=10
```

## Running the Server

```bash
# Development
python main.py

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```
