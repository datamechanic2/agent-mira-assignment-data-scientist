# House Price Prediction

ML pipeline for predicting real estate prices with model versioning and REST API deployment.

## Project Overview

- **Dataset**: 247K real estate transactions (2020-2024)
- **Best Model**: LightGBM (R² = 0.55, RMSE = $155K, MAE = $115K)
- **Features**: Multi-core processing, model versioning, REST API

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run training
python main.py --train

# Start API server
python main.py --api

# For EDA, open the notebook
jupyter notebook notebooks/eda_and_feature_engineering.ipynb
```

## Project Structure

```
house_price_prediction/
├── .github/workflows/
│   └── deploy.yml          # CI/CD pipeline
├── notebooks/
│   └── eda_and_feature_engineering.ipynb
├── src/
│   ├── preprocessing.py    # Data preprocessing
│   ├── model_training.py   # Model training
│   ├── model_registry.py   # Version management
│   └── create_presentation.py
├── api/
│   └── app.py              # FastAPI application
├── tests/
│   └── test_api.py         # API tests
├── models/
│   └── registry/           # Versioned models
│       ├── manifest.json
│       └── house_price_predictor/
│           ├── v1.0.0/
│           └── v1.0.1/
├── Dockerfile              # Container config
├── render.yaml             # Render.com config
├── main.py
└── requirements.txt
```

## Features

### Multi-Core Processing
- Parallel model training using `n_jobs` parameter
- GridSearchCV with parallel cross-validation
- ThreadPoolExecutor for batch API predictions

### Model Versioning

The `ModelRegistry` class provides:

```python
from src.model_registry import ModelRegistry

registry = ModelRegistry("models/registry")

# Register new version
version = registry.register(
    model=trained_model,
    model_name="house_price_predictor",
    metrics={"rmse": 157307, "r2": 0.53},
    params={"learning_rate": 0.05},
    tags=["lgbm", "tuned"]
)

# Promote to production
registry.promote("house_price_predictor", "v1.0.0", stage="production")

# Load model
model, preprocessor, metadata = registry.load("house_price_predictor", stage="production")

# Compare versions
registry.compare("house_price_predictor", "v1.0.0", "v1.0.1")

# Rollback
registry.rollback("house_price_predictor", "v1.0.0")

# View all versions
registry.summary()
```

### REST API

Start server:
```bash
python main.py --api --port 8000
```

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single property prediction |
| `/predict/batch` | POST | Batch predictions |
| `/health` | GET | Health check |
| `/model/info` | GET | Model details |
| `/model/versions` | GET | List all versions |
| `/model/load/{version}` | POST | Load specific version |
| `/model/load?stage=production` | POST | Load by stage |
| `/model/compare?v1=v1.0.0&v2=v1.0.1` | GET | Compare versions |

**Example Request:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "location": "CityA",
    "size": 2500,
    "bedrooms": 3,
    "bathrooms": 2,
    "year_built": 2010,
    "condition": "Good",
    "property_type": "Single Family"
  }'
```

**Response:**
```json
{
  "predicted_price": 485000.50,
  "price_low": 436500.45,
  "price_high": 533500.55,
  "model_version": "v1.0.0",
  "details": { ... }
}
```

## Models Evaluated

| Model | Val RMSE | Val MAE | R² | Training Time |
|-------|----------|---------|-----|---------------|
| LightGBM | $155,770 | $115,354 | 0.55 | 0.8s |
| XGBoost | $156,003 | $115,507 | 0.54 | 0.4s |
| Gradient Boosting | $156,047 | $115,516 | 0.54 | 48s |
| Random Forest | $157,830 | $116,811 | 0.53 | 6.9s |
| Ridge | $161,201 | $120,078 | 0.51 | 0.01s |

## Feature Engineering

| Feature | Description |
|---------|-------------|
| `property_age` | Years since construction at sale |
| `total_rooms` | Bedrooms + Bathrooms |
| `bath_ratio` | Bathrooms / Bedrooms |
| `size_cat` | Size category (small/medium/large/xlarge) |
| `year_sold` | Year of sale |
| `month_sold` | Month of sale |
| `is_new` | Built within 5 years of sale |
| `size_per_bedroom` | Sq ft per bedroom (spaciousness) |
| `size_per_room` | Sq ft per total room |
| `bed_bath_product` | Bedroom × Bathroom interaction |
| `age_bucket` | Age category (new/mid/old) |
| `season` | Season of sale (winter/spring/summer/fall) |
| `days_since_2020` | Days since Jan 2020 (time trend) |
| `is_luxury` | Large size + New condition flag |

## Top Predictive Features

1. Days Since 2020 (845)
2. Size (581)
3. Size Per Room (443)
4. Size Per Bedroom (302)
5. Year Built (178)

## API Documentation

Interactive docs available at `http://localhost:8000/docs` when server is running.

## Deployment (Free)

### Option 1: Render.com (Recommended)

1. Push code to GitHub
2. Go to [render.com](https://render.com) and sign up (free)
3. Click "New" → "Web Service"
4. Connect your GitHub repo
5. Render auto-detects the Dockerfile
6. Click "Create Web Service"

Your API will be live at `https://your-app.onrender.com`

### Option 2: Railway.app

1. Go to [railway.app](https://railway.app) and sign up
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repo
4. Railway auto-deploys from Dockerfile

### Option 3: Docker (Local/Self-hosted)

```bash
# Build image
docker build -t house-price-api .

# Run container
docker run -p 8000:8000 house-price-api
```

### GitHub Actions CI/CD

The repo includes `.github/workflows/deploy.yml` that:
- Runs tests on every push/PR
- Builds and tests Docker image
- Auto-deploys to Render on push to `main`

To enable auto-deploy to Render:
1. Get your Render API key from Dashboard → Account Settings
2. Get your Service ID from the service URL
3. Add secrets in GitHub: `RENDER_API_KEY` and `RENDER_SERVICE_ID`

## Requirements

- Python 3.9+
- pandas, numpy
- scikit-learn, xgboost, lightgbm
- fastapi, uvicorn
- joblib

## License

MIT
