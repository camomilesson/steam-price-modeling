# Steam Price API

FastAPI service for the final Steam price XGBoost model.

## Run

From the project root:

```bash
source .venv/bin/activate
dvc pull
uvicorn app.main:app --reload
```

The API runs at:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness check and model-loaded flag |
| `GET` | `/model-info` | Final model metadata and tuning params |
| `POST` | `/predict` | Predict market-aligned price for one game |

## Input Schema

The API uses a friendly schema and derives the engineered model features internally.

Required fields:

```json
{
  "name": "Example Quest",
  "proposed_price": 19.99,
  "release_year": 2024,
  "required_age": 0,
  "dlc_count": 1,
  "achievements": 24,
  "windows": true,
  "mac": false,
  "linux": false,
  "genres": ["Indie", "Adventure"],
  "tags": ["Story Rich", "Singleplayer", "Exploration"],
  "categories": ["Single-player", "Family Sharing", "Steam Cloud"]
}
```

The most important model inputs are `tags`, `categories`, and `genres`. Numeric fields such as DLC count, achievements, age rating, release year, and platform flags are included because they are part of the trained feature matrix.

## Example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/input_example.json
```

Response:

```json
{
  "name": "Example Quest",
  "proposed_price": 19.99,
  "predicted_market_price": 8.43,
  "price_ratio": 2.371,
  "price_alignment": "above-market",
  "selected_scope": "review_count >= 10",
  "model_family": "XGBRegressor"
}
```

`price_alignment` uses the same fixed rule as the evaluation notebook:

- below-market: `proposed_price / predicted_market_price < 0.6`
- market-aligned: `0.6 <= ratio <= 1.5`
- above-market: `ratio > 1.5`
