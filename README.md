# Steam Price Modeling

Market-aligned Steam game price modeling using metadata from the Kaggle Steam Games Dataset.

The project estimates a paid game's expected market list price from Steam metadata, then compares the current price against that expected price. It supports two product views:

- developer-facing launch or benchmark pricing from pre-release metadata;
- player-facing value assessment from the current price and comparable metadata.

## Project Structure

```text
notebooks/                 exploration, feature engineering, modeling, evaluation
src/steam_price/           modular training and prediction code
configs/train.toml         training, tuning, MLflow, and artifact config
models/*.dvc               DVC pointers for model artifacts
app/                       FastAPI serving app
examples/input_example.json
```

## Data

The raw dataset is not committed because it is large. Download it from Kaggle:

https://www.kaggle.com/datasets/fronkongames/steam-games-dataset/data

Place the downloaded file here:

```text
data/games.csv
```

The Kaggle page lists the dataset license as MIT. The dataset is maintained by Martin Bustos / FronkonGames.

## Setup

Python `3.11` was used.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
```

## Notebook Workflow

Run notebooks in this order:

```text
notebooks/01_eda.ipynb
notebooks/02_feature_eng.ipynb
notebooks/03_modeling.ipynb
notebooks/04_evaluation.ipynb
```

`02_feature_eng.ipynb` generates:

```text
data/games_price_model.csv
```

This generated CSV is not committed.

The notebooks contain the experiment story: EDA decisions, baseline models, XGBoost improvement, feature importance, and final interpretation.

## Modeling Decisions

- Target: `log_list_price`, derived from reconstructed `estimated_list_price`.
- Discount handling: reconstruct list price when `0 < Discount < 100`.
- Exclude free games, `Discount == 100`, and estimated list prices above `$100`.
- Main practical market scope: `review_count >= 10`.
- Main metric: dollar MAE, with median AE, RMSE, and within-25% / within-50% rates as diagnostics.
- Final modular model: XGBoost regressor.
- Notebook baselines: median price, Ridge, and KNN comparable-game pricing.

## Train The Final Model

The modular training pipeline uses XGBoost only. Hyperparameter search is done with Optuna on a validation split, and runs are logged to local MLflow.

```bash
python -m steam_price.train --config configs/train.toml
```

Current tuned held-out test metrics:

```text
MAE:        4.554
Median AE: 2.813
RMSE:       7.333
Within 25%: 24.054%
Within 50%: 42.175%
```

Generated artifacts:

```text
models/final_model.joblib
models/metrics.json
models/metadata.json
```

These files are ignored by git and versioned with DVC.

## MLflow

Training logs Optuna trials and the final model run locally.

```bash
mlflow ui --backend-store-uri mlruns
```

Open:

```text
http://127.0.0.1:5000
```

Do not commit `mlruns/`.

## DVC Artifacts

Model artifacts are stored in the configured Google Drive DVC remote. The repo commits only DVC pointer files:

```text
models/final_model.joblib.dvc
models/metrics.json.dvc
models/metadata.json.dvc
```

To fetch artifacts after cloning:

```bash
dvc pull
```

The Google Drive folder has been shared with the instructor.

## FastAPI Serving

The API loads `models/final_model.joblib` and serves one-game predictions.

Run locally:

```bash
uvicorn app.main:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/input_example.json
```

Example response:

```json
{
  "name": "Example Quest",
  "current_price": 19.99,
  "predicted_market_price": 8.43,
  "price_ratio": 2.371,
  "price_alignment": "above-market",
  "selected_scope": "review_count >= 10",
  "model_family": "XGBRegressor"
}
```

See [app/README.md](app/README.md) for API details.

## Production Notes

This model is best suited for batch scoring or on-demand scoring of candidate game metadata. In production, monitor:

- input drift in genres, tags, categories, platforms, release years, and DLC counts;
- prediction distribution drift by price band;
- share of below-market / market-aligned / above-market labels;
- error by price band when later observed prices or human review labels are available;
- missing or unusual metadata rates.
