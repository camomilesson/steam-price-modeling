# Steam Price Modeling

Market-aligned Steam game price modeling using metadata from the Kaggle Steam Games Dataset.

## Goal

The project estimates a game's market-aligned list price from Steam metadata, then compares actual/current pricing against that expected market price. The intended use cases are:

- helping players judge whether a game is priced in line with similar alternatives;
- helping developers benchmark pricing for new or existing games.

## Data

The raw dataset is not committed to this repository because it is large. Download it from Kaggle:

https://www.kaggle.com/datasets/fronkongames/steam-games-dataset/data

Place the downloaded `games.csv` file here:

```text
data/games.csv
```

The Kaggle page lists the dataset license as MIT. The dataset is maintained by Martin Bustos / FronkonGames.

## Notebook Order

Run notebooks in this order:

```text
notebooks/01_eda.ipynb
notebooks/02_price_eda.ipynb
notebooks/03_feature_engineering.ipynb
```

`03_feature_engineering.ipynb` generates:

```text
data/games_price_model.csv
```

This generated CSV is also not committed.

## Current Modeling Decisions

- Use estimated list price as the main target.
- Reconstruct list price from discounted rows when `0 < Discount < 100`.
- Exclude free games, `Discount == 100`, and estimated list prices above `$100`.
- Use `log_list_price` as the regression target.
- Keep current price separately for deal/current-price analysis.
- Separate pre-release features from post-release outcome features.

## Environment

Python `3.11` was used for the notebooks.

Install dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Then select the project virtual environment as the Jupyter kernel.

