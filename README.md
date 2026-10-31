# Restaurant rating model

A scikit-learn model that predicts the `Rating text` category for a restaurant from its location, price, cuisine, and service attributes. This repository contains the trained pipeline and evaluation summary; it does not contain the source dataset.

## What it predicts

The model was trained on the Zomato market analysis dataset snapshot. Its target is the source dataset's `Rating text` label (for example, Excellent, Very Good, Good, Average, Poor, or Not rated). Output labels follow the dataset and are not a guarantee of a restaurant's current quality.

Input columns expected by the pipeline:

- `Average Cost for two`
- `Price range`
- `Longitude`
- `Latitude`
- `Country Code`
- `City`
- `Cuisines`
- `Currency`
- `Has Table booking`
- `Has Online delivery`
- `Is delivering now`
- `Locality`

## Use

Use Python with the dependencies below. Supply one row per restaurant and include all twelve columns using the original dataset's value conventions.

```python
import joblib
import pandas as pd

model = joblib.load("restaurant_rating_model.joblib")
restaurant = pd.DataFrame([{
    "Average Cost for two": 50,
    "Price range": 2,
    "Longitude": 69.2401,
    "Latitude": 41.2995,
    "Country Code": 189,
    "City": "Tashkent",
    "Cuisines": "Uzbek",
    "Currency": "UZD",
    "Has Table booking": "No",
    "Has Online delivery": "No",
    "Is delivering now": "No",
    "Locality": "Example locality",
}])

print(model.predict(restaurant)[0])
```

The sample values only illustrate the input shape. Use values and category spellings that match the training dataset. Predictions are not calibrated probabilities or advice for choosing a restaurant.

## Evaluation

The saved pipeline was evaluated with a random train/test split of 9,542 rows (7,633 train, 1,909 test). The recorded scikit-learn version was 1.8.0.

| Metric | Result |
|---|---:|
| 5-fold CV accuracy, mean ± std | 0.5971 ± 0.0140 |
| 5-fold CV macro F1 | 0.4231 |
| Test accuracy | 0.5762 (1,100 / 1,909) |
| Test macro F1 | 0.4214 |
| Test balanced accuracy | 0.4121 |
| Test majority-class baseline accuracy | 0.3913 |

Full recorded metrics and the model SHA-256 are in [`restaurant_rating_results.json`](restaurant_rating_results.json). The split is random and has no time-based or geographic holdout, so these scores do not establish performance on future data, unseen cities, or current restaurants. Treat this as an experimental baseline.

## Data and licensing

The dataset source is [Zomato Market Analysis on Kaggle](https://www.kaggle.com/datasets/srisyra02/zomato-market-analysis). The source data is not redistributed here. The dataset's reuse terms were not verified when preparing this repository, and this repository does not specify a license for the model artifact. Check the source terms and obtain any permissions needed before reuse or redistribution.

## Security

The model is serialized with joblib, which uses Python pickle mechanisms. Loading a model file can execute code. Only load this artifact if you trust its source.

## Dependencies

Install compatible versions with:

```bash
python -m pip install -r requirements.txt
```
