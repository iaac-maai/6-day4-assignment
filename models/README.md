---
license: mit
library_name: pytorch
pipeline_tag: tabular-regression
tags:
  - pytorch
  - mlp
  - regression
  - energy
  - built-environment
  - iaac-maai
  - ai-methods-1
metrics:
  - r2
  - mae
---

# Seattle buildings: energy use per m2 with a small neural network

This model gives a first estimate of the yearly energy use per m2 (site EUI) of a large building in Seattle, from what an architect knows in a feasibility study: the use, the size, the number of floors, the year and the location. It is a small MLP (50 → 64 → 64 → 1 with ReLU, 7,489 parameters) trained from scratch with PyTorch on 2,054 buildings of Seattle's 2016 energy report. It was built as a course assignment for AI Methods I (IAAC MaAI 2026/27), to compare a neural network with our [Day 2 random forest](https://huggingface.co/iaac-maai/day2-team6-energy-use) on the same test buildings.

- **GitHub repository (full pipeline):** https://github.com/iaac-maai/6-day4-assignment
- **Team:** Chloe Hraibeh, Jeff El Khoury, Luca Obejdeanu

## Model at a glance

![Network vs Day 2 model vs baseline](https://huggingface.co/iaac-maai/day4-team6-energy-use-mlp/resolve/main/test_comparison.png)

*On the same 643 test buildings the network ties with our Day 2 random forest and both are far better than the mean. The network is not better: on a table of 2,000 rows a neural network doesn't beat the trees.*

![Predicted vs actual](https://huggingface.co/iaac-maai/day4-team6-energy-use-mlp/resolve/main/predicted_vs_actual.png)

*Both models pull their predictions towards the middle: buildings that use very little are overestimated, and the few above 1,000 kWh/m2 are strongly underestimated by both.*

| Model | R2 on log10 (test) | MAE (kWh/m2 per year) | Typical error |
|---|---|---|---|
| Baseline: always the mean | 0.00 | 89 | 66% |
| Day 2 random forest | 0.387 | 68 | 45% |
| **This model** | **0.382** | **69** | **46%** |

## Intended use

- **Intended users and use:** an architect or engineer who wants an order of magnitude of the energy use per m2 of a new large building in Seattle during a feasibility study, before any energy simulation. A teaching model.
- **Out-of-scope uses:**
  - a number for design decisions or energy certificates: the typical error is around 45%;
  - judging or ranking a specific existing building or its owner;
  - data centers, labs or very mixed buildings;
  - other cities, small buildings (the data only has buildings over 20,000 sqft, about 1,860 m2) or other years, without retraining.

## Model details

| | |
|---|---|
| **Task** | tabular regression |
| **Architecture** | MLP: Linear 50 → 64, ReLU, Linear 64 → 64, ReLU, Linear 64 → 1 (with `Dropout(0.0)` layers, kept so the layer numbers match the weights) |
| **Parameters** | 7,489, all trainable |
| **Base model** | none, trained from scratch |
| **Input** | 11 columns, turned into 50 inputs by the preprocessing (8 scaled numbers, 42 one-hot columns) |
| **Output** | log10 of the site EUI; `10 ** output` gives kWh/m2 per year |
| **Files** | `model.pt` (weights, `state_dict`) and `scaler.joblib` (the fitted scikit-learn `ColumnTransformer`: imputers, scaler and one-hot encoder) |
| **Library versions** | Python 3.12, torch 2.14.1, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3, joblib 1.6.0 |

### Input features

The model expects a `pandas.DataFrame` with exactly these 11 columns. Missing values (NaN) are filled like in training, and unknown categories are ignored.

| Column | Type | Unit | Description | Preprocessing |
|---|---|---|---|---|
| `primary_property_type` | category | - | main use, e.g. `"Low-Rise Multifamily"`, `"Small- and Mid-Sized Office"`, `"Hotel"`, `"Warehouse"` | one-hot |
| `building_type` | category | - | city category: `"Nonresidential"`, `"Multifamily LR (1-4)"`, `"Multifamily MR (5-9)"`, `"Multifamily HR (10+)"`, `"Campus"`, `"SPS-District K-12"` | one-hot |
| `neighborhood` | category | - | Seattle neighborhood in capitals, e.g. `"DOWNTOWN"`, `"EAST"`, `"BALLARD"` | one-hot |
| `year_built` | integer | year | year of construction | median imputation, scaling |
| `number_of_floors` | integer | floors | number of floors | median imputation, scaling |
| `number_of_buildings` | integer | count | buildings on the property (usually 1) | median imputation, scaling |
| `gfa_m2` | float | m2 | gross floor area of the buildings, without parking | median imputation, scaling |
| `gfa_per_floor_m2` | float | m2 | `gfa_m2 / number_of_floors`, roughly the footprint | median imputation, scaling |
| `parking_share` | float | 0-1 | part of the total area that is parking | median imputation, scaling |
| `latitude` | float | degrees | location | median imputation, scaling |
| `longitude` | float | degrees | location | median imputation, scaling |

## Training data

| | |
|---|---|
| **Dataset** | 2016 Building Energy Benchmarking (Seattle), cleaned in our [Day 1 repository](https://github.com/iaac-maai/6-day1-assignment) |
| **Source** | [kaggle.com/datasets/city-of-seattle/sea-building-energy-benchmarking](https://www.kaggle.com/datasets/city-of-seattle/sea-building-energy-benchmarking) |
| **Licence** | Public Domain (City of Seattle) |
| **One row is** | one building (or property) in 2016 |
| **Size and splits** | 3,211 buildings: train 2,054, validation 514, test 643. The test set is the same as for our Day 2 model (`test_size=0.2`, `random_state=42`) |
| **Coverage** | Seattle, 2016, non-residential and multifamily buildings over 20,000 sqft |

## Training procedure

- **Settings:** MSE loss on log10 of the site EUI, Adam with learning rate 0.001, batches of 32, at most 300 epochs, seed 42.
- **Regularisation:** early stopping. Training stops when the validation loss hasn't improved for 30 epochs and goes back to the best weights. We also tried dropout 0.3 with weight decay 0.001 (validation R2 0.387) and a smaller network with 16 neurons per layer (0.385); early stopping alone was best (0.396).
- **Hardware and time:** a laptop CPU, a few seconds (the 300-epoch run without regularisation took 8 s).
- **Model selection:** the weights of epoch 28, where the validation loss was lowest. The test set was used once, at the end.

![Training curves](https://huggingface.co/iaac-maai/day4-team6-energy-use-mlp/resolve/main/training_curves.png)

*Training and validation improve together only for the first 28 epochs; after that the training loss keeps falling while the validation loss rises. Without early stopping the network memorises the training buildings: after 300 epochs R2 0.81 on training and 0.11 on validation.*

![The network before and after training](https://huggingface.co/iaac-maai/day4-team6-energy-use-mlp/resolve/main/network_before_after.png)

*The first 12 inputs and neurons of each layer, before and after training (here the 300-epoch run). Training only moves the numbers on the lines: a few weights grow large, most stay small. A thick line is not the same as an important input.*

## Evaluation

On the 643 test buildings:

| Metric | Mean (baseline) | Day 2 random forest | This model |
|---|---|---|---|
| R2 on log10 | 0.00 | 0.387 | 0.382 |
| MAE (kWh/m2 per year) | 89 | 68 | 69 |
| Typical error | 66% | 45% | 46% |
| Buildings underestimated | 49% | 49% | 46% |

The predictions of the network and of the random forest are correlated at 0.90.

**Where it fails** (median error per use, uses with at least 10 test buildings):

| Use | Test buildings | This model | Day 2 random forest |
|---|---|---|---|
| Other | 52 | 52% | 52% |
| Warehouse | 37 | 52% | 49% |
| Mixed Use Property | 38 | 50% | 39% |
| K-12 School | 12 | 38% | 29% |
| High-Rise Multifamily | 19 | 29% | 16% |
| Large Office | 37 | 26% | 36% |
| Hotel | 24 | 17% | 26% |
| Medical Office | 13 | 14% | 20% |

The network is worse than the random forest on high-rise apartments, mixed use buildings and schools, and better on large offices, hotels and medical offices. Warehouses and the "Other" group stay around 50% for both: the use alone doesn't say what happens inside. With 12 to 52 test buildings per use these medians are rough.

## Limitations and biases

- R2 0.38: it explains less than half of the differences. Energy depends a lot on how the building is used (hours, equipment, people), which isn't in the data.
- It pulls predictions towards the middle, so it underestimates the buildings that use the most.
- The network learned on 2,054 buildings and the random forest on 2,568 (it didn't need a validation set), so the comparison slightly favours the random forest.
- One seed and one validation split: the ranking of the experiments (0.385 to 0.396) is within noise.
- Only large buildings that report, only Seattle, only 2016, self-reported data. The split is random, so it is tested on existing buildings, not on buildings built later.

## How to use

```python
import joblib
import pandas as pd
import torch
from torch import nn
from huggingface_hub import hf_hub_download

REPO = "iaac-maai/day4-team6-energy-use-mlp"

#exactly the trained architecture, including the Dropout(0.0) layers
model = nn.Sequential(
    nn.Linear(50, 64), nn.ReLU(), nn.Dropout(0.0),
    nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.0),
    nn.Linear(64, 1),
)
model.load_state_dict(torch.load(hf_hub_download(REPO, "model.pt"), map_location="cpu"))
model.eval()
preprocessing = joblib.load(hf_hub_download(REPO, "scaler.joblib"))

#a new 6 floor apartment block in the east of the city
example = pd.DataFrame([{
    "primary_property_type": "Mid-Rise Multifamily",
    "building_type": "Multifamily MR (5-9)",
    "neighborhood": "EAST",
    "year_built": 2016,
    "number_of_floors": 6,
    "number_of_buildings": 1,
    "gfa_m2": 6000,
    "gfa_per_floor_m2": 1000,
    "parking_share": 0.1,
    "latitude": 47.615,
    "longitude": -122.315,
}])
x = torch.tensor(preprocessing.transform(example), dtype=torch.float32)
with torch.no_grad():
    log_eui = model(x).item()
print(f"{10 ** log_eui:.0f} kWh/m2 per year")   # 117 kWh/m2 per year
```

**Security note:** `.pt` and `.joblib` files are Python pickles and can run code when they are loaded, so only load models from sources you trust. Use the library versions listed above: `scaler.joblib` may not load with another scikit-learn version.

## Reproducibility

The full pipeline (data, splits, baseline, training, experiments, evaluation) is in the GitHub repository: https://github.com/iaac-maai/6-day4-assignment. Run `notebook/neural_network.ipynb` with the packages in `requirements.txt`.

## Authors and licence

Chloe Hraibeh, Jeff El Khoury, Luca Obejdeanu. Course assignment for **AI Methods I: Machine Learning for the Built Environment**, IAAC MaAI 2026/27.

Model released under the MIT License. The training data is Public Domain (City of Seattle).
