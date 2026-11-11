# New York City Central Park Weather Prediction

A course project completed for **DATA 602 at the University of Maryland, College Park (Fall 2023)**. The project uses historical NOAA GHCND observations for New York City to engineer weather features, train a neural-network regression model for average-temperature prediction, and present exploratory analysis and forecasts through an interactive Plotly Dash dashboard.

**Authors:** Jiayuan Shen, Ke Xu, Yan Liu  
**Primary language:** Python  
**Core stack:** pandas, NumPy, scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn, Plotly, Dash

---

## 1. Project overview

The project studies daily weather observations from the NOAA Global Historical Climatology Network Daily (GHCND) dataset, focusing on the Central Park station in New York City.

The original report describes a workflow that:

1. Retrieves historical daily observations from NOAA NCEI Climate Data Online (CDO) Web Services.
2. Combines historical and newly retrieved observations into a unified time-indexed dataset.
3. Handles missing observations and creates lagged weather features.
4. Trains a two-hidden-layer feed-forward neural network to predict average temperature.
5. Evaluates predictions against held-out observations.
6. Builds an interactive dashboard for exploring feature relationships, model structure, actual-versus-predicted values, and a 30-day forecast view.

This repository is an educational machine-learning and data-analysis project. It is not an operational weather-forecasting service.

## 2. Data source

- **Provider:** NOAA National Centers for Environmental Information (NCEI)
- **Dataset:** GHCND (Global Historical Climatology Network Daily)
- **API:** NOAA Climate Data Online Web Services, v2
- **Station ID used in the notebook:** `GHCND:USW00094728`
- **Location ID used in the notebook:** `CITY:US360019`
- **Units:** metric
- **Historical period described in the report:** 2020-08-15 onward; the notebook retrieves an initial period through 2023-11-01 and then requests data beginning 2023-11-02 through the execution date.

The source contains daily meteorological measurements and weather-condition indicators. Variables referenced in the project include:

| Field | Meaning |
|---|---|
| `TMAX` | Daily maximum temperature |
| `TMIN` | Daily minimum temperature |
| `TAVG` | Average temperature derived in the project as `(TMAX + TMIN) / 2` |
| `PRCP` | Precipitation |
| `SNOW` | Snowfall |
| `AWND` | Average wind speed |
| `RHAV`, `RHMN`, `RHMX` | Relative-humidity variables used by the project |
| `ADPT`, `ASLP`, `ASTP` | Additional atmospheric variables used by the project |
| `WT01`–`WT08` | Weather-condition indicators described in the report |

NOAA API availability, station coverage, and returned records may change. Re-running the notebook on a later date may therefore produce different data and metrics.

## 3. Repository structure

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── weather_prediction.ipynb
├── data/
│   └── data.csv
├── outputs/
│   └── .....
└── reports/
    └── report.pdf
```

- `notebooks/weather_prediction.ipynb`: sanitized project notebook, with API credentials removed and the test-set scaling inconsistency corrected.
- `requirements.txt`: Python dependencies.
- `.env.example`: documents the required NOAA token variable without containing a real secret.
- `data/`: location for locally downloaded or prepared data; raw data are not committed.
- `outputs/`: location for generated charts and model artifacts.
- `reports/`: location for a **credential-redacted** course report, if you choose to publish it.

## 4. Environment setup

Recommended: Python 3.9–3.11 in a virtual environment. TensorFlow compatibility depends on the operating system and Python version; consult the official TensorFlow installation guide for your environment.

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 5. NOAA API credential setup

A NOAA CDO token is required to retrieve data. The original uploaded notebook and report appendix contained a hard-coded token. That credential should be considered exposed and revoked; create a replacement through NOAA before running the data-retrieval cells.

Set the new token as an environment variable. Do not paste it into source code, notebooks, screenshots, or Git commits.

**Windows PowerShell**
```powershell
$env:NOAA_CDO_TOKEN = "YOUR_NEW_NOAA_TOKEN"
```

**macOS / Linux**
```bash
export NOAA_CDO_TOKEN="YOUR_NEW_NOAA_TOKEN"
```

The notebook reads the credential using:

```python
import os
api_key = os.getenv("NOAA_CDO_TOKEN")
if not api_key:
    raise RuntimeError("NOAA_CDO_TOKEN is not set.")
```

`.env.example` is a placeholder only. If you use a local `.env` file, keep it untracked and load it with a package such as `python-dotenv`.

## 6. Run the project

1. Clone the repository.
2. Create and activate a Python environment.
3. Install `requirements.txt`.
4. Set `NOAA_CDO_TOKEN`.
5. Open Jupyter:

```bash
jupyter notebook
```

6. Run `notebooks/weather_prediction.ipynb` from top to bottom.

The notebook retrieves data from NOAA and builds the analysis and dashboard objects. When the Dash application cell is executed, open `http://127.0.0.1:8050` in a browser.

**Important:** NOAA API requests require network access and a valid token. The notebook may need adjustment if NOAA changes API behavior or the returned data schema.

## 7. Data preparation and feature engineering

The course workflow includes:

- Retrieving observations in date chunks to work within API response limits.
- Pivoting NOAA records so dates are rows and measurement types are columns.
- Comparing historical and newly retrieved data columns before combining them.
- Filling missing daily values using historical averages for the corresponding month/day, followed by zero filling for remaining missing values in the original workflow.
- Deriving `TAVG` from daily maximum and minimum temperatures.
- Creating 1-, 2-, and 3-day lagged features for selected weather variables, including temperature, precipitation, snowfall, wind, humidity, and atmospheric measurements.
- Removing rows made incomplete by the lag operation.

The report describes an expanded feature set and a training/validation split of 80%/20%. The model-training section describes 58 input features after excluding contemporaneous `TAVG`, `TMAX`, and `TMIN` from the predictor set.

## 8. Model architecture and training

The report describes a TensorFlow/Keras feed-forward neural network with:

| Component | Configuration |
|---|---|
| Input | 58 features, as reported |
| Hidden layer 1 | 64 units, ELU activation |
| Hidden layer 2 | 64 units, ELU activation |
| Output | 1 unit for temperature regression |
| Regularization | L2 regularization |
| Optimizer | Adam |
| Loss | Mean Squared Error (MSE) |
| Early stopping | Monitor validation loss; patience = 10 |
| Batch size | 32 |
| Maximum epochs | 200 |

The notebook applies `StandardScaler` to the input features. The cleaned notebook corrects the original scaling inconsistency by reusing the scaler fitted on the training data for the test set and subsequent forecast inputs.

## 9. Evaluation and reported result

The report records a test-set **Mean Absolute Error (MAE) of approximately 0.9245** for average-temperature prediction. The report presents an actual-versus-predicted scatter plot on page 5.

This is the result reported for the original course run, not a guaranteed result from a fresh execution. Random model initialization, changes in the NOAA data, preprocessing behavior, and environment/library versions can change the metric.

The original notebook uses a randomized `train_test_split`. For a time-series forecasting task, a chronological train/validation/test split is preferable because random splitting can allow observations from later dates to appear in training while earlier dates appear in testing. Treat the reported MAE as the original course-project result, not as a leakage-proof estimate of future forecasting performance.

## 10. Interactive dashboard

The report (pages 6–8) describes a Plotly Dash dashboard with:

- A selectable time-series chart comparing average temperature with a selected weather feature.
- An interactive feature-correlation heatmap.
- A visualization of the neural-network architecture.
- An actual-versus-predicted temperature comparison.
- A forecast chart with a date range slider and a 30-day forecast view.

Run the Dash application from the notebook after the data and model objects have been created.

## 11. Limitations and future improvements

The original report identifies limitations in the availability and update frequency of some weather features. It notes that using historical averages as a compromise for unavailable future inputs can reduce accuracy during abrupt temperature changes and extreme-weather events.

Potential next steps:

1. Use chronological validation and rolling-origin backtesting.
2. Fit all preprocessing transformations only on training data.
3. Compare the neural network against persistence, seasonal-naive, linear regression, Random Forest, and gradient-boosting baselines.
4. Report MAE, RMSE, and error by season and temperature range.
5. Build an explicitly recursive or multi-horizon forecasting strategy for future lag features instead of relying only on historical month/day averages.
6. Track the data retrieval date, station metadata, feature schema, and package versions for reproducibility.
7. Add automated checks for duplicate dates, missing dates, impossible values, and API pagination.

## 12. References

- NOAA NCEI Climate Data Online: https://www.ncei.noaa.gov/cdo-web/
- NOAA CDO Web Services v2 documentation: https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation
- pandas: https://pandas.pydata.org/docs/
- scikit-learn: https://scikit-learn.org/stable/
- TensorFlow: https://www.tensorflow.org/
- Keras: https://keras.io/
- Plotly: https://plotly.com/python/
- Dash: https://dash.plotly.com/

## 13. Academic context

This project was completed as the DATA 602 final project at the University of Maryland, College Park, in Fall 2023. The original report lists Jiayuan Shen, Ke Xu, and Yan Liu as authors. Please retain the original team attribution when reusing or publishing the course deliverables.
