

# 🧠 Introverts vs Extroverts Classifier (Kaggle Playground S5E7)

A modular and production-ready machine learning pipeline for classifying individuals as **introverts or extroverts**, built for the [Kaggle Playground Series - Season 5, Episode 7](https://www.kaggle.com/competitions/playground-series-s5e7).

This project integrates **DVC** for data and experiment versioning, **dvclive** for live metric tracking, and **XGBoost** for high-performance classification, with clean modular Python code and logging.

---

## 📁 Project Structure

```


├── src/                        # Modular ML pipeline components
│   ├── utils/                  # Utility functions (e.g., param loader)
│   ├── data\_ingestion.py
│   ├── data\_preprocessing.py
│   ├── feature\_engineering.py
│   ├── model\_building.py
│   └── model\_evaluation.py
├── data/                       # Contains raw, interim, processed data
├── models/                     # Trained models and encoders
├── logs/                       # Log files for each stage
├── dvclive/                    # dvclive experiment tracking data
├── params.yaml                 # Configurable parameters
├── dvc.yaml                    # DVC pipeline stages
├── dvc.lock                    # DVC lock file
├── metrics.json                # Final evaluation metrics
├── README.md                   # ReadMe File

````

---

## Key Features

- 🔄 **Reproducible ML pipeline** using DVC
- 📦 **Modular code** for each stage (ingestion, preprocessing, etc.)
- 🧪 **Experiment tracking** via `dvclive` and `dvc exp`
- 🔍 **XGBoost + Stratified K-Fold CV** for robust performance
- 📈 **Live metrics logging** (accuracy, F1, precision, recall)
- 🧹 **Custom feature engineering and preprocessing**

---

##  Final Results

| Metric     | Score     |
|------------|-----------|
| Accuracy   | **0.9679** |
| Precision  | 0.9678    |
| Recall     | 0.9679    |
| F1-Score   | 0.9678    |

These results were computed using `sklearn.metrics` on the training set (used for cross-validation) and logged using `dvclive`.

---

##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/knight22-21/Mlops-project-dvc.git
````

### 2. Install dependencies

You’ll need:

* Python 3.8+
* `dvc[dvclive]`
* `scikit-learn`, `xgboost`, `pandas`, etc.

### 3. Run the full pipeline

```bash
dvc repro
```

### 4. Run individual scripts (To verify each step is working fine)

```bash
python src/data_ingestion.py
python src/data_preprocessing.py
python src/feature_engineering.py
python src/model_building.py
python src/model_evaluation.py
```

### 5. Run and track experiments

```bash
dvc exp run
dvc exp show
```


##  DVC Pipeline Stages

Run `dvc dag` to visualize:

```text
data_ingestion
    ↓
data_preprocessing
    ↓
feature_engineering
    ↓           \
model_building  |
    ↓           ↓
model_evaluation
```

---

## 📌 Notable Engineering Practices

* **Logging:** Each script logs both to console and file inside `/logs/`
* **DVC + Git Integration:** Track experiment history and changes easily
* **Custom Feature Creation:** Ratios and compound social behavior metrics
* **Cross-Validation:** Ensures robust, unbiased performance estimation

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

* [Kaggle Playground Series S5E7](https://www.kaggle.com/competitions/playground-series-s5e7)
* [DVC](https://dvc.org/)
* [dvclive](https://dvc.org/doc/dvclive)
* [XGBoost](https://xgboost.ai/)
* [scikit-learn](https://scikit-learn.org/)

---



