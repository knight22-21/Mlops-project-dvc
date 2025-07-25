

# MLOps with DVC: A Reproducible Pipeline Demo

[![GitHub Repo](https://img.shields.io/badge/GitHub-knight22--21/Mlops--project--dvc-blue?logo=github)](https://github.com/knight22-21/Mlops-project-dvc)

This project showcases how to build a **version-controlled, reproducible machine learning pipeline** using [DVC (Data Version Control)](https://dvc.org/) — from raw data ingestion to model evaluation. It uses a classification problem (Introvert vs Extrovert) as a base use case, but the focus is on the **DVC-driven workflow**, not the ML model itself.


---

A modular and production-ready machine learning pipeline for classifying individuals as **introverts or extroverts**, built for the [Kaggle Playground Series - Season 5, Episode 7](https://www.kaggle.com/competitions/playground-series-s5e7).

This project integrates **DVC** for data and experiment versioning, **dvclive** for live metric tracking, and **XGBoost** for high-performance classification, with clean modular Python code and logging.

---

## Project Structure

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

## Key Objectives

✅ Build a modular ML pipeline  
✅ Version data, models, and experiments using DVC  
✅ Automate the workflow using `dvc.yaml`  
✅ Track experiments with `dvclive`  
✅ Use Git + DVC to manage reproducible workflows

---

## Tools & Frameworks

- **DVC** for pipeline, data & experiment versioning
- **dvclive** for live metrics tracking
- **XGBoost** as the model (optional placeholder)
- **scikit-learn**, **pandas**, **YAML configs**
- **Python logging** for traceability

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

## Notable Engineering Practices

* **Logging:** Each script logs both to console and file inside `/logs/`
* **DVC + Git Integration:** Track experiment history and changes easily
* **Custom Feature Creation:** Ratios and compound social behavior metrics
* **Cross-Validation:** Ensures robust, unbiased performance estimation

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

* [Kaggle Playground Series S5E7](https://www.kaggle.com/competitions/playground-series-s5e7)
* [DVC](https://dvc.org/)
* [dvclive](https://dvc.org/doc/dvclive)
* [XGBoost](https://xgboost.ai/)
* [scikit-learn](https://scikit-learn.org/)

---



