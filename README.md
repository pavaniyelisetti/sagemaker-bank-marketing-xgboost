# End-to-End ML Pipeline with Amazon SageMaker
### Bank Marketing Subscription Prediction · XGBoost · Real-Time Inference

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-FF9900?logo=amazonaws)](https://aws.amazon.com/sagemaker/)
[![XGBoost](https://img.shields.io/badge/Algorithm-XGBoost-green)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

This project demonstrates a production-style, end-to-end machine learning pipeline built on **Amazon SageMaker**. Using the UCI Bank Marketing dataset (41,188 records, 62 features), I train an XGBoost binary classifier to predict whether a customer will subscribe to a term deposit — then deploy it as a live REST endpoint for real-time inference.

The workflow mirrors how ML is shipped in industry: data lives in S3, training runs on managed compute, and predictions are served through a scalable endpoint — all orchestrated without managing servers.

---

## Architecture

```
Raw Data (S3 URL)
      │
      ▼
┌─────────────────┐
│  Preprocessing  │  shuffle · split 70/30 · label-first CSV format
└────────┬────────┘
         │  upload
         ▼
┌─────────────────┐
│   Amazon S3     │  train/ · test/
└────────┬────────┘
         │  TrainingInput
         ▼
┌──────────────────────────────┐
│  SageMaker Training Job      │  XGBoost 1.7-1 · ml.m5.large
│  (Managed, containerized)    │  binary:logistic · 50 rounds
└────────┬─────────────────────┘
         │  model artifacts → S3/output
         ▼
┌─────────────────────┐
│  SageMaker Endpoint │  ml.t2.medium · CSVSerializer
│  (Real-time)        │  → P(subscribe) per customer
└─────────────────────┘
```

---

## Dataset

**UCI Bank Marketing Dataset** — Portuguese banking institution phone campaign records.

| Property | Value |
|---|---|
| Records | 41,188 |
| Features (post one-hot) | 61 |
| Target | `y_yes` — term deposit subscription (binary) |
| Class imbalance | ~11.3% positive |
| Source | [AWS Tutorial Dataset](https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv) |

Feature groups include customer demographics (age, job, marital status, education), financial indicators (default, housing, loan), and campaign contact metadata (month, day of week, previous outcome).

---

## Pipeline Steps

**1. Environment Setup** — initialize SageMaker session, boto3 client, retrieve default S3 bucket and IAM execution role.

**2. Data Loading & EDA** — load the pre-cleaned CSV into pandas; inspect shape, dtypes, and summary statistics across all 62 columns.

**3. Preprocessing** — drop the index column, shuffle with a fixed random seed, perform a 70/30 train-test split, and reorder columns so the label (`y_yes`) is first (XGBoost CSV format requirement).

**4. S3 Upload** — write headerless CSVs to `s3://<bucket>/bank-marketing-xgboost/train/` and `.../test/` using boto3.

**5. Training Job Configuration** — retrieve the XGBoost 1.7-1 ECR container URI for the region, set hyperparameters, create `TrainingInput` channels, and configure a SageMaker `Estimator` on `ml.m5.large`.

**6. Model Training** — call `estimator.fit()` to launch a fully managed training job; SageMaker provisions compute, pulls data from S3, trains, and saves model artifacts back to S3.

**7. Endpoint Deployment** — deploy the trained model to an `ml.t2.medium` real-time endpoint with a `CSVSerializer`/`CSVDeserializer`.

**8. Inference** — send a single test record to the endpoint and receive the predicted subscription probability.

**9. Cleanup** — delete the endpoint to stop billing.

---

## Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `max_depth` | 5 | Controls tree depth |
| `eta` | 0.2 | Learning rate |
| `gamma` | 4 | Min loss reduction for split |
| `min_child_weight` | 6 | Min sum of instance weight per leaf |
| `subsample` | 0.7 | Row sampling ratio |
| `objective` | `binary:logistic` | Binary classification |
| `num_round` | 50 | Boosting rounds |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| ML Framework | XGBoost 1.7-1 (SageMaker built-in) |
| Cloud | AWS SageMaker, S3, IAM, STS |
| SDK | `sagemaker` 2.257, `boto3` 1.43 |
| Data | pandas, NumPy |
| Compute (training) | `ml.m5.large` |
| Compute (inference) | `ml.t2.medium` |

---

## Project Structure

```
├── sagemaker_end2enddeploy_basics_portfolio.ipynb   # Full pipeline notebook
├── requirements.txt                                  # Python dependencies
└── README.md
```

> **Note:** `train.csv` and `test.csv` are generated at runtime and excluded from the repo (see `.gitignore`). AWS credentials and account IDs are never stored here.

---

## How to Run

### Prerequisites
- An AWS account with SageMaker access
- An IAM role with `AmazonSageMakerFullAccess` and S3 read/write permissions
- A SageMaker Studio or notebook instance (the notebook is designed to run inside SageMaker)

### Steps

```bash
# Clone the repo
git clone https://github.com/pavaniyelisetti/sagemaker-bank-marketing-xgboost.git
cd sagemaker-bank-marketing-xgboost

# Install dependencies (if running locally with SageMaker SDK)
pip install -r requirements.txt
```

Open `sagemaker_end2enddeploy_basics_portfolio.ipynb` in SageMaker Studio or a SageMaker Notebook Instance and run cells top-to-bottom.

> **Cost note:** Running this notebook incurs AWS charges for the training job (`ml.m5.large`) and inference endpoint (`ml.t2.medium`). The final cleanup cell deletes the endpoint — always run it after testing.

---

## Key Takeaways

- SageMaker's built-in XGBoost container eliminates environment setup overhead and runs on fully managed, auto-provisioned compute.
- Separating data (S3), training (managed job), and inference (endpoint) makes each stage independently scalable and auditable.
- The label-first CSV convention is a subtle but critical requirement for SageMaker's built-in algorithms — a common source of errors for first-time users.
- Real-time endpoints return a probability score, enabling downstream business logic (e.g., threshold tuning for precision/recall trade-off on an imbalanced target).

---

## Related Projects

- [Heat Pump Analysis & Load Forecasting](https://github.com/pavaniyelisetti/Heat-pump-analysis-and-load-forecasting) — 700M+ smart meter records, customer segmentation
- [Electricity Price Forecasting](https://github.com/pavaniyelisetti/price_forecasting) — ensemble methods for energy market bidding
- [Scheduling Optimization](https://github.com/pavaniyelisetti/scheduling_optimization) — Gurobi-based backlog reduction

---

## Author

**Pavani Yelisetti** — Data Scientist | Energy Analytics & ML | UNC Charlotte · Duke Energy

[![LinkedIn](https://img.shields.io/badge/LinkedIn-pavani--yelisetti-blue?logo=linkedin)](https://www.linkedin.com/in/pavani-yelisetti)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-blueviolet)](https://my-personal-portfolio-phi-gules.vercel.app/)
[![GitHub](https://img.shields.io/badge/GitHub-pavaniyelisetti-black?logo=github)](https://github.com/pavaniyelisetti)

---

*MIT License · Feel free to fork and adapt for your own SageMaker projects.*
