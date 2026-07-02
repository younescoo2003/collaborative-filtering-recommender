# 🎬 Movie Recommendation System – Collaborative Filtering

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> A movie recommendation engine built with collaborative filtering using the Surprise library. Implements SVD (matrix factorization) and KNN (user‑based & item‑based) models, evaluated on the MovieLens 100k dataset.

---

## 📊 Project Overview

This project demonstrates how to build a **recommendation system** using **collaborative filtering** techniques. It uses the **Surprise** library to train and evaluate different algorithms:

- **SVD** (Singular Value Decomposition) – matrix factorization
- **KNN User‑Based** – recommend items that similar users liked
- **KNN Item‑Based** – recommend items similar to those the user liked
- **Random** – baseline for comparison

The models are evaluated on the **MovieLens 100k** dataset (100,000 ratings from 943 users on 1,682 movies).

---

## 🎯 Key Features

- ✅ Loads MovieLens dataset (built‑in with Surprise)
- ✅ Trains and compares multiple collaborative filtering models
- ✅ Evaluates using RMSE and MAE (cross‑validation and train‑test split)
- ✅ Provides top‑N recommendations for any user
- ✅ Visualizes model performance with bar charts
- ✅ Modular, reusable Python code (`src/`)

---

## 🧪 Results Summary

| Model          | RMSE (test) | MAE (test) |
|----------------|-------------|------------|
| **SVD**        | **~0.94**   | **~0.74**  |
| KNN‑User       | ~1.01       | ~0.79      |
| KNN‑Item       | ~1.05       | ~0.82      |
| Random (baseline) | ~1.51   | ~1.21      |

> SVD outperforms all others – it captures latent factors between users and items.

---

## 🛠️ Tech Stack

| Tool / Library | Purpose |
|----------------|---------|
| Python 3.8+ | Programming language |
| Pandas, NumPy | Data manipulation |
| Scikit‑Surprise | Collaborative filtering algorithms |
| Matplotlib, Seaborn | Visualisations |
| Jupyter Notebook | Interactive development & exploration |

---

## 📁 Project Structure

collaborative-filtering-recommender/
├── data/
│ ├── raw/ # (dataset downloaded automatically)
│ └── processed/ # (processed data – not used here)
├── notebooks/
│ └── 01_recommender.ipynb # Main analysis & model training
├── src/
│ ├── init.py
│ ├── data_loader.py # Load MovieLens dataset
│ ├── recommender.py # Model creation functions
│ └── evaluator.py # Evaluation & comparison utilities
├── tests/
│ └── test_recommender.py # (placeholder for future tests)
├── results/
│ ├── plots/ # Saved figures 
│ └── reports/ # (future report outputs)
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git
- (Optional) virtual environment

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/younESCOO2003/collaborative-filtering-recommender.git
   cd collaborative-filtering-recommender