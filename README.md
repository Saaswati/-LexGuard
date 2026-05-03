# LexGuard — AI Legal Document Analyzer

An intelligent NLP system that analyzes legal clauses, classifies risk levels, extracts legal entities, and generates summaries — trained on real SEC legal contracts.

## Results

| Metric | Score |
|---|---|
| Accuracy | 93.88% |
| Precision | 0.9604 |
| Recall | 0.9435 |
| F1 Score | 0.9519 |
| AUC-ROC | 0.9819 |
| CV Accuracy | 98.79% |

## Dataset

**LEDGAR** — 80,000 real legal clauses from SEC filings (US Securities and Exchange Commission). Sourced from `coastalcph/lex_glue` on HuggingFace.

## What It Does

- **Clause Risk Classification** — classifies legal clauses as HIGH RISK or LOW RISK using TF-IDF + Logistic Regression
- **Legal Entity Extraction** — extracts 7 entity types: MONETARY, DATE, PARTY, JURISDICTION, OBLIGATION, PROHIBITION, LEGAL_REF
- **Document Summarization** — extractive summarization evaluated with BLEU and ROUGE scores
- **Explainability** — attention entropy heatmaps showing which tokens drive risk predictions
- **Embedding Visualization** — PCA and t-SNE plots of legal clause semantic space

## Tech Stack

Python · Scikit-learn · TF-IDF · Logistic Regression · BLEU · ROUGE · Perplexity · Attention Entropy · PCA · t-SNE · Matplotlib · Seaborn · HuggingFace Datasets

## Project Structure
LexGuard/
├── lexguard.ipynb          # Full Kaggle notebook
├── lexguard_cells.py       # Python source code
└── README.md

## How to Run

**On Kaggle:**
1. Create a new notebook
2. Enable Internet — Settings → Internet ON
3. Paste the code cells
4. Click Run All

**Locally:**
pip install datasets scikit-learn matplotlib seaborn pandas numpy scipy
python lexguard_cells.py

## Visualizations Generated

- Confusion Matrix
- ROC Curve
- 5-Fold Cross Validation
- Top 20 Feature Importance
- PCA + t-SNE Embedding Space
- NLP Metrics Distribution
- Risk Probability Distribution
- Clause Type Risk Analysis
- Legal Entity Distribution
- Attention Heatmaps
- ROUGE Score Comparison

## Author

**Saaswati Chinni** — B.Tech CSE AI/ML | Lovely Professional University
