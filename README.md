# FairLens AI

**Detect Bias Before It Detects You**

Most machine learning models are tested for accuracy. Almost none are tested for fairness. FairLens AI fixes that — upload a dataset, and within seconds you'll know exactly which groups your model is treating unfairly, by how much, and what to do about it.

🔗 **Live:** https://fairlens-production-8320.up.railway.app

---

## The Problem

AI models are making real decisions — who gets hired, who gets a loan, who gets flagged by a system. These models are usually evaluated on accuracy alone. A model can be 85% accurate and still systematically disadvantage women, older applicants, or minority groups. Most teams don't discover this until after deployment.

FairLens makes fairness auditing accessible to any developer, not just researchers with fairness expertise.

---

## What It Does

Upload any CSV dataset. Select the outcome column you want to predict (like income, loan approval, hiring decision) and the demographic columns you want to audit (like age, gender, race). FairLens will:

- Train a machine learning model on your data
- Measure how fairly it treats different groups using industry-standard fairness metrics
- Show you exactly which groups are being disadvantaged and by how much
- Explain the findings in plain English using Google Gemini
- Let you apply mitigation strategies and immediately see the before/after impact

The whole workflow — upload, audit, fix, compare — runs in one browser tab with no setup needed.

---

## UN Sustainable Development Goals

| Goal | Connection |
|------|-----------|
| **SDG 10 — Reduced Inequalities** | Directly detects when AI systems produce discriminatory outcomes across gender, age, race, or other demographic groups |
| **SDG 8 — Decent Work and Economic Growth** | Audits hiring and income prediction models where biased AI causes real economic harm to individuals |
| **SDG 16 — Peace, Justice and Strong Institutions** | Promotes transparent and accountable AI in institutional decision-making |

---

## Demo

No dataset? No problem. Click **View Sample Demo** on the landing page — the full workflow runs instantly on built-in demo data showing a gender-biased loan approval model.

Or try it with the classic UCI Adult Income dataset (adult.csv) — upload it, select `income` as the target and `age` or `sex` as sensitive features, and see real bias findings in seconds.

---

## How It Works

```
Upload CSV
    ↓
Auto-detect target + sensitive columns
    ↓
Train Random Forest model (scikit-learn)
    ↓
Measure Demographic Parity + Equalized Odds (fairlearn)
    ↓
Generate plain-English insights (Google Gemini)
    ↓
Show bias score, group charts, alerts
    ↓
Apply mitigation (remove feature / rebalance / fairness constraint)
    ↓
Compare before vs after
```

### Fairness Metrics

**Demographic Parity Difference** — are prediction rates equal across groups? A score of 0.75 means one group is predicted a positive outcome 75% less often than another.

**Equalized Odds Difference** — are error rates equal across groups? Catches cases where the model is wrong more often for one group than another.

**Bias Score** — combined severity: Low (under 0.12), Moderate (under 0.25), High (0.25 and above).

### Mitigation Options

**Remove sensitive feature** — excludes the demographic column from model inputs entirely.

**Rebalance dataset** — upsamples under-represented groups so the model trains on more equal footing.

**Apply fairness constraint** — trains the model with a mathematical constraint that directly penalizes parity gaps. Slowest but most effective.

---

## Tech Stack

**Frontend:** React 18 + Vite — single-page app, no page reloads, works on any device

**Backend:** FastAPI (Python) — async REST API with automatic validation

**ML:** scikit-learn — Random Forest inside a Pipeline so preprocessing never leaks between train and test sets

**Fairness:** Microsoft fairlearn — the same library used by industry data science teams

**AI:** Google Gemini API — generates plain-English insights and mitigation recommendations from raw fairness numbers

**Deployment:** Docker on Railway — single container serves both the React frontend and Python backend from one URL

---

## Project Structure

```
fairlens/
├── backend/
│   ├── main.py                  # App entry point, CORS, health checks, static file serving
│   ├── requirements.txt
│   ├── models/schema.py         # Pydantic response models
│   ├── routes/analyze.py        # POST /api/analyze and /api/mitigate endpoints
│   ├── services/
│   │   ├── bias_detection.py    # Full ML pipeline + Gemini integration
│   │   └── mitigation.py        # Bias fix strategies
│   └── utils/preprocessing.py   # Data normalization, group metrics, fairness helpers
├── src/
│   └── App.jsx                  # React frontend — all 6 screens in one component
├── Dockerfile                   # Builds React then starts FastAPI — one container
├── railway.toml                 # Deployment config
└── vite.config.js               # Dev proxy: forwards /api to backend port
```

---

## Running Locally

**Requirements:** Python 3.11+, Node.js 18+

```bash
git clone https://github.com/ruchikamakkar01-creator/fairlens.git
cd fairlens

# Python setup
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
pip install -r backend/requirements.txt

# Node setup
npm install

# Optional: add Gemini key for AI insights
# Create a .env file and add:
# GEMINI_API_KEY=your_key_here
# Get a free key at https://aistudio.google.com/app/apikey

# Start backend (Terminal 1)
python -m uvicorn backend.main:app --reload --port 8001

# Start frontend (Terminal 2)
npm run dev

# Open http://localhost:5173
```

The app works without a Gemini key — it falls back to rule-based insights automatically.

---

## API

```
POST /api/analyze
  dataset            → CSV file
  target_column      → column to predict (e.g. "income")
  sensitive_features → JSON array (e.g. ["age", "sex"])
  Returns: bias score, fairness metrics, group rates, insights, alerts, recommendation

POST /api/mitigate
  dataset            → CSV file
  target_column      → column to predict
  sensitive_features → JSON array
  fixes              → JSON object e.g. {"sensitive": true, "rebalance": true, "constraint": false}
  Returns: updated metrics after mitigation

GET /api/health
  Returns: {"ok": true, "status": "healthy"}
```

---

## Google Solutions Challenge 2026

Built for the Google Solutions Challenge 2026.

This project uses **Google Gemini** to bridge the gap between raw fairness numbers and human understanding — turning metrics like "Demographic Parity Difference: 0.75" into actionable plain-English insights that non-technical stakeholders can act on.

---

## Acknowledgements

- [Microsoft fairlearn](https://fairlearn.org/) for fairness metrics
- [Google Gemini](https://ai.google.dev/) for AI-powered insights
- [scikit-learn](https://scikit-learn.org/) for the ML pipeline
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
