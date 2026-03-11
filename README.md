# 🏭 AI-Driven Manufacturing Intelligence Dashboard
> Code4Carbon | AVEVA National AI/ML Hackathon | Track A: Predictive Modelling

## 📌 Problem Statement
Predict multiple pharmaceutical manufacturing quality outcomes simultaneously 
while minimising energy consumption and CO₂ emissions.

## 🚀 Features
- **Multi-target prediction** — Quality, Yield, Performance, Process Efficiency
- **Energy Pattern Intelligence** — Phase-level power analysis across 8 manufacturing phases
- **SHAP Explainability** — Feature importance per target
- **Golden Signature Management** — Human-in-the-loop optimal batch tracking
- **Adaptive Carbon Targets** — Aligned with India Net Zero 2070

## 🤖 ML Models
| Target | Model | CV R² | Test R² |
|---|---|---|---|
| Content Uniformity | RandomForest | 0.9727 | 0.9782 |
| Dissolution Rate | RandomForest | 0.9668 | 0.9625 |
| Friability | RandomForest | 0.9754 | 0.9784 |
| Disintegration Time | RandomForest | 0.9819 | 0.9666 |

- 80/20 train-test split (48 train / 12 test batches)
- 5-fold cross validation
- Competitive selection between XGBoost and RandomForest per target

## 📁 Project Structure
```
manufacturing-ai/
├── data/
│   ├── _h_batch_process_data.xlsx
│   └── _h_batch_production_data.xlsx
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── data_pipeline.py
├── app/
│   ├── dashboard.py
│   └── api.py
├── models/
│   ├── multi_target_models.pkl
│   ├── scaler.pkl
│   └── model_meta.json
├── requirements.txt
└── README.md
```

## ⚙️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/Sai-Ramya-prog/ai-manufacturing-intelligence.git
cd ai-manufacturing-intelligence
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the models
```bash
python src/train_model.py
```

### 4. Run the dashboard
```bash
streamlit run app/dashboard.py
```

### 5. (Optional) Run the API
```bash
uvicorn app.api:app --port 8000
```

## 📦 Requirements
- Python 3.11+
- streamlit
- scikit-learn
- xgboost
- shap
- plotly
- pandas
- numpy
- joblib
- openpyxl
- fastapi
- uvicorn

## 🌿 Carbon Impact
- Baseline: ~62.9 kg CO₂/batch
- 2030 Target (−45%): ~34.6 kg CO₂/batch
- At 300 batches/year: potential saving of ~8,500 kg CO₂/year
- Emission factor: 0.82 kg CO₂/kWh (India grid average)

## 👥 Team
Code4Carbon — AVEVA National AI/ML Hackathon 2025

