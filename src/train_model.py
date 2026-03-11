import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import shap

from preprocessing import load_and_prepare, get_feature_target_split

BASE_DIR  = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

EMISSION_FACTOR = 0.82


def train(X_train, y_train, feature_cols, target_cols):
    print("=" * 60)
    print("TRAINING — 48 batches (train) | 12 batches (test)")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    models  = {}
    metrics = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for target in target_cols:
        y_t = y_train[target].values
        print(f"\n--- Target: {target} ---")

        xgb_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=3,
            learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, random_state=42, verbosity=0
        )
        rf_model = RandomForestRegressor(
            n_estimators=100, max_depth=5,
            min_samples_split=4, random_state=42
        )

        xgb_cv = cross_val_score(xgb_model, X_train_sc, y_t, cv=kf, scoring='r2').mean()
        rf_cv  = cross_val_score(rf_model,  X_train_sc, y_t, cv=kf, scoring='r2').mean()

        if xgb_cv >= rf_cv:
            best_model, best_name, best_cv = xgb_model, "XGBoost", xgb_cv
        else:
            best_model, best_name, best_cv = rf_model,  "RandomForest", rf_cv

        best_model.fit(X_train_sc, y_t)

        models[target]  = best_model
        metrics[target] = {
            'model':   best_name,
            'cv_r2':   round(float(best_cv), 4),
        }
        print(f"  Best : {best_name} | CV R²: {best_cv:.4f}")

    return models, scaler, metrics


def evaluate_on_test(models, scaler, X_test, y_test, target_cols, metrics):
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION — 12 unseen batches")
    print("=" * 60)

    X_test_sc = scaler.transform(X_test)

    for target in target_cols:
        y_true = y_test[target].values
        y_pred = models[target].predict(X_test_sc)

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100

        metrics[target]['test_r2']   = round(float(r2),   4)
        metrics[target]['test_mae']  = round(float(mae),  4)
        metrics[target]['test_rmse'] = round(float(rmse), 4)
        metrics[target]['test_mape'] = round(float(mape), 2)

        print(f"\n  {target}")
        print(f"    Test R²  : {r2:.4f}")
        print(f"    Test MAE : {mae:.4f}")
        print(f"    Test RMSE: {rmse:.4f}")
        print(f"    Test MAPE: {mape:.2f}%")

        # Actual vs predicted for 5 sample batches
        print(f"    Sample predictions (actual → predicted):")
        for i in range(min(5, len(y_true))):
            print(f"      Batch {i+1}: {y_true[i]:.3f} → {y_pred[i]:.3f}")

    return metrics


def compute_shap(models, scaler, X_train, feature_cols, target_cols):
    print("\n" + "=" * 60)
    print("SHAP FEATURE IMPORTANCE")
    print("=" * 60)

    X_sc = scaler.transform(X_train)
    shap_importance = {}

    for target in target_cols:
        try:
            explainer  = shap.TreeExplainer(models[target])
            shap_vals  = explainer.shap_values(X_sc)
            importance = np.abs(shap_vals).mean(axis=0)
            top8 = dict(sorted(
                zip(feature_cols, importance.tolist()),
                key=lambda x: x[1], reverse=True
            )[:8])
            shap_importance[target] = top8
            print(f"\n  Top features → {target}:")
            for f, v in top8.items():
                print(f"    {f:35s}: {v:.4f}")
        except Exception as e:
            if hasattr(models[target], 'feature_importances_'):
                imp  = models[target].feature_importances_
                top8 = dict(sorted(
                    zip(feature_cols, imp.tolist()),
                    key=lambda x: x[1], reverse=True
                )[:8])
                shap_importance[target] = top8

    return shap_importance


def build_golden_signature(models, scaler, X, y, feature_cols, target_cols):
    X_sc = scaler.transform(X)
    scores = []
    for i in range(len(X)):
        cu = y['Content_Uniformity'].iloc[i] if 'Content_Uniformity' in y else 98
        dr = y['Dissolution_Rate'].iloc[i]    if 'Dissolution_Rate'   in y else 90
        fr = y['Friability'].iloc[i]          if 'Friability'         in y else 0.5
        co = y['co2_kg'].iloc[i]              if 'co2_kg'             in y else 40
        score = (cu / 106.3) + (dr / 99.9) - (fr / 2.0) - (co / 100)
        scores.append(score)

    best_idx     = int(np.argmax(scores))
    golden_params  = X.iloc[best_idx].to_dict()
    golden_targets = y.iloc[best_idx].to_dict()

    print(f"\n  Golden batch index: {best_idx}")
    print("  Golden targets:")
    for t, v in golden_targets.items():
        print(f"    {t}: {v:.3f}")

    return {
        'parameters':       {k: round(float(v), 4) for k, v in golden_params.items()},
        'targets_achieved': {k: round(float(v), 4) for k, v in golden_targets.items()},
        'batch_index':      best_idx
    }


if __name__ == "__main__":
    # Load data
    df = load_and_prepare()
    X, y, feature_cols, target_cols = get_feature_target_split(df)

    print(f"Total dataset : {X.shape[0]} batches × {X.shape[1]} features")
    print(f"Targets       : {target_cols}\n")

    # ── 80/20 train-test split ──────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)} batches | Test: {len(X_test)} batches\n")

    # Train
    models, scaler, metrics = train(X_train, y_train, feature_cols, target_cols)

    # Evaluate on held-out test set
    metrics = evaluate_on_test(models, scaler, X_test, y_test, target_cols, metrics)

    # SHAP on training data
    shap_importance = compute_shap(models, scaler, X_train, feature_cols, target_cols)

    # Golden signature on full dataset
    golden = build_golden_signature(models, scaler, X, y, feature_cols, target_cols)

    # Retrain final model on full data for deployment
    print("\n" + "=" * 60)
    print("RETRAINING ON FULL DATA FOR DEPLOYMENT")
    print("=" * 60)
    scaler_final = StandardScaler()
    X_all_sc = scaler_final.fit_transform(X)
    for target in target_cols:
        models[target].fit(X_all_sc, y[target].values)
        print(f"  {target} — retrained on all 60 batches")

    # Save
    joblib.dump(models,       MODELS_DIR / "multi_target_models.pkl")
    joblib.dump(scaler_final, MODELS_DIR / "scaler.pkl")

    meta = {
        'feature_cols':    feature_cols,
        'target_cols':     target_cols,
        'metrics':         metrics,
        'shap_importance': shap_importance,
        'golden_signature':golden,
        'dataset_info': {
            'total_batches': len(X),
            'train_batches': len(X_train),
            'test_batches':  len(X_test),
            'n_features':    len(feature_cols),
            'note': 'Features from real production data only. Energy features from real process sensor data.'
        }
    }
    with open(MODELS_DIR / "model_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ All models saved to {MODELS_DIR}")
    print("Run: streamlit run app/dashboard.py")