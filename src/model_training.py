"""
Train ML models for house price prediction
Uses parallel processing and model versioning
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from pathlib import Path
import time
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

from preprocessing import Preprocessor
from model_registry import ModelRegistry

N_JOBS = max(1, multiprocessing.cpu_count() - 1)


def get_metrics(y_true, y_pred):
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'mape': float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    }


def get_models():
    return {
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=0.1),
        'elasticnet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'rf': RandomForestRegressor(
            n_estimators=100, max_depth=15,
            min_samples_split=5, n_jobs=N_JOBS, random_state=42
        ),
        'gbm': GradientBoostingRegressor(
            n_estimators=100, max_depth=6,
            learning_rate=0.1, random_state=42
        ),
        'xgb': XGBRegressor(
            n_estimators=100, max_depth=6,
            learning_rate=0.1, n_jobs=N_JOBS,
            random_state=42, verbosity=0
        ),
        'lgbm': LGBMRegressor(
            n_estimators=100, max_depth=6,
            learning_rate=0.1, n_jobs=N_JOBS,
            random_state=42, verbosity=-1
        )
    }


def train_models(X_train, y_train, X_val, y_val):
    models = get_models()
    results = {}

    print(f"\nTraining {len(models)} models (using {N_JOBS} cores)...")
    print("-" * 60)

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        results[name] = {
            'model': model,
            'train': get_metrics(y_train, train_pred),
            'val': get_metrics(y_val, val_pred),
            'time': elapsed
        }
        val = results[name]['val']
        print(f"  {name}: rmse=${val['rmse']:,.0f} mae=${val['mae']:,.0f} ({elapsed:.2f}s)")

    return results


def select_best_model(results):
    """Select best model based on RMSE and MAE (weighted average rank)"""
    # rank models by each metric
    names = list(results.keys())
    rmse_vals = [(n, results[n]['val']['rmse']) for n in names]
    mae_vals = [(n, results[n]['val']['mae']) for n in names]

    rmse_rank = {n: i for i, (n, _) in enumerate(sorted(rmse_vals, key=lambda x: x[1]))}
    mae_rank = {n: i for i, (n, _) in enumerate(sorted(mae_vals, key=lambda x: x[1]))}

    # combined score (lower is better)
    scores = {n: rmse_rank[n] + mae_rank[n] for n in names}
    best = min(scores, key=scores.get)

    print(f"\nModel Selection (RMSE + MAE ranking):")
    print("-" * 60)
    for n in sorted(scores, key=scores.get):
        val = results[n]['val']
        print(f"  {n}: rmse_rank={rmse_rank[n]+1} mae_rank={mae_rank[n]+1} "
              f"-> score={scores[n]} | rmse=${val['rmse']:,.0f} mae=${val['mae']:,.0f}")

    return best


def tune_model(model_name, X_train, y_train):
    print(f"\nTuning {model_name}...")

    param_grids = {
        'lgbm': {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [31, 50]
        },
        'xgb': {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0]
        },
        'rf': {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5]
        }
    }

    if model_name not in param_grids:
        print(f"  No param grid for {model_name}, skipping")
        return None, None

    base_model = get_models()[model_name]

    grid = GridSearchCV(
        base_model, param_grids[model_name],
        cv=5, scoring='neg_mean_absolute_error',  # use MAE for tuning
        n_jobs=N_JOBS, verbose=1
    )

    t0 = time.time()
    grid.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV MAE: ${-grid.best_score_:,.0f}")

    return grid.best_estimator_, grid.best_params_


def main(promote_to_prod=True):
    print("=" * 60)
    print("HOUSE PRICE PREDICTION - MODEL TRAINING")
    print("=" * 60)
    print(f"CPUs: {multiprocessing.cpu_count()} (using {N_JOBS})")

    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "Case Study 1 Data (1).xlsx"
    output_dir = base_dir / "models"
    output_dir.mkdir(exist_ok=True)

    # init model registry
    registry = ModelRegistry(str(output_dir / "registry"))

    # load and prep data
    print("\nLoading data...")
    df = pd.read_excel(data_path)
    print(f"  {len(df):,} rows")

    prep = Preprocessor()
    X, y, features = prep.fit_transform(df)
    print(f"  {X.shape[1]} features")

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    print(f"\nSplit: train={len(y_train):,} val={len(y_val):,} test={len(y_test):,}")

    # train all models
    results = train_models(X_train, y_train, X_val, y_val)

    # select best using RMSE + MAE
    best_name = select_best_model(results)
    print(f"\n>> Best model: {best_name}")

    # tune best model
    tuned, best_params = tune_model(best_name, X_train, y_train)
    if tuned is not None:
        best_model = tuned
    else:
        best_model = results[best_name]['model']
        best_params = {}

    # final eval on test set
    test_pred = best_model.predict(X_test)
    test_metrics = get_metrics(y_test, test_pred)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"RMSE: ${test_metrics['rmse']:,.0f}")
    print(f"MAE:  ${test_metrics['mae']:,.0f}")
    print(f"R2:   {test_metrics['r2']:.4f}")
    print(f"MAPE: {test_metrics['mape']:.1f}%")

    # register model version (registry is the single source of truth)
    print("\n" + "=" * 60)
    print("MODEL VERSIONING")
    print("=" * 60)

    prep_data = {
        'encoders': prep.encoders,
        'scaler': prep.scaler,
        'feature_cols': prep.feature_cols,
        'num_cols': prep.num_cols,
        'cat_cols': prep.cat_cols
    }

    version = registry.register(
        model=best_model,
        model_name="house_price_predictor",
        metrics=test_metrics,
        params=best_params or {},
        tags=[best_name, "tuned", f"n_features={len(features)}"],
        preprocessor=prep_data
    )

    if promote_to_prod:
        registry.promote("house_price_predictor", version, stage="production")

    # feature importance
    if hasattr(best_model, 'feature_importances_'):
        imp = pd.DataFrame({
            'feature': features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"\nTop features:\n{imp.head(10).to_string(index=False)}")
        imp.to_csv(output_dir / 'feature_importance.csv', index=False)

    # save comparison
    comparison = []
    for name, res in results.items():
        comparison.append({
            'model': name,
            'val_rmse': res['val']['rmse'],
            'val_mae': res['val']['mae'],
            'val_r2': res['val']['r2'],
            'time_s': res['time']
        })
    pd.DataFrame(comparison).to_csv(output_dir / 'model_comparison.csv', index=False)

    # test predictions
    pd.DataFrame({
        'actual': y_test,
        'predicted': test_pred,
        'error': y_test - test_pred
    }).to_csv(output_dir / 'test_predictions.csv', index=False)

    # show registry summary
    registry.summary()

    print("\nDone!")
    return registry, version


if __name__ == "__main__":
    main()
