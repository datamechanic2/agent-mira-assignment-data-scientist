"""
Train ML models for house price prediction
Uses stacking ensemble, parallel processing, and model versioning
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from pathlib import Path
import joblib
import time
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

from preprocessing import ImprovedPreprocessor
from model_registry import ModelRegistry

N_JOBS = max(1, multiprocessing.cpu_count() - 1)


def get_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'mape': float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    }


def get_base_models():
    """Get individual models for comparison"""
    return {
        'lgbm': LGBMRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            num_leaves=50, n_jobs=N_JOBS, random_state=42, verbosity=-1
        ),
        'xgb': XGBRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            n_jobs=N_JOBS, random_state=42, verbosity=0
        ),
        'gbm': GradientBoostingRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
        'rf': RandomForestRegressor(
            n_estimators=300, max_depth=15, min_samples_split=5,
            n_jobs=N_JOBS, random_state=42
        ),
    }


def create_stacking_model():
    """Create a stacking ensemble with multiple base models"""
    base_models = [
        ('lgbm', LGBMRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            num_leaves=50, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=N_JOBS, random_state=42, verbosity=-1
        )),
        ('xgb', XGBRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=N_JOBS, random_state=42, verbosity=0
        )),
        ('gbm', GradientBoostingRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, random_state=42
        )),
        ('rf', RandomForestRegressor(
            n_estimators=300, max_depth=15, min_samples_split=5,
            n_jobs=N_JOBS, random_state=42
        )),
    ]

    return StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=N_JOBS,
        passthrough=True
    )


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """Train individual models and stacking ensemble, return best"""
    print("\n" + "-" * 60)
    print("Training individual models...")
    print("-" * 60)

    models = get_base_models()
    results = {}
    best_mae = float('inf')
    best_model = None
    best_name = None

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = get_metrics(y_test, pred)
        elapsed = time.time() - t0

        results[name] = {'model': model, 'metrics': metrics, 'time': elapsed}
        print(f"  {name}: MAE=${metrics['mae']:,.0f} RMSE=${metrics['rmse']:,.0f} "
              f"R²={metrics['r2']:.4f} ({elapsed:.1f}s)")

        if metrics['mae'] < best_mae:
            best_mae = metrics['mae']
            best_model = model
            best_name = name

    # Train stacking ensemble
    print("\n" + "-" * 60)
    print("Training Stacking Ensemble...")
    print("-" * 60)

    t0 = time.time()
    stacking = create_stacking_model()
    stacking.fit(X_train, y_train)
    stack_pred = stacking.predict(X_test)
    stack_metrics = get_metrics(y_test, stack_pred)
    elapsed = time.time() - t0

    results['stacking'] = {'model': stacking, 'metrics': stack_metrics, 'time': elapsed}
    print(f"  Stacking: MAE=${stack_metrics['mae']:,.0f} RMSE=${stack_metrics['rmse']:,.0f} "
          f"R²={stack_metrics['r2']:.4f} ({elapsed:.1f}s)")

    # Choose best
    if stack_metrics['mae'] < best_mae:
        best_model = stacking
        best_name = 'stacking'
        best_metrics = stack_metrics
    else:
        best_metrics = results[best_name]['metrics']

    return best_model, best_name, best_metrics, results


def plot_feature_importance(model, feature_cols, output_dir):
    """Generate and save feature importance graph"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Extract feature importances
    if hasattr(model, 'named_estimators_'):
        # Stacking model - average across base models
        importances = []
        for _, estimator in model.named_estimators_.items():
            if hasattr(estimator, 'feature_importances_'):
                importances.append(estimator.feature_importances_)
        if importances:
            avg_importance = np.mean(importances, axis=0)
        else:
            print("  No feature importances available")
            return None
    elif hasattr(model, 'feature_importances_'):
        avg_importance = model.feature_importances_
    else:
        print("  No feature importances available")
        return None

    # Create DataFrame
    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': avg_importance
    }).sort_values('importance', ascending=False)

    # Save to CSV
    imp_df.to_csv(output_dir / 'feature_importance.csv', index=False)

    # Define feature categories for coloring
    def get_category_color(feature):
        temporal = ['days_since_2020', 'year_sold', 'month_sold', 'quarter',
                    'day_of_year', 'year_sold_numeric', 'is_spring_summer']
        size = ['Size', 'size_squared', 'log_size', 'size_per_bedroom', 'size_per_room']
        room = ['Bedrooms', 'Bathrooms', 'total_rooms', 'bath_ratio', 'bed_bath_product']
        age = ['property_age', 'decade_built', 'is_new_construction', 'is_recent']
        interactions = ['size_time_interaction', 'age_size_interaction']
        encoded = ['Location_encoded', 'Condition_encoded', 'Type_encoded']

        if feature in temporal:
            return '#4CAF50'  # Green
        if feature in size:
            return '#2196F3'  # Blue
        if feature in room:
            return '#FF9800'  # Orange
        if feature in age:
            return '#9C27B0'  # Purple
        if feature in interactions:
            return '#F44336'  # Red
        if feature in encoded:
            return '#00BCD4'  # Cyan
        return '#666666'

    # Get top 15 for plotting
    plot_df = imp_df.head(15).sort_values('importance', ascending=True)
    colors = [get_category_color(f) for f in plot_df['feature']]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(plot_df['feature'], plot_df['importance'],
                   color=colors, edgecolor='white', height=0.7)

    for bar, val in zip(bars, plot_df['importance']):
        ax.text(val + 10, bar.get_y() + bar.get_height() / 2, f'{val:.0f}',
                va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Importance Score (averaged across base models)', fontsize=12)
    ax.set_title('Top 15 Feature Importances - Stacking Ensemble',
                 fontsize=16, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_elements = [
        Patch(facecolor='#4CAF50', label='Temporal'),
        Patch(facecolor='#2196F3', label='Size'),
        Patch(facecolor='#FF9800', label='Room'),
        Patch(facecolor='#9C27B0', label='Age'),
        Patch(facecolor='#F44336', label='Interactions'),
        Patch(facecolor='#00BCD4', label='Encoded')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_chart.png', dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"  Chart saved: {output_dir / 'feature_importance_chart.png'}")
    return imp_df


def main(promote_to_prod=True):
    """Main training pipeline"""
    print("=" * 60)
    print("HOUSE PRICE PREDICTION - MODEL TRAINING")
    print("=" * 60)
    print(f"CPUs: {multiprocessing.cpu_count()} (using {N_JOBS})")

    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "docs" / "Case Study 1 Data (1).xlsx"
    output_dir = base_dir / "models"
    output_dir.mkdir(exist_ok=True)

    # Initialize registry
    registry = ModelRegistry(str(output_dir / "registry"))

    # Load data
    print("\nLoading data...")
    df = pd.read_excel(data_path)
    print(f"  {len(df):,} rows")

    # Preprocess
    print("\nPreprocessing...")
    prep = ImprovedPreprocessor()
    X, y = prep.fit_transform(df)
    print(f"  {X.shape[1]} features")
    print(f"  Features: {prep.feature_cols}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nSplit: train={len(y_train):,} test={len(y_test):,}")

    # Train and evaluate
    best_model, best_name, best_metrics, all_results = train_and_evaluate(
        X_train, y_train, X_test, y_test
    )

    # Print results
    print("\n" + "=" * 60)
    print(f"BEST MODEL: {best_name}")
    print("=" * 60)
    print(f"MAE:  ${best_metrics['mae']:,.0f}")
    print(f"RMSE: ${best_metrics['rmse']:,.0f}")
    print(f"R²:   {best_metrics['r2']:.4f}")
    print(f"MAPE: {best_metrics['mape']:.2f}%")

    # Register model
    print("\n" + "=" * 60)
    print("MODEL VERSIONING")
    print("=" * 60)

    prep_data = {
        'target_encoders': prep.target_encoders,
        'scaler': prep.scaler,
        'feature_cols': prep.feature_cols,
        'cat_cols': prep.cat_cols
    }

    # Determine params based on model type
    if best_name == 'stacking':
        params = {
            'model_type': 'StackingRegressor',
            'base_models': ['LGBMRegressor', 'XGBRegressor',
                          'GradientBoostingRegressor', 'RandomForestRegressor'],
            'meta_model': 'Ridge',
            'cv_folds': 5
        }
    else:
        params = {'model_type': type(best_model).__name__}

    version = registry.register(
        model=best_model,
        model_name="house_price_predictor",
        metrics=best_metrics,
        params=params,
        tags=[best_name, 'ensemble' if best_name == 'stacking' else 'single',
              f'n_features={len(prep.feature_cols)}'],
        preprocessor=prep_data
    )

    if promote_to_prod:
        registry.promote("house_price_predictor", version, stage="production")

    # Feature importance
    print("\nGenerating feature importance...")
    imp_df = plot_feature_importance(best_model, prep.feature_cols, base_dir / 'docs')
    if imp_df is not None:
        print(f"\n  Top 5 features:")
        for _, row in imp_df.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance']:.0f}")

    # Save model comparison
    comparison = []
    for name, res in all_results.items():
        comparison.append({
            'model': name,
            'rmse': res['metrics']['rmse'],
            'mae': res['metrics']['mae'],
            'r2': res['metrics']['r2'],
            'time_s': res['time']
        })
    pd.DataFrame(comparison).to_csv(output_dir / 'model_comparison.csv', index=False)

    # Show registry summary
    registry.summary()

    print("\nDone!")
    return registry, version, best_model


if __name__ == "__main__":
    main()
