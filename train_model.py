# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import joblib
import warnings

warnings.filterwarnings("ignore")


NUM_FEATURES = ['Year', 'Mileage_km', 'Engine_cm3', 'Power_hp']
CAT_FEATURES = ['Brand', 'Model', 'BodyType', 'FuelType', 'Transmission']
TARGET = 'Price_eur'

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df[TARGET].between(1500, 85000)]
    df = df[df['Year'].between(2005, 2025)]
    df = df[df['Power_hp'].between(50, 450)]
    df = df[df['Mileage_km'].between(500, 450000)]
    df = df[df['Engine_cm3'].between(600, 5000) | (df['Engine_cm3'] == 0)]  # 0 = электрокары

    for col in CAT_FEATURES:
        vc = df[col].value_counts()
        rare = vc[vc < 5].index
        if len(rare) > 0:
            df.loc[df[col].isin(rare), col] = 'other'

    print(f"После очистки: {df.shape}")
    return df.reset_index(drop=True)


def build_preprocessor():
    num_pipe = Pipeline([
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('ohe', OneHotEncoder(
            handle_unknown='infrequent_if_exist',
            max_categories=60,
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, NUM_FEATURES),
        ('cat', cat_pipe, CAT_FEATURES),
    ], remainder='drop')

    return preprocessor


def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def preprocess_and_train(
    input_path: str,
    processed_path: str = None
):
    print(f"Читаем данные: {input_path}")
    df = pd.read_csv(input_path)

    df = clean_data(df)

    if processed_path:
        df.to_csv(processed_path, index=False)
        print(f"Сохранено очищенное: {processed_path}")

    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET]

    pt_y = PowerTransformer(method='yeo-johnson')
    y_transformed = pt_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_transformed,
        test_size=0.25,
        random_state=42
    )

    preprocessor = build_preprocessor()

    model_pipe = Pipeline([
        ('prep', preprocessor),
        ('reg', SGDRegressor(random_state=42, max_iter=8000, tol=1e-4))
    ])

    param_grid = {
        'reg__alpha': [1e-5, 1e-4, 5e-4, 1e-3, 5e-3],
        'reg__l1_ratio': [0.01, 0.1, 0.3, 0.5],
        'reg__penalty': ['elasticnet', 'l2'],
        'reg__loss': ['squared_error', 'huber'],
        'reg__fit_intercept': [True],
    }

    mlflow.set_experiment("synthetic_cars_price_sgd")

    with mlflow.start_run(run_name="SGD_Synthetic_250"):
        print("Запуск GridSearch...")
        gs = GridSearchCV(
            model_pipe,
            param_grid,
            cv=4,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        gs.fit(X_train, y_train)

        best_model = gs.best_estimator_
        y_pred_tr = best_model.predict(X_val)
        y_pred = pt_y.inverse_transform(y_pred_tr.reshape(-1, 1)).ravel()
        y_true = pt_y.inverse_transform(y_val.reshape(-1, 1)).ravel()

        rmse, mae, r2 = eval_metrics(y_true, y_pred)

        # Логируем
        mlflow.log_params(gs.best_params_)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("cv_neg_rmse", gs.best_score_)

        signature = infer_signature(X_train.head(5), y_pred[:5])
        mlflow.sklearn.log_model(best_model, "model", signature=signature)

        joblib.dump(pt_y, "price_power_transformer.pkl")

        print("\n" + "="*60)
        print(f"RMSE   : {rmse:,.0f} €")
        print(f"MAE    : {mae:,.0f} €")
        print(f"R²     : {r2:.3f}")
        print("="*60 + "\n")

    return {"rmse": rmse, "mae": mae, "r2": r2}


if __name__ == "__main__":
    preprocess_and_train(
        input_path="synthetic_cars_2025.csv",
        processed_path="synthetic_cars_clean.csv"
    )
