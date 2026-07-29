import numpy as np
import pandas as pd
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from xgboost import XGBRegressor


def run_causal_comparison():
    df = pd.read_csv('data/pricing_dataset.csv')
    
    y = df['log_quantity'].values
    d = df['log_price'].values
    X = df.drop(columns=['log_quantity', 'log_price']).values
    feature_names = [col for col in df.columns if col not in ['log_quantity', 'log_price']]
    
    true_theta = -1.5
    print("=" * 75)
    print(f"🎯 ELASTICIDADE PREÇO VERDADEIRA (GROUND TRUTH): {true_theta}")
    print("=" * 75)

    dml_data = DoubleMLData(df, y_col='log_quantity', d_cols='log_price', x_cols=feature_names)

    # 1. OLS Controlado
    X_with_d = np.column_stack((d, X))
    ols_controlled = LinearRegression().fit(X_with_d, y)
    theta_ols = ols_controlled.coef_[0]
    print(f"1. OLS Controlado (Linear Simples):       {theta_ols:.4f} | Erro: {abs((theta_ols - true_theta)/true_theta)*100:.1f}%")

    # 2. DoubleML com XGBoost
    xgb_learner = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    dml_xgb = DoubleMLPLR(dml_data, ml_l=xgb_learner, ml_m=xgb_learner, n_folds=5)
    dml_xgb.fit()
    theta_xgb = dml_xgb.coef[0]
    print(f"2. DoubleML (XGBoost):                     {theta_xgb:.4f} | Erro: {abs((theta_xgb - true_theta)/true_theta)*100:.1f}%")

    # 3. DoubleML com Rede Neural (MLP)
    mlp_learner = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True)
    )
    dml_mlp = DoubleMLPLR(dml_data, ml_l=mlp_learner, ml_m=mlp_learner, n_folds=5)
    dml_mlp.fit()
    theta_mlp = dml_mlp.coef[0]
    print(f"3. DoubleML (Rede Neural MLP):             {theta_mlp:.4f} | Erro: {abs((theta_mlp - true_theta)/true_theta)*100:.1f}%")

    # 4. DoubleML com Splines Cúbicas + Ridge (Aproximação Suave Eficiente)
    spline_learner = make_pipeline(
        StandardScaler(),
        SplineTransformer(n_knots=5, degree=3, include_bias=False),
        RidgeCV(alphas=np.logspace(-3, 3, 10))
    )
    dml_spline = DoubleMLPLR(dml_data, ml_l=spline_learner, ml_m=spline_learner, n_folds=5)
    dml_spline.fit()
    theta_spline = dml_spline.coef[0]
    print(f"4. DoubleML (Splines Cúbicas + Ridge):     {theta_spline:.4f} | Erro: {abs((theta_spline - true_theta)/true_theta)*100:.1f}%")
    print("=" * 75)

if __name__ == '__main__':
    run_causal_comparison()
