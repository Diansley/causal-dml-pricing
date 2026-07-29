# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from doubleml import DoubleMLData, DoubleMLPLR

def run_causal_comparison():
    # 1. Carregar Dados
    df = pd.read_csv('data/pricing_dataset.csv')
    
    y = df['log_quantity'].values
    d = df['log_price'].values
    X = df.drop(columns=['log_quantity', 'log_price']).values
    feature_names = [col for col in df.columns if col not in ['log_quantity', 'log_price']]
    
    true_theta = -1.5
    print("=" * 60)
    print(f"🎯 ELASTICIDADE PREÇO VERDADEIRA (GROUND TRUTH): {true_theta}")
    print("=" * 60)

    # -------------------------------------------------------------
    # A. OLS Ingênuo (Regressão Simples sem controlar confundidores)
    # -------------------------------------------------------------
    ols_naive = LinearRegression().fit(d.reshape(-1, 1), y)
    theta_naive = ols_naive.coef_[0]
    print(f"\n1. OLS Ingênuo (Sem X): {theta_naive:.4f}")
    print(f"   Erro relativo: {abs((theta_naive - true_theta)/true_theta)*100:.1f}%")

    # -------------------------------------------------------------
    # B. OLS Controlado (Incluindo X linearmente)
    # -------------------------------------------------------------
    X_with_d = np.column_stack((d, X))
    ols_controlled = LinearRegression().fit(X_with_d, y)
    theta_ols_controlled = ols_controlled.coef_[0]
    print(f"\n2. OLS Controlado (Com X Linear): {theta_ols_controlled:.4f}")
    print(f"   Erro relativo: {abs((theta_ols_controlled - true_theta)/true_theta)*100:.1f}%")

    # -------------------------------------------------------------
    # C. DML Passo a Passo (Manual: XGBoost + Cross-Fitting)
    # -------------------------------------------------------------
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    d_res = np.zeros_like(d)
    y_res = np.zeros_like(y)

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        d_tr, d_te = d[train_idx], d[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Modelo m(X): Prever Preço
        model_m = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
        model_m.fit(X_tr, d_tr)
        d_res[test_idx] = d_te - model_m.predict(X_te)

        # Modelo g(X): Prever Demanda
        model_g = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
        model_g.fit(X_tr, y_tr)
        y_res[test_idx] = y_te - model_g.predict(X_te)

    # Estimar theta nos resíduos (Ortogonalização de Neyman)
    dml_manual = LinearRegression(fit_intercept=False).fit(d_res.reshape(-1, 1), y_res)
    theta_dml_manual = dml_manual.coef_[0]
    print(f"\n3. DML Passo a Passo (Manual + XGBoost): {theta_dml_manual:.4f}")
    print(f"   Erro relativo: {abs((theta_dml_manual - true_theta)/true_theta)*100:.1f}%")

    # -------------------------------------------------------------
    # D. DML via Biblioteca Oficial DoubleML
    # -------------------------------------------------------------
    dml_data = DoubleMLData(df, y_col='log_quantity', d_cols='log_price', x_cols=feature_names)
    ml_m = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    ml_g = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    
    dml_plr = DoubleMLPLR(dml_data, ml_l=ml_g, ml_m=ml_m, n_folds=5)
    dml_plr.fit()
    theta_doubleml = dml_plr.coef[0]
    
    print(f"\n4. DoubleML Package (PLR + XGBoost): {theta_doubleml:.4f}")
    print(f"   Erro relativo: {abs((theta_doubleml - true_theta)/true_theta)*100:.1f}%")
    print("=" * 60)

if __name__ == '__main__':
    run_causal_comparison()
