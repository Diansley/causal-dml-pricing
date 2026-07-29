import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


def evaluate_heterogeneous_elasticity():
    os.makedirs('reports/figures', exist_ok=True)
    
    # 1. Carregar dataset com heterogeneidade real
    df = pd.read_csv('data/pricing_dataset_heterogeneous.csv')
    
    y = df['log_quantity'].values
    d = df['log_price'].values
    true_theta = df['true_theta_x'].values
    
    x_cols = [c for c in df.columns if c not in ['log_quantity', 'log_price', 'true_theta_x']]
    X = df[x_cols].values
    
    # 2. Estimar funcoes de estorvo (Neyman Orthogonalization)
    spline_learner = make_pipeline(
        StandardScaler(),
        SplineTransformer(n_knots=5, degree=3, include_bias=False),
        RidgeCV(alphas=np.logspace(-3, 3, 10))
    )
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Estimando modelos de estorvo via cross-fitting...")
    l_hat = cross_val_predict(spline_learner, X, y, cv=cv)
    m_hat = cross_val_predict(spline_learner, X, d, cv=cv)
    
    y_res = y - l_hat
    d_res = d - m_hat
    
    # Mascara de estabilidade numerica
    epsilon = 1e-4
    valid_mask = np.abs(d_res) > epsilon
    
    X_v = X[valid_mask]
    y_res_v = y_res[valid_mask]
    d_res_v = d_res[valid_mask]
    
    # 3. Formulacao R-Learner
    pseudo_target = y_res_v / d_res_v
    sample_weights = d_res_v ** 2
    
    # 4. Treinamento do Modelo de CATE
    print("Treinando R-Learner (Random Forest) para capturar theta(X)...")
    cate_model = RandomForestRegressor(
        n_estimators=300, 
        max_depth=6, 
        min_samples_leaf=15, 
        random_state=42
    )
    
    cate_model.fit(X_v, pseudo_target, sample_weight=sample_weights)
    
    # Predicao de elasticidade individual
    estimated_cate = cate_model.predict(X)
    
    # 5. Avaliacao de Performance Causal
    rmse_cate = np.sqrt(mean_squared_error(true_theta, estimated_cate))
    r2_cate = r2_score(true_theta, estimated_cate)
    corr = np.corrcoef(true_theta, estimated_cate)[0, 1]
    
    print("=" * 70)
    print("AVALIACAO DE PERFORMANCE DO CATE (R-LEARNER)")
    print("=" * 70)
    print(f"ATE Real                       : {np.mean(true_theta):.4f}")
    print(f"ATE Estimado                   : {np.mean(estimated_cate):.4f}")
    print("-" * 70)
    print(f"Desvio Padrao Real de theta    : {np.std(true_theta):.4f}")
    print(f"Desvio Padrao Estimado theta   : {np.std(estimated_cate):.4f}")
    print("-" * 70)
    print(f"RMSE de theta(X)               : {rmse_cate:.4f}")
    print(f"R2 na Predicao do CATE         : {r2_cate:.4f}")
    print(f"Correlacao (Real vs Estimado)  : {corr:.4f}")
    print("=" * 70)
    
    # 6. Grafico Scatter: theta(X) Real vs Estimado
    plt.figure(figsize=(8, 6))
    plt.scatter(true_theta, estimated_cate, alpha=0.3, color='#2b5c8f', edgecolors='none', s=20)
    
    # Linha ideal de 45 graus
    grid = np.linspace(true_theta.min(), true_theta.max(), 100)
    plt.plot(grid, grid, color='red', linestyle='--', linewidth=2, label='Predicao Perfeita (45 deg)')
    
    plt.title('CATE: Elasticidade Real vs Estimada pelo R-Learner', fontsize=12, pad=12)
    plt.xlabel('Elasticidade Real theta(X)', fontsize=10)
    plt.ylabel('Elasticidade Estimada hat_theta(X)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    output_path = 'reports/figures/cate_real_vs_estimated.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Grafico comparativo salvo em: {output_path}")
    print("=" * 70)

if __name__ == '__main__':
    evaluate_heterogeneous_elasticity()
