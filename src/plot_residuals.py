import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


def plot_residual_orthogonalization():
    os.makedirs('reports/figures', exist_ok=True)
    
    # 1. Carregar dataset
    df = pd.read_csv('data/pricing_dataset.csv')
    
    y = df['log_quantity'].values
    d = df['log_price'].values
    x_cols = [c for c in df.columns if c not in ['log_quantity', 'log_price']]
    X = df[x_cols].values
    
    # 2. Pipeline de regressao flexivel (Splines + Ridge)
    spline_learner = make_pipeline(
        StandardScaler(),
        SplineTransformer(n_knots=5, degree=3, include_bias=False),
        RidgeCV(alphas=np.logspace(-3, 3, 10))
    )
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 3. Estimativa de funcoes de estorvo por cross-fitting
    print("Calculando residuos cross-fitted...")
    l_hat = cross_val_predict(spline_learner, X, y, cv=cv)
    m_hat = cross_val_predict(spline_learner, X, d, cv=cv)
    
    # 4. Ortogonalizacao de Neyman (Residuos)
    y_res = y - l_hat
    d_res = d - m_hat
    
    # Inclinacao dos residuos (OLS nos residuos)
    slope, intercept = np.polyfit(d_res, y_res, 1)
    
    # 5. Construcao do grafico comparativo
    _fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    
    # Subplot 1: Dados Brutos (Vies de endogeneidade)
    axes[0].scatter(d, y, alpha=0.25, color='#2b5c8f', edgecolors='none', s=15)
    axes[0].set_title('Dados Brutos (Com Vies de Confundimento)', fontsize=11, pad=10)
    axes[0].set_xlabel('log(Preco) [D]', fontsize=10)
    axes[0].set_ylabel('log(Quantidade) [Y]', fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Subplot 2: Residuos Ortogonalizados (Efeito Causal)
    axes[1].scatter(d_res, y_res, alpha=0.25, color='#1b8a5a', edgecolors='none', s=15, label='Residuos (DML)')
    
    d_grid = np.linspace(d_res.min(), d_res.max(), 100)
    axes[1].plot(
        d_grid, 
        slope * d_grid + intercept, 
        color='black', 
        linestyle='--', 
        linewidth=2,
        label=f'Inclinacao Estimada: {slope:.4f}\n(Ground Truth: -1.5000)'
    )
    
    axes[1].set_title('Residuos Ortogonalizados (Relacao Causal Limpa)', fontsize=11, pad=10)
    axes[1].set_xlabel('Residuos do Preco (D - m(X))', fontsize=10)
    axes[1].set_ylabel('Residuos da Demanda (Y - l(X))', fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    
    output_path = 'reports/figures/residuals_orthogonalization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("=" * 70)
    print("DIAGNOSTICO VISUAL DE ORTOGONALIZACAO CONCLUIDO")
    print("=" * 70)
    print(f"Inclinacao dos residuos (theta_hat) : {slope:.4f}")
    print("Efeito real (Ground Truth)          : -1.5000")
    print(f"Grafico salvo em                    : {output_path}")
    print("=" * 70)

if __name__ == '__main__':
    plot_residual_orthogonalization()
