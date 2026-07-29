import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


def estimate_causal_forest_uncertainty():
    os.makedirs('reports/figures', exist_ok=True)
    
    # 1. Carregar dataset com heterogeneidade real
    df = pd.read_csv('data/pricing_dataset_heterogeneous.csv')
    
    y = df['log_quantity'].values
    d = df['log_price'].values
    true_theta = df['true_theta_x'].values
    
    x_cols = [c for c in df.columns if c not in ['log_quantity', 'log_price', 'true_theta_x']]
    X = df[x_cols].values
    
    print("=" * 70)
    print("RE-TREINANDO CAUSAL FOREST COM HONEST SPLITTING E SPLINES...")
    print("=" * 70)
    
    # 2. Modelos de estorvo (nuance) de alta precisao para purificar residuos
    model_y = make_pipeline(
        StandardScaler(),
        SplineTransformer(n_knots=5, degree=3, include_bias=False),
        RidgeCV(alphas=np.logspace(-3, 3, 10))
    )
    
    model_t = make_pipeline(
        StandardScaler(),
        SplineTransformer(n_knots=5, degree=3, include_bias=False),
        RidgeCV(alphas=np.logspace(-3, 3, 10))
    )
    
    # 3. Configurar CausalForestDML com Honest Trees
    causal_forest = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        n_estimators=400,
        min_samples_leaf=10,
        max_depth=8,
        honest=True,
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    
    # 4. Ajustar modelo
    causal_forest.fit(Y=y, T=d, X=X)
    
    # 5. Predicao de Efeitos e Intervalos de Confianca (95%)
    print("Calculando estimativas pontuais e Intervalos de Confianca (95%)...")
    cate_pred = causal_forest.effect(X)
    cate_ci_lower, cate_ci_upper = causal_forest.effect_interval(X, alpha=0.05)
    
    # Taxa de Cobertura Estatistica
    coverage = np.mean((true_theta >= cate_ci_lower) & (true_theta <= cate_ci_upper))
    
    print("-" * 70)
    print("RESULTADOS DA INFERENCIA INDIVIDUAL (CAUSAL FOREST)")
    print("-" * 70)
    print(f"ATE Real                       : {np.mean(true_theta):.4f}")
    print(f"ATE Estimado                   : {np.mean(cate_pred):.4f}")
    print(f"Largura Media do CI (95%)      : {np.mean(cate_ci_upper - cate_ci_lower):.4f}")
    print(f"Taxa de Cobertura do CI (95%)  : {coverage * 100:.2f}%")
    print("=" * 70)
    
    # 6. Visualizacao: Amostra com Bandas de Incerteza
    sort_idx = np.argsort(true_theta)
    sample_indices = sort_idx[::50]  
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(sample_indices)), true_theta[sample_indices], 'r--', label='Theta Real', linewidth=2)
    plt.plot(range(len(sample_indices)), cate_pred[sample_indices], 'b-o', label='Causal Forest CATE', markersize=4, alpha=0.8)
    
    plt.fill_between(
        range(len(sample_indices)),
        cate_ci_lower[sample_indices],
        cate_ci_upper[sample_indices],
        color='blue', alpha=0.2, label='IC 95% (Honest Forest)'
    )
    
    plt.title('CATE com Intervalo de Confiança 95% (Causal Forest + Splines)', fontsize=12, pad=12)
    plt.xlabel('Amostra de Observações (Ordenadas por Elasticidade Real)', fontsize=10)
    plt.ylabel('Elasticidade-Preço Estimada θ(X)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    output_path = 'reports/figures/causal_forest_uncertainty.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Grafico atualizado salvo em: {output_path}")
    print("=" * 70)

if __name__ == '__main__':
    estimate_causal_forest_uncertainty()
