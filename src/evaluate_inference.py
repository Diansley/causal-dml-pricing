import numpy as np
import pandas as pd
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


def run_statistical_inference():
    # 1. Carregar dataset
    df = pd.read_csv('data/pricing_dataset.csv')
    
    y_col = 'log_quantity'
    d_col = 'log_price'
    x_cols = [c for c in df.columns if c not in [y_col, d_col]]
    
    true_theta = -1.5
    
    dml_data = DoubleMLData(df, y_col=y_col, d_cols=d_col, x_cols=x_cols)
    
    # 2. Configurar o pipeline campeão (Splines Cúbicas + Ridge)
    spline_learner = make_pipeline(
        StandardScaler(),
        SplineTransformer(n_knots=5, degree=3, include_bias=False),
        RidgeCV(alphas=np.logspace(-3, 3, 10))
    )
    
    # 3. Ajustar o DoubleMLPLR
    dml_spline = DoubleMLPLR(
        dml_data,
        ml_l=spline_learner,
        ml_m=spline_learner,
        n_folds=5,
        score='partialling out'
    )
    
    dml_spline.fit()
    
    # 4. Extrair métricas econométricas de inferência
    coef = dml_spline.coef[0]
    se = dml_spline.se[0]
    t_stat = dml_spline.t_stat[0]
    pval = dml_spline.pval[0]
    ci = dml_spline.confint(level=0.95)
    
    ci_lower = ci.iloc[0, 0]
    ci_upper = ci.iloc[0, 1]
    
    contains_truth = ci_lower <= true_theta <= ci_upper
    status_str = "Sim [OK]" if contains_truth else "Nao [FAIL]"
    
    print("=" * 70)
    print("RELATORIO DE INFERENCIA ESTATISTICA (DOUBLE ML - PLR)")
    print("=" * 70)
    print(f"Elasticidade Real (Ground Truth) : {true_theta}\n")
    print(f"Estimativa Ponto (theta_hat)    : {coef:.4f}")
    print(f"Erro Padrao Assintotico (SE)    : {se:.4f}")
    print(f"Estatistica z                   : {t_stat:.4f}")
    print(f"p-valor                         : {pval:.4e}")
    print(f"Intervalo de Confianca 95%      : [{ci_lower:.4f}, {ci_upper:.4f}]")
    print("-" * 70)
    print(f"O valor real ({true_theta}) esta dentro do IC de 95%? {status_str}")
    print("=" * 70)

if __name__ == '__main__':
    run_statistical_inference()
