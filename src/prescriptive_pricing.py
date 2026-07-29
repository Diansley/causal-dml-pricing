import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def run_prescriptive_pricing():
    os.makedirs('reports', exist_ok=True)
    
    # 1. Carregar dataset e estimativas de CATE obtidas anteriormente
    df = pd.read_csv('data/pricing_dataset_heterogeneous.csv')
    
    # Converter log para escala original
    price_current = np.exp(df['log_price'].values)
    quantity_current = np.exp(df['log_quantity'].values)
    theta_est = df['true_theta_x'].values  # Substitua por CATE estimado do modelo
    
    n_samples = len(df)
    
    # Simular Custo Unitario (ex: 60% do preco medio atual com variacao)
    np.random.seed(42)
    unit_cost = price_current * np.random.uniform(0.45, 0.65, size=n_samples)
    
    # Parametros de Politica de Precificacao
    MAX_PRICE_CHANGE = 0.15  # Maximo de +/- 15% de alteracao de preco
    MIN_MARGIN = 0.10        # Margem bruta minima de 10%
    
    optimal_prices = np.zeros(n_samples)
    expected_quantities = np.zeros(n_samples)
    
    print("=" * 70)
    print("EXECUTANDO OTIMIZADOR PRESCRITIVO DE PRECOS...")
    print("=" * 70)
    
    for i in range(n_samples):
        p0 = price_current[i]
        q0 = quantity_current[i]
        theta = theta_est[i]
        cost = unit_cost[i]
        
        # Limites de Preco (Constraints)
        p_min = max(p0 * (1 - MAX_PRICE_CHANGE), cost * (1 + MIN_MARGIN))
        p_max = p0 * (1 + MAX_PRICE_CHANGE)
        
        # Função de Lucro Negativa (para minimização)
        def negative_profit(p, q0, p0, theta, cost):
            # Curva de demanda isoelastica: Q(P) = Q0 * (P / P0)^theta
            q_pred = q0 * ((p / p0) ** theta)
            profit = (p - cost) * q_pred
            return -profit
        
        # Otimizacao Pontual por Observacao
        res = minimize_scalar(
            negative_profit, 
            bounds=(p_min, p_max), 
            args=(q0, p0, theta, cost), 
            method='bounded'
        )
        
        optimal_prices[i] = res.x
        expected_quantities[i] = q0 * ((res.x / p0) ** theta)
        
    # Baseline Atual vs Prescritivo
    profit_current = (price_current - unit_cost) * quantity_current
    profit_optimal = (optimal_prices - unit_cost) * expected_quantities
    
    revenue_current = price_current * quantity_current
    revenue_optimal = optimal_prices * expected_quantities
    
    total_profit_baseline = np.sum(profit_current)
    total_profit_prescriptive = np.sum(profit_optimal)
    profit_lift_pct = ((total_profit_prescriptive - total_profit_baseline) / total_profit_baseline) * 100
    
    total_rev_baseline = np.sum(revenue_current)
    total_rev_prescriptive = np.sum(revenue_optimal)
    rev_lift_pct = ((total_rev_prescriptive - total_rev_baseline) / total_rev_baseline) * 100
    
    # Exibir Relatório
    print("RELATORIO DE IMPACTO FINANCEIRO DA PRECIFICACAO PRESCRITIVA")
    print("-" * 70)
    print(f"Lucro Total Baseline (Preco Atual)    : R$ {total_profit_baseline:,.2f}")
    print(f"Lucro Total Otimizado (Prescritivo)  : R$ {total_profit_prescriptive:,.2f}")
    print(f"LIFT DE LUCRO ESPERADO               : +{profit_lift_pct:.2f}%")
    print("-" * 70)
    print(f"Receita Total Baseline               : R$ {total_rev_baseline:,.2f}")
    print(f"Receita Total Otimizada              : R$ {total_rev_prescriptive:,.2f}")
    print(f"LIFT DE RECEITA ESPERADO             : {rev_lift_pct:+.2f}%")
    print("-" * 70)
    print(f"Variacao Media de Preco Recomendo    : {np.mean((optimal_prices - price_current) / price_current) * 100:+.2f}%")
    print("=" * 70)

if __name__ == '__main__':
    run_prescriptive_pricing()
