# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def generate_pricing_data(n_samples=5000, n_features=20, true_elasticity=-1.5, random_state=42):
    """
    Gera dados sinteticos de precificacao com endogeneidade de alta dimensao.
    
    Modelo de Regressao Parcialmente Linear (PLR):
        Y = theta_0 * D + g(X) + U   (Log-Quantidade)
        D = m(X) + V                  (Log-Preco)
    """
    np.random.seed(random_state)
    
    # 1. Confundidores (X): Clima, Sazonalidade, Preco dos Concorrentes, Gastos com Marketing, etc.
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    feature_names = [f'confounder_{i+1}' for i in range(n_features)]
    
    # 2. Relacao nao-linear m(X) que afeta o Preco D (Log-Price)
    # Regras de precificacao da empresa baseadas nas condicoes de mercado (X)
    m_X = (
        1.2 * X[:, 0] 
        - 0.8 * X[:, 1] 
        + 0.5 * (X[:, 2] ** 2) 
        + np.sin(X[:, 3]) 
        + 0.7 * X[:, 4] * X[:, 5]
    )
    
    # Preco observado D = m(X) + ruido exogeno V
    v = np.random.normal(0, 0.5, size=n_samples)
    log_price = m_X + v
    
    # 3. Relacao nao-linear g(X) que afeta a Demanda Y (Log-Quantity)
    # Choques de demanda causados pelas mesmas variaveis X
    g_X = (
        1.5 * X[:, 0] 
        - 1.0 * X[:, 1] 
        + 0.8 * np.exp(X[:, 2] / 2) 
        + 1.2 * np.cos(X[:, 3]) 
        + 0.5 * (X[:, 4] + X[:, 5])
    )
    
    # Vendas observadas Y = theta_0 * D + g(X) + ruido exogeno U
    u = np.random.normal(0, 0.5, size=n_samples)
    log_quantity = true_elasticity * log_price + g_X + u
    
    # Montar DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['log_price'] = log_price
    df['log_quantity'] = log_quantity
    
    return df, true_elasticity

if __name__ == '__main__':
    df, true_beta = generate_pricing_data()
    df.to_csv('data/pricing_dataset.csv', index=False)
    print("Dataset gerado com sucesso em data/pricing_dataset.csv!")
    print(f"Formato dos dados: {df.shape}")
    print(f"Elasticidade Preco Real (theta_0): {true_beta}")
