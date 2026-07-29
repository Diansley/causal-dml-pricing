import os

import numpy as np
import pandas as pd


def generate_heterogeneous_pricing_data(n_samples=5000, seed=42):
    np.random.seed(seed)
    
    # 1. Matriz de Confundidores High-Dimensional (10 variaveis)
    X = np.random.normal(0, 1, size=(n_samples, 10))
    
    # 2. Efeito Causal Heterogeneo Verdadeiro theta(X)
    # Variando entre ~ -2.3 e ~ -0.7 dependendo de X1 e X2
    x1, x2 = X[:, 0], X[:, 1]
    sigmoid_x1 = 1 / (1 + np.exp(-x1))
    true_theta_x = -1.5 + 0.8 * sigmoid_x1 - 0.5 * np.tanh(x2)
    
    # 3. Equacao de Preco (Tratamento) com Confundimento Nao Linear
    m_x = np.sin(X[:, 0]) + 0.5 * X[:, 1]**2 - 0.8 * np.exp(X[:, 2] / 2)
    v = np.random.normal(0, 0.5, size=n_samples)
    log_price = m_x + v
    
    # 4. Equacao de Demanda (Resultado)
    l_x = np.cos(X[:, 0]) + 1.2 * np.abs(X[:, 1]) + 0.5 * X[:, 3]**3
    u = np.random.normal(0, 0.5, size=n_samples)
    
    # Y = l(X) + theta(X)*D + u
    log_quantity = l_x + true_theta_x * log_price + u
    
    # 5. Criar DataFrame e salvar
    os.makedirs('data', exist_ok=True)
    feature_names = [f'X_{i+1}' for i in range(10)]
    df = pd.DataFrame(X, columns=feature_names)
    df['log_price'] = log_price
    df['log_quantity'] = log_quantity
    df['true_theta_x'] = true_theta_x
    
    df.to_csv('data/pricing_dataset_heterogeneous.csv', index=False)
    
    print("=" * 70)
    print("DATASET HETEROGENEO GERADO COM SUCESSO")
    print("=" * 70)
    print(f"Elasticidade Real Media (ATE) : {np.mean(true_theta_x):.4f}")
    print(f"Elasticidade Real Minima     : {np.min(true_theta_x):.4f}")
    print(f"Elasticidade Real Maxima     : {np.max(true_theta_x):.4f}")
    print(f"Desvio Padrao Real de theta  : {np.std(true_theta_x):.4f}")
    print("=" * 70)

if __name__ == '__main__':
    generate_heterogeneous_pricing_data()
