import os

import numpy as np
import pandas as pd

from src.pipeline import DoubleMLPricingPipeline


def main():
    print("=" * 60)
    print("EXECUTANDO PIPELINE END-TO-END DE PRECIFICAO CAUSAL (DML)")
    print("=" * 60)

    data_path = os.path.join("data", "pricing_dataset_heterogeneous.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset nao encontrado em '{data_path}'. Verifique o caminho.")

    df = pd.read_csv(data_path)
    print(f"Dataset carregado com sucesso ({len(df)} registros).")

    if "price" in df.columns:
        price = df["price"].values
        log_price = np.log(price) if "log_price" not in df.columns else df["log_price"].values
    elif "log_price" in df.columns:
        log_price = df["log_price"].values
        price = np.exp(log_price)
    else:
        raise KeyError("Coluna 'price' ou 'log_price' nao encontrada no CSV.")

    if "quantity" in df.columns:
        quantity = df["quantity"].values
        log_quantity = np.log(quantity) if "log_quantity" not in df.columns else df["log_quantity"].values
    elif "log_quantity" in df.columns:
        log_quantity = df["log_quantity"].values
        # Tratar outliers logaritmicos para evitar estouro numerico em exp()
        log_quantity = np.clip(log_quantity, a_min=-10.0, a_max=8.0)
        quantity = np.exp(log_quantity)
    else:
        raise KeyError("Coluna 'quantity' ou 'log_quantity' nao encontrada no CSV.")

    cost = df["cost"].values if "cost" in df.columns else price * 0.5

    exclude_cols = ["price", "quantity", "cost", "log_price", "log_quantity", "d", "y"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].values

    print("\nTreinando o modelo Double Machine Learning (Causal Forest)...")
    pipeline = DoubleMLPricingPipeline(n_estimators=100, max_depth=5, cv=3)
    pipeline.fit(X, log_price, log_quantity)

    print("Estimando elasticidades-preco individuais (CATE)...")
    theta_est, ci_lower, ci_upper = pipeline.predict_cate(X)
    df["elasticity_est"] = theta_est
    df["elasticity_ci_lower"] = ci_lower
    df["elasticity_ci_upper"] = ci_upper

    print("Executando motor prescritivo de otimizacao de precos...")
    opt_prices = pipeline.optimize_prices(
        price_current=price,
        quantity_current=quantity,
        unit_cost=cost,
        theta_est=theta_est,
        max_change=0.15,
        min_margin=0.10
    )
    df["price_optimized"] = opt_prices

    quantity_opt_est = quantity * (opt_prices / price) ** theta_est
    profit_baseline = np.sum((price - cost) * quantity)
    profit_optimized = np.sum((opt_prices - cost) * quantity_opt_est)
    lift_percent = ((profit_optimized - profit_baseline) / profit_baseline) * 100

    df["quantity_est_opt"] = quantity_opt_est
    df["profit_baseline"] = (price - cost) * quantity
    df["profit_optimized"] = (opt_prices - cost) * quantity_opt_est

    output_path = os.path.join("data", "pricing_recommendations.csv")
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("RESULTADOS DA OTIMIZACAO DE PRECOS")
    print("=" * 60)
    print(f"* Elasticidade Media Estimada: {np.mean(theta_est):.4f}")
    print(f"* Preco Medio Atual:         R$ {np.mean(price):.2f}")
    print(f"* Preco Medio Otimizado:     R$ {np.mean(opt_prices):.2f}")
    print(f"* Lucro Total Baseline:     R$ {profit_baseline:,.2f}")
    print(f"* Lucro Total Otimizado:    R$ {profit_optimized:,.2f}")
    print(f"* Lift de Lucro Esperado:   +{lift_percent:.2f}%")
    print("=" * 60)
    print(f"Recomendacoes completas salvas em: '{output_path}'\n")

if __name__ == "__main__":
    main()
