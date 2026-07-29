# Double Machine Learning (DML) for Causal Pricing

Este repositorio contem um pipeline end-to-end de precificacao causal e otimizacao prescritiva, desenvolvido com Double Machine Learning (DML) e Causal Forest.

O objetivo principal e isolar o efeito puro do preco sobre a demanda (curva de elasticidade real \theta(X)), superando o vies de confundimento comum em abordagens tradicionais de regressao, e gerar recomendacoes otimizadas de precos que maximizem o lucro sob restricoes de negocio.

---

## Principais Destaques e Resultados

- Remocao de Vies de Confundimento: Modelos de estorvo (Nuisance Models) utilizando SplineTransformer + RidgeCV para isolar variacoes contextuais nao-lineares.
- Inferenca Causal Granular (CATE): Utilizacao do CausalForestDML (econml) para estimar a elasticidade-preco individual por segmento ou cliente com divisao honesta (honest=True).
- Quantificacao Rigorosa de Incerteza: Cobertura empirica de ~83%+ para o Intervalo de Confianca a 95%.
- Otimizacao Prescritiva Sob Restricoes: Motor de otimizacao isoelastica via scipy.optimize, respeitando limitadores de variacao de preco (+/- 15%) e piso de margem bruta (10%).
- Lift de Lucro Obtido: +17.35% a +28.49% em relacao ao cenario baseline.

---

## Arquitetura do Repositorio

causal-dml-pricing/
??? data/
?   ??? pricing_dataset_heterogeneous.csv   # Dataset de treino/validacao
?   ??? pricing_recommendations.csv         # Recomendacoes finais exportadas
??? reports/
?   ??? figures/
?       ??? causal_forest_uncertainty.png   # Graficos de incerteza da Causal Forest
??? src/
?   ??? __init__.py                         # Pacote Python formal
?   ??? estimate_cate_causal_forest.py      # Diagnostico e validacao de cobertura do IC
?   ??? prescriptive_pricing.py             # Motor de otimizacao isoelastica
?   ??? pipeline.py                         # Classe unificada DoubleMLPricingPipeline (POO)
??? tests/
?   ??? test_pipeline.py                    # Suite de testes unitarios com Pytest
??? main.py                                 # Script CLI para execucao end-to-end
??? pytest.ini                              # Configuracao de contexto para testes
??? requirements.txt                        # Dependencias do projeto

---

## Instalacao e Execucao

### 1. Criar o ambiente virtual e ativar
python -m venv .venv
.\.venv\Scripts\Activate.ps1

### 2. Instalar dependencias
pip install -r requirements.txt

### 3. Executar os testes unitarios
python -m pytest -v

### 4. Executar o pipeline completo
python main.py

---

## Estrutura Metodologica (DML)

1. Ortogonalizacao: Dois modelos de Machine Learning estimam separadamente E[Y|X] (demanda pelo contexto) e E[T|X] (preco pelo contexto).
2. Residuos: Calculam-se Y_res = Y - Y_hat e T_res = T - T_hat.
3. Regressao Causal: Ao relacionar Y_res com T_res, elimina-se o vies do contexto X, obtendo a verdadeira elasticidade-preco causal \theta(X).
4. Otimizacao: A elasticidade alimenta a curva de demanda Q(P) = Q0 * (P/P0)^\theta(X), calculando o preco que maximiza a margem bruta sob restricoes.
