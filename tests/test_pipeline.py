import numpy as np
import pytest

from src.pipeline import DoubleMLPricingPipeline


@pytest.fixture
def dummy_data():
    """Gera um mini dataset para testes rapidos de integracao."""
    np.random.seed(42)
    n = 80
    X = np.random.normal(0, 1, size=(n, 2))
    log_price = np.random.normal(2, 0.2, size=n)
    log_quantity = 5.0 - 1.5 * log_price + 0.2 * X[:, 0]
    
    price = np.exp(log_price)
    quantity = np.exp(log_quantity)
    cost = price * 0.5
    
    return X, log_price, log_quantity, price, quantity, cost

def test_pipeline_fit_and_prediction(dummy_data):
    X, d, y, _price, _quantity, _cost = dummy_data
    
    # n_estimators=40 (multiplo de subforest_size=4)
    pipeline = DoubleMLPricingPipeline(n_estimators=40, max_depth=4, cv=2)
    pipeline.fit(X, d, y)
    
    cate, ci_lower, ci_upper = pipeline.predict_cate(X)
    
    assert len(cate) == len(X)
    assert len(ci_lower) == len(X)
    assert len(ci_upper) == len(X)
    assert np.all(ci_lower <= ci_upper)

def test_price_optimization_constraints(dummy_data):
    _X, _d, _y, price, quantity, cost = dummy_data
    pipeline = DoubleMLPricingPipeline()
    
    # Elasticidade constante para teste
    theta_dummy = np.full(len(price), -1.5)
    max_change = 0.15
    
    opt_prices = pipeline.optimize_prices(
        price_current=price,
        quantity_current=quantity,
        unit_cost=cost,
        theta_est=theta_dummy,
        max_change=max_change
    )
    
    # Validar que nenhuma recomendacao violou o limite maximo de +/- 15%
    assert np.all(opt_prices <= price * (1 + max_change) + 1e-5)
    assert np.all(opt_prices >= price * (1 - max_change) - 1e-5)

def test_predict_before_fit_raises_error(dummy_data):
    X, _, _, _, _, _ = dummy_data
    pipeline = DoubleMLPricingPipeline()
    
    with pytest.raises(ValueError):
        pipeline.predict_cate(X)
