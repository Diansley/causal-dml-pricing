import numpy as np
from econml.dml import CausalForestDML
from scipy.optimize import minimize_scalar
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


class DoubleMLPricingPipeline:
    """
    Pipeline de Inferencia Causal e Precificacao Prescritiva usando Double/Debiased Machine Learning (DML).
    """
    def __init__(self, n_estimators: int = 400, max_depth: int = 8, cv: int = 5, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.cv = cv
        self.random_state = random_state
        self.model = None

    def _build_nuisance_model(self):
        """Pipeline de Splines + RidgeCV para capturar nao-linearidades de confundidores."""
        return make_pipeline(
            StandardScaler(),
            SplineTransformer(n_knots=5, degree=3, include_bias=False),
            RidgeCV(alphas=np.logspace(-3, 3, 10))
        )

    def fit(self, X: np.ndarray, d: np.ndarray, y: np.ndarray):
        """Treina a Causal Forest DML com Honest Splitting."""
        model_y = self._build_nuisance_model()
        model_t = self._build_nuisance_model()

        self.model = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            n_estimators=self.n_estimators,
            min_samples_leaf=10,
            max_depth=self.max_depth,
            honest=True,
            cv=self.cv,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(Y=y, T=d, X=X)
        return self

    def predict_cate(self, X: np.ndarray, alpha: float = 0.05):
        """Retorna as estimativas pontuais do CATE e os limites do Intervalo de Confianca."""
        if self.model is None:
            raise ValueError("O modelo precisa ser treinado com .fit() antes de prever.")
        
        cate_pred = self.model.effect(X)
        ci_lower, ci_upper = self.model.effect_interval(X, alpha=alpha)
        return cate_pred, ci_lower, ci_upper

    def optimize_prices(
        self, 
        price_current: np.ndarray, 
        quantity_current: np.ndarray, 
        unit_cost: np.ndarray, 
        theta_est: np.ndarray,
        max_change: float = 0.15,
        min_margin: float = 0.10
    ) -> np.ndarray:
        """Resolve o problema de otimizacao isoelastica de lucros sob restricao de borda."""
        n_samples = len(price_current)
        optimal_prices = np.zeros(n_samples)

        for i in range(n_samples):
            p0 = price_current[i]
            q0 = quantity_current[i]
            theta = theta_est[i]
            cost = unit_cost[i]

            p_min = max(p0 * (1 - max_change), cost * (1 + min_margin))
            p_max = p0 * (1 + max_change)

            def neg_profit(p, q0, p0, theta, cost):
                q_pred = q0 * ((p / p0) ** theta)
                return -((p - cost) * q_pred)
            res = minimize_scalar(
                    neg_profit, 
                    bounds=(p_min, p_max), 
                    args=(q0, p0, theta, cost), 
                    method='bounded'
                )
            optimal_prices[i] = res.x

        return optimal_prices
