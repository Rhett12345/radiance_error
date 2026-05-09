# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


@dataclass
class FitResult:
    coef1: float
    coef2: Optional[float]
    intercept: float
    r: float
    residual_std: float
    model: object
    model_type: str
    n: int


def finite_xy(x, y) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float).reshape(-1, 1)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr).ravel() & np.isfinite(y_arr)
    return x_arr[mask], y_arr[mask]


def safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    r, _ = pearsonr(y_true, y_pred)
    return float(r)


def residual_std(x_arr: np.ndarray, y_arr: np.ndarray, model: object) -> float:
    y_pred = model.predict(x_arr)
    if len(y_arr) <= 1:
        return float("nan")
    return float(np.std(y_arr - y_pred, ddof=1))


def fit_model(x, y, r_threshold: float = 0.98, polynomial_degree: int = 2) -> FitResult | None:
    x_arr, y_arr = finite_xy(x, y)
    n = len(y_arr)
    if n < 5:
        return None

    linear = LinearRegression()
    linear.fit(x_arr, y_arr)
    pred_linear = linear.predict(x_arr)
    r_linear = safe_pearsonr(y_arr, pred_linear)

    best_model = linear
    best_type = "linear"
    best_r = r_linear

    if not np.isfinite(r_linear) or r_linear < r_threshold:
        poly = make_pipeline(PolynomialFeatures(degree=polynomial_degree), LinearRegression())
        poly.fit(x_arr, y_arr)
        pred_poly = poly.predict(x_arr)
        r_poly = safe_pearsonr(y_arr, pred_poly)
        if np.isfinite(r_poly) and (not np.isfinite(best_r) or r_poly > best_r):
            best_model = poly
            best_type = f"poly{polynomial_degree}"
            best_r = r_poly

    sigma = residual_std(x_arr, y_arr, best_model)
    if best_type == "linear":
        return FitResult(
            coef1=float(best_model.coef_[0]),
            coef2=None,
            intercept=float(best_model.intercept_),
            r=float(best_r),
            residual_std=sigma,
            model=best_model,
            model_type=best_type,
            n=n,
        )

    lr = best_model.named_steps["linearregression"]
    coefs = lr.coef_
    # For degree=2 this means: y = coef2*x^2 + coef1*x + intercept.
    # For higher degree, only the first two powers are exported for backward compatibility.
    return FitResult(
        coef1=float(coefs[1]) if len(coefs) > 1 else 0.0,
        coef2=float(coefs[2]) if len(coefs) > 2 else None,
        intercept=float(lr.intercept_),
        r=float(best_r),
        residual_std=sigma,
        model=best_model,
        model_type=best_type,
        n=n,
    )
