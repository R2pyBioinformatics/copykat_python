"""Dynamic linear model (Kalman) smoothing (R source: copykat.R, dlm.sm inline)."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

__all__ = ["dlm_smooth"]


def _kalman_smooth_single(y: np.ndarray, dV: float, dW: float) -> np.ndarray:
    """Apply a 1st-order polynomial DLM Kalman smoother to a 1-D signal.

    Equivalent to R's dlm::dlmModPoly(order=1, dV=dV, dW=dW) followed by
    dlm::dlmSmooth.

    Parameters
    ----------
    y : np.ndarray
        Observed signal (1-D).
    dV : float
        Observation variance.
    dW : float
        System (state evolution) variance.

    Returns
    -------
    np.ndarray
        Smoothed signal, same length as *y*, mean-centered.
    """
    n = len(y)

    # Forward pass (Kalman filter)
    # State: theta_t (scalar), model: y_t = theta_t + v_t, theta_t = theta_{t-1} + w_t
    m = np.zeros(n + 1)  # Filtered state mean: m[0] is prior, m[t] is after obs t
    C = np.zeros(n + 1)  # Filtered state variance
    # Prior: diffuse initialization matching dlm defaults
    m[0] = 0.0
    C[0] = 1e7  # dlm default: large prior variance

    f_vals = np.zeros(n)  # Forecast mean
    Q_vals = np.zeros(n)  # Forecast variance

    for t in range(n):
        # Prediction step
        a_t = m[t]        # Prior state mean
        R_t = C[t] + dW   # Prior state variance
        # Update step
        f_vals[t] = a_t   # Forecast mean
        Q_vals[t] = R_t + dV  # Forecast variance
        e_t = y[t] - f_vals[t]  # Forecast error
        K_t = R_t / Q_vals[t]   # Kalman gain
        m[t + 1] = a_t + K_t * e_t
        C[t + 1] = R_t - K_t * R_t  # = R_t * (1 - K_t)

    # Backward pass (Rauch-Tung-Striebel smoother)
    s = np.zeros(n + 1)  # Smoothed state mean
    S = np.zeros(n + 1)  # Smoothed state variance
    s[n] = m[n]
    S[n] = C[n]

    for t in range(n - 1, -1, -1):
        R_t = C[t] + dW
        B_t = C[t] / R_t if R_t > 0 else 0.0
        s[t] = m[t] + B_t * (s[t + 1] - (m[t]))  # Note: a_{t+1} = m[t]
        S[t] = C[t] + B_t * B_t * (S[t + 1] - R_t)

    # dlm::dlmSmooth returns s[0:n], where s[0] is the smoothed prior
    # R code takes s[2:length(s)] which is s[1:n] (0-indexed)
    smoothed = s[1:]  # length n, matching R: x <- x[2:length(x)]
    smoothed = smoothed - np.mean(smoothed)
    return smoothed


def dlm_smooth(
    mat: np.ndarray,
    dV: float = 0.16,
    dW: float = 0.001,
    n_cores: int = 1,
) -> np.ndarray:
    """Apply DLM smoothing to each column of a matrix.

    Parameters
    ----------
    mat : np.ndarray
        Matrix of shape (n_genes, n_cells). Each column is smoothed
        independently.
    dV : float
        Observation variance for the DLM (default 0.16).
    dW : float
        System variance for the DLM (default 0.001).
    n_cores : int
        Number of parallel cores to use.

    Returns
    -------
    np.ndarray
        Smoothed matrix, same shape as input, each column mean-centered.
    """
    n_cells = mat.shape[1]

    if n_cores > 1:
        results = Parallel(n_jobs=n_cores)(
            delayed(_kalman_smooth_single)(mat[:, c], dV, dW)
            for c in range(n_cells)
        )
    else:
        results = [_kalman_smooth_single(mat[:, c], dV, dW) for c in range(n_cells)]

    return np.column_stack(results)
