"""Tests for DLM smoothing (R source: copykat.R dlm.sm inline)."""

import numpy as np
import pytest

from copykat.smoothing import dlm_smooth, _kalman_smooth_single


class TestKalmanSmooth:
    def test_output_length(self):
        y = np.random.RandomState(42).randn(100)
        result = _kalman_smooth_single(y, dV=0.16, dW=0.001)
        assert len(result) == len(y)

    def test_mean_centered(self):
        y = np.random.RandomState(42).randn(100)
        result = _kalman_smooth_single(y, dV=0.16, dW=0.001)
        assert abs(np.mean(result)) < 1e-10

    def test_smoothing_reduces_variance(self):
        y = np.random.RandomState(42).randn(100)
        result = _kalman_smooth_single(y, dV=0.16, dW=0.001)
        assert np.std(result) < np.std(y)


class TestDlmSmooth:
    def test_shape(self):
        mat = np.random.RandomState(42).randn(100, 10)
        result = dlm_smooth(mat, dV=0.16, dW=0.001)
        assert result.shape == mat.shape

    def test_columns_mean_centered(self):
        mat = np.random.RandomState(42).randn(100, 10)
        result = dlm_smooth(mat, dV=0.16, dW=0.001)
        for c in range(10):
            assert abs(np.mean(result[:, c])) < 1e-10
