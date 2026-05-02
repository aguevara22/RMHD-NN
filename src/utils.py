import numpy as np

def moving_average(series, window):
    if len(series) == 0:
        return series
    if len(series) < window:
        return series[:]  # not enough points; just return raw
    a = np.asarray(series, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / window
    ma_valid = np.convolve(a, kernel, mode="valid")
    # pad the front so output length matches input length
    ma = np.concatenate([np.full(window - 1, ma_valid[0]), ma_valid])
    return ma.tolist()
