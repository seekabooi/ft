def compute_difficulty(horizon, volatility):
    if horizon <= 1 and volatility < 0.5:
        return 1
    elif horizon <= 3:
        return 2
    elif horizon <= 7:
        return 3
    else:
        return 4