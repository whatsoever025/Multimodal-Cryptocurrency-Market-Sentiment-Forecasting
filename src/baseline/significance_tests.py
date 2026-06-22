import numpy as np
from scipy.stats import norm
import json
import logging

logger = logging.getLogger(__name__)

def diebold_mariano_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray, h: int = 1):
    """
    Diebold-Mariano test for equal predictive accuracy.
    Tests if Model 2 is significantly different from Model 1.
    
    Args:
        y_true: Ground truth target values
        y_pred1: Predictions from the baseline model (e.g., XGBoost or Historical Mean)
        y_pred2: Predictions from the proposed model (e.g., MFN)
        h: Forecast horizon. For h>1, uses Newey-West standard errors with lag = h-1
           (e.g., for y_baseline t+8h, use h=8. For y_heuristic t+1h, use h=1).
           
    Returns:
        dm_stat: Test statistic (> 0 means Model 2 has smaller error/is better)
        p_value: Two-sided p-value
    """
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2
    
    # Loss differential (using squared error for R2/MSE equivalence)
    d = e1**2 - e2**2
    
    T = len(d)
    mean_d = np.mean(d)
    
    # Compute autocovariances for Newey-West estimator
    lag = h - 1
    gamma = np.zeros(lag + 1)
    
    for i in range(lag + 1):
        if i == 0:
            gamma[i] = np.sum((d - mean_d)**2) / T
        else:
            gamma[i] = np.sum((d[i:] - mean_d) * (d[:-i] - mean_d)) / T
            
    # Newey-West variance with Bartlett weights
    var_d = gamma[0]
    for i in range(1, lag + 1):
        weight = 1.0 - i / (lag + 1)
        var_d += 2 * weight * gamma[i]
        
    if var_d == 0:
        return 0.0, 1.0
        
    # Diebold-Mariano statistic
    dm_stat = mean_d / np.sqrt(var_d / T)
    
    # p-value (two-sided)
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    
    return float(dm_stat), float(p_value)

def clark_west_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray, h: int = 1):
    """
    Clark-West (2007) test for equal predictive accuracy in nested models.
    Use this when Model 1 is a simpler (nested) version of Model 2 
    (e.g., Model 1 = Historical Mean (predicts constant), Model 2 = ML Model).
    
    Args:
        y_true: Ground truth target values
        y_pred1: Predictions from the restricted/nested model (e.g., Historical Mean)
        y_pred2: Predictions from the unrestricted/larger model (e.g., LSTM)
        h: Forecast horizon (uses Newey-West standard errors)
           
    Returns:
        cw_stat: Test statistic
        p_value: One-sided p-value (CW test is theoretically one-sided)
    """
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2
    
    # CW adjusted loss differential:
    # Subtracts the noise introduced by estimating the larger model's extra parameters
    f = e1**2 - e2**2 + (y_pred1 - y_pred2)**2
    
    T = len(f)
    mean_f = np.mean(f)
    
    lag = h - 1
    gamma = np.zeros(lag + 1)
    
    for i in range(lag + 1):
        if i == 0:
            gamma[i] = np.sum((f - mean_f)**2) / T
        else:
            gamma[i] = np.sum((f[i:] - mean_f) * (f[:-i] - mean_f)) / T
            
    var_f = gamma[0]
    for i in range(1, lag + 1):
        weight = 1.0 - i / (lag + 1)
        var_f += 2 * weight * gamma[i]
        
    if var_f == 0:
        return 0.0, 1.0
        
    cw_stat = mean_f / np.sqrt(var_f / T)
    
    # One-sided p-value (we test H0: MSE1 = MSE2 vs H1: MSE2 < MSE1)
    p_value = 1 - norm.cdf(cw_stat)
    
    return float(cw_stat), float(p_value)

def example_usage():
    """
    This function demonstrates how you should extract your test predictions
    and run the test. Since baseline_results.json currently only stores metrics
    and not raw predictions, you need to save the test predictions from Kaggle.
    """
    print("=== Statistical Significance Testing (Diebold-Mariano / Clark-West) ===")
    print("Since your test predictions (y_pred) are not saved in the JSON, ")
    print("you will need to run this code on Kaggle or modify your scripts to save test predictions.")
    print("\nExample code you can run on Kaggle once you have the arrays:")
    
    print('''
    # Example: Comparing MFN Tabular vs XGBoost on y_baseline (h=8)
    # y_true, y_pred_xgb, y_pred_mfn must be numpy arrays of shape (13288,)
    
    # Use Diebold-Mariano for non-nested models (like XGBoost vs LSTM)
    dm_stat, p_val = diebold_mariano_test(y_true, y_pred_xgb, y_pred_mfn, h=8)
    
    if p_val < 0.05:
        print(f"Significant difference! DM={dm_stat:.3f}, p={p_val:.4f}")
    else:
        print(f"Not significant. DM={dm_stat:.3f}, p={p_val:.4f}")
        
        
    # Example: Comparing ML model against Historical Mean (Nested)
    # Use Clark-West for nested models
    cw_stat, p_val_cw = clark_west_test(y_true, y_pred_hist_mean, y_pred_mfn, h=8)
    ''')

if __name__ == "__main__":
    example_usage()
