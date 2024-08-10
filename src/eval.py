import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
from scipy.stats import ks_2samp


def calculate_ece(df, n_bins=10):
    """
    Calculate the Expected Calibration Error (ECE). Lower ECE means better calibration.
    
    Parameters:
    - df: dataframe contain probabilities and labels
    - n_bins: int, the number of bins to divide the probability space.
    
    Returns:
    - ECE: the expected calibration error.
    """
    probabilities, labels = df['probability'], df['label']

    bins = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Indices of probabilities within this bin
        in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
        prop_in_bin = in_bin.mean() # Proportion of predictions in this bin
        
        if prop_in_bin > 0:
            # Average probability in this bin
            avg_prob_in_bin = probabilities[in_bin].mean()
            # Actual average outcome in this bin
            avg_true_in_bin = labels[in_bin].mean()
            # Weighted absolute difference
            ece += np.abs(avg_prob_in_bin - avg_true_in_bin) * prop_in_bin
    
    return ece

def calculate_brier_score(df):

#Calculate the Brier score.,calibration smaller, better

    probabilities, labels = df['probability'], df['label']
    brier_score = brier_score_loss(labels, probabilities)
    return brier_score

def calculate_auroc(df):

    probabilities, labels = df['probability'], df['label']
    auc_score = roc_auc_score(labels, probabilities)
    if auc_score<0.5:
        auc_score = 1 - auc_score
    return auc_score

def calculate_KS_stat(df):

    probabilities, labels = df['probability'], df['label']
    probabilities_0 = probabilities[labels == 0]
    probabilities_1 = probabilities[labels == 1]
    # Calculate the Kolmogorov-Smirnov statistic
    ks_statistic, p_value = ks_2samp(probabilities_0, probabilities_1)

    return ks_statistic



if __name__ == "__main__":

    # test
    n = 500
    np.random.seed(42)

    index = np.arange(n)

    probability = np.random.rand(n)

    label = np.random.randint(2, size=n)

    df = pd.DataFrame({'index': index, 'probability': probability, 'label': label})

    print(f'ECE: {calculate_ece(df)}')
    print(f'Beier score: {calculate_brier_score(df)}')
    print(f'AUROC: {calculate_auroc(df)}')
    print(f'KS stat: {calculate_KS_stat(df)}')


