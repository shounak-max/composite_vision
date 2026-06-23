"""
Statistical significance testing for CompositeVision benchmark.

Implements paired statistical tests for comparing model performance on 
the same set of stimuli, following best practices for ML model comparison 
(Dietterich 1998, Demsar 2006).

Tests included:
- McNemar's test (paired per-image comparison)
- Bootstrap confidence intervals (10,000 resamples)
- Cohen's h (effect size for proportion differences)
- Bonferroni correction for multiple comparisons
"""

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
import os
import json


def mcnemar_test(preds_a, preds_b, labels):
    """
    McNemar's test for paired nominal data.
    
    Tests whether two models have significantly different error rates
    on the same set of examples. This is the correct test for paired
    per-image predictions (not a chi-squared test on aggregate accuracy).
    
    Args:
        preds_a: array of predictions from model A
        preds_b: array of predictions from model B
        labels: array of ground truth labels
        
    Returns:
        dict with keys: chi2, p_value, significant (at alpha=0.05),
        n_ab (A correct, B wrong), n_ba (A wrong, B correct)
    """
    correct_a = (np.array(preds_a) == np.array(labels))
    correct_b = (np.array(preds_b) == np.array(labels))
    
    # Contingency counts
    # n_ab: A correct and B incorrect
    n_ab = np.sum(correct_a & ~correct_b)
    # n_ba: A incorrect and B correct
    n_ba = np.sum(~correct_a & correct_b)
    
    # McNemar's test statistic (with continuity correction)
    if n_ab + n_ba == 0:
        return {
            'chi2': 0.0,
            'p_value': 1.0,
            'significant': False,
            'n_ab': int(n_ab),
            'n_ba': int(n_ba),
        }
    
    chi2 = (abs(n_ab - n_ba) - 1) ** 2 / (n_ab + n_ba)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return {
        'chi2': round(float(chi2), 4),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'n_ab': int(n_ab),
        'n_ba': int(n_ba),
    }


def bootstrap_accuracy_ci(preds, labels, n_resamples=10000, ci=0.95, seed=42):
    """
    Non-parametric bootstrap confidence interval for classification accuracy.
    
    Resamples (with replacement) the paired (prediction, label) tuples and 
    computes accuracy on each resample to estimate the sampling distribution.
    
    Args:
        preds: array of model predictions
        labels: array of ground truth labels
        n_resamples: number of bootstrap resamples (default 10,000)
        ci: confidence level (default 0.95)
        seed: random seed for reproducibility
        
    Returns:
        tuple: (mean_accuracy, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    preds = np.array(preds)
    labels = np.array(labels)
    n = len(preds)
    
    correct = (preds == labels).astype(float)
    point_estimate = correct.mean()
    
    boot_accs = np.empty(n_resamples)
    for i in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        boot_accs[i] = correct[indices].mean()
    
    alpha = 1 - ci
    ci_lower = np.percentile(boot_accs, 100 * alpha / 2)
    ci_upper = np.percentile(boot_accs, 100 * (1 - alpha / 2))
    
    return (
        round(float(point_estimate), 6),
        round(float(ci_lower), 6),
        round(float(ci_upper), 6),
    )


def bootstrap_accuracy_ci_grouped(preds, labels, groups, n_resamples=10000, ci=0.95, seed=42):
    """
    Bootstrap CI for accuracy within each group (e.g., per composition type).
    
    Args:
        preds, labels: arrays of predictions and ground truth
        groups: array of group labels (e.g., composition type)
        n_resamples, ci, seed: bootstrap parameters
        
    Returns:
        dict mapping group_name -> (mean, ci_lower, ci_upper)
    """
    preds = np.array(preds)
    labels = np.array(labels)
    groups = np.array(groups)
    
    results = {}
    for group in np.unique(groups):
        mask = groups == group
        results[group] = bootstrap_accuracy_ci(
            preds[mask], labels[mask], n_resamples, ci, seed
        )
    return results


def cohens_h(p1, p2):
    """
    Cohen's h effect size for the difference between two proportions.
    
    Uses the arcsine transformation: h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    
    Interpretation (Cohen 1988):
        |h| < 0.2: negligible
        0.2 <= |h| < 0.5: small
        0.5 <= |h| < 0.8: medium
        |h| >= 0.8: large
        
    Args:
        p1: proportion for group 1 (e.g., model A accuracy)
        p2: proportion for group 2 (e.g., model B accuracy)
        
    Returns:
        float: Cohen's h value
    """
    h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
    return round(float(h), 4)


def interpret_cohens_h(h):
    """Return human-readable interpretation of Cohen's h."""
    abs_h = abs(h)
    if abs_h < 0.2:
        return "negligible"
    elif abs_h < 0.5:
        return "small"
    elif abs_h < 0.8:
        return "medium"
    else:
        return "large"


def bonferroni_correct(p_values, alpha=0.05):
    """
    Bonferroni correction for multiple comparisons.
    
    Adjusts the significance threshold by dividing alpha by the number of tests.
    This is conservative but guarantees family-wise error rate control.
    
    Args:
        p_values: list of p-values from individual tests
        alpha: desired family-wise error rate (default 0.05)
        
    Returns:
        list of dicts with keys: original_p, corrected_threshold, significant
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests
    
    return [
        {
            'original_p': round(float(p), 6),
            'corrected_threshold': round(float(corrected_alpha), 6),
            'significant': p < corrected_alpha,
        }
        for p in p_values
    ]


def run_all_pairwise_tests(df, models=None):
    """
    Run McNemar's test, Cohen's h, and bootstrap CIs for all model pairs.
    
    Args:
        df: DataFrame with columns: filename, model, top1_pred, class1
        models: list of model names to compare (default: all unique models)
        
    Returns:
        DataFrame with pairwise comparison results
    """
    if models is None:
        models = sorted(df['model'].unique())
    
    # Ensure consistent ordering by filename for paired comparison
    results = []
    
    for m_a, m_b in combinations(models, 2):
        df_a = df[df['model'] == m_a].sort_values('filename')
        df_b = df[df['model'] == m_b].sort_values('filename')
        
        # Ensure same set of filenames
        common = set(df_a['filename']) & set(df_b['filename'])
        if len(common) == 0:
            continue
            
        df_a = df_a[df_a['filename'].isin(common)].sort_values('filename')
        df_b = df_b[df_b['filename'].isin(common)].sort_values('filename')
        
        preds_a = df_a['top1_pred'].values
        preds_b = df_b['top1_pred'].values
        labels = df_a['class1'].values
        
        # McNemar's test
        mcnemar = mcnemar_test(preds_a, preds_b, labels)
        
        # Accuracies with bootstrap CIs
        acc_a = bootstrap_accuracy_ci(preds_a, labels)
        acc_b = bootstrap_accuracy_ci(preds_b, labels)
        
        # Effect size
        h = cohens_h(acc_a[0], acc_b[0])
        
        results.append({
            'model_a': m_a,
            'model_b': m_b,
            'acc_a': f"{acc_a[0]*100:.2f}%",
            'ci_a': f"[{acc_a[1]*100:.2f}%, {acc_a[2]*100:.2f}%]",
            'acc_b': f"{acc_b[0]*100:.2f}%",
            'ci_b': f"[{acc_b[1]*100:.2f}%, {acc_b[2]*100:.2f}%]",
            'diff_pp': f"{(acc_a[0] - acc_b[0])*100:+.2f}pp",
            'mcnemar_chi2': mcnemar['chi2'],
            'mcnemar_p': mcnemar['p_value'],
            'cohens_h': h,
            'effect_size': interpret_cohens_h(h),
            'significant_p05': mcnemar['significant'],
        })
    
    df_results = pd.DataFrame(results)
    
    # Apply Bonferroni correction
    if len(df_results) > 0:
        corrections = bonferroni_correct(df_results['mcnemar_p'].values)
        df_results['bonferroni_significant'] = [c['significant'] for c in corrections]
        df_results['corrected_threshold'] = [c['corrected_threshold'] for c in corrections]
    
    return df_results


def compute_per_model_bootstrap_cis(df, models=None, n_resamples=10000):
    """
    Compute bootstrap 95% CIs for each model's overall accuracy and per-composition accuracy.
    
    Args:
        df: DataFrame with columns: filename, model, top1_pred, class1, composition_type
        models: list of model names (default: all)
        n_resamples: number of bootstrap resamples
        
    Returns:
        dict with 'overall' and 'by_composition' DataFrames
    """
    if models is None:
        models = sorted(df['model'].unique())
    
    overall_rows = []
    comp_rows = []
    
    for model in models:
        df_m = df[df['model'] == model]
        
        # Overall
        mean, lo, hi = bootstrap_accuracy_ci(
            df_m['top1_pred'].values, df_m['class1'].values, n_resamples
        )
        overall_rows.append({
            'model': model,
            'accuracy': f"{mean*100:.2f}%",
            'ci_95': f"[{lo*100:.2f}%, {hi*100:.2f}%]",
            'accuracy_raw': mean,
            'ci_lower': lo,
            'ci_upper': hi,
        })
        
        # Per composition type
        if 'composition_type' in df_m.columns:
            for comp in sorted(df_m['composition_type'].unique()):
                df_mc = df_m[df_m['composition_type'] == comp]
                mean_c, lo_c, hi_c = bootstrap_accuracy_ci(
                    df_mc['top1_pred'].values, df_mc['class1'].values, n_resamples
                )
                comp_rows.append({
                    'model': model,
                    'composition_type': comp,
                    'accuracy': f"{mean_c*100:.2f}%",
                    'ci_95': f"[{lo_c*100:.2f}%, {hi_c*100:.2f}%]",
                    'accuracy_raw': mean_c,
                    'ci_lower': lo_c,
                    'ci_upper': hi_c,
                })
    
    return {
        'overall': pd.DataFrame(overall_rows),
        'by_composition': pd.DataFrame(comp_rows),
    }


def generate_statistical_report(results_dir, output_dir=None):
    """
    Generate a complete statistical significance report from benchmark results.
    
    Reads benchmark_results.csv and produces:
    1. Pairwise McNemar's test results
    2. Bootstrap 95% CIs for all accuracies
    3. Effect sizes (Cohen's h) 
    4. Summary tables saved as CSV
    
    Args:
        results_dir: directory containing benchmark_results.csv
        output_dir: where to save statistical results (default: results_dir)
    """
    if output_dir is None:
        output_dir = results_dir
    
    csv_path = os.path.join(results_dir, "benchmark_results.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} predictions from {len(df['model'].unique())} models")
    
    # 1. Pairwise comparisons
    print("\n=== Pairwise Statistical Comparisons ===")
    pairwise_df = run_all_pairwise_tests(df)
    pairwise_path = os.path.join(output_dir, "pairwise_statistical_tests.csv")
    pairwise_df.to_csv(pairwise_path, index=False)
    print(f"Saved pairwise tests to {pairwise_path}")
    
    # Print significant results
    sig = pairwise_df[pairwise_df['bonferroni_significant']]
    if len(sig) > 0:
        print(f"\nSignificant differences (Bonferroni-corrected, α=0.05):")
        for _, row in sig.iterrows():
            print(f"  {row['model_a']} vs {row['model_b']}: "
                  f"Δ={row['diff_pp']}, p={row['mcnemar_p']:.6f}, "
                  f"|h|={abs(row['cohens_h']):.3f} ({row['effect_size']})")
    else:
        print("\nNo pairwise differences survived Bonferroni correction.")
    
    # 2. Per-model bootstrap CIs
    print("\n=== Bootstrap 95% Confidence Intervals ===")
    ci_results = compute_per_model_bootstrap_cis(df)
    
    overall_path = os.path.join(output_dir, "accuracy_bootstrap_ci.csv")
    ci_results['overall'].to_csv(overall_path, index=False)
    print(f"\nOverall accuracy with 95% CIs:")
    for _, row in ci_results['overall'].sort_values('accuracy_raw', ascending=False).iterrows():
        print(f"  {row['model']:20s}: {row['accuracy']:>8s} {row['ci_95']}")
    
    comp_path = os.path.join(output_dir, "composition_bootstrap_ci.csv")
    ci_results['by_composition'].to_csv(comp_path, index=False)
    print(f"\nSaved composition-level CIs to {comp_path}")
    
    # 3. Summary JSON
    summary = {
        'n_models': len(df['model'].unique()),
        'n_stimuli_per_model': int(df.groupby('model').size().iloc[0]),
        'n_pairwise_tests': len(pairwise_df),
        'n_significant_bonferroni': int(pairwise_df['bonferroni_significant'].sum()),
        'models': list(df['model'].unique()),
    }
    summary_path = os.path.join(output_dir, "statistical_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Statistical report complete. Files saved to {output_dir} ===")
    return pairwise_df, ci_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run statistical significance tests on CompositeVision benchmark results")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory containing benchmark_results.csv")
    args = parser.parse_args()
    
    if args.results_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        args.results_dir = os.path.join(base_dir, "results")
    
    generate_statistical_report(args.results_dir)
