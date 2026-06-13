import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

def compute_wilson_ci(k, n, z=1.96):
    """
    Compute the Wilson score confidence interval for a binomial proportion.
    
    More accurate than the normal approximation (Wald interval) for small n,
    which is critical for our per-composition-type sample sizes (n~80-160).
    
    Args:
        k: number of successes
        n: number of trials
        z: z-score (1.96 for 95% CI, 2.576 for 99% CI)
    
    Returns:
        (lower, upper) bounds of the confidence interval
    """
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return (max(0.0, center - margin), min(1.0, center + margin))


def compute_mcnemar_test(correct_a, correct_b):
    """
    Compute McNemar's test for paired nominal data (two classifiers on same samples).
    
    Tests whether two models have the same error rate. More appropriate than 
    independent proportion tests because the same images are used for both models.
    
    The test statistic uses the continuity-corrected form:
        chi2 = (|b - c| - 1)^2 / (b + c)
    where b = A correct & B wrong, c = A wrong & B correct.
    
    Args:
        correct_a: boolean array, True where model A is correct
        correct_b: boolean array, True where model B is correct
    
    Returns:
        dict with 'chi2', 'p_value', 'n_discordant', 'b' (A right B wrong), 'c' (A wrong B right)
    """
    from scipy import stats
    
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    
    # b: model A correct, model B wrong
    b = np.sum(correct_a & ~correct_b)
    # c: model A wrong, model B correct
    c = np.sum(~correct_a & correct_b)
    
    n_discordant = b + c
    
    if n_discordant == 0:
        # Models agree on every sample — no test possible
        return {'chi2': 0.0, 'p_value': 1.0, 'n_discordant': 0, 'b': int(b), 'c': int(c)}
    
    # Continuity-corrected McNemar's test
    chi2 = (abs(b - c) - 1)**2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return {
        'chi2': round(chi2, 4),
        'p_value': round(p_value, 6),
        'n_discordant': int(n_discordant),
        'b': int(b),
        'c': int(c)
    }


def compute_error_consistency(df):
    # Fair metric: only class1 (primary shape) counts as correct.
    # This ensures all models are evaluated on the same standard.
    df['is_correct'] = (df['top1_pred'] == df['class1'])
    
    models = df['model'].unique()
    
    consistency_matrix = pd.DataFrame(index=models, columns=models)
    for m1 in models:
        for m2 in models:
            preds1 = df[df['model'] == m1].sort_values('filename')['top1_pred'].values
            preds2 = df[df['model'] == m2].sort_values('filename')['top1_pred'].values
            agreement = np.mean(preds1 == preds2)
            consistency_matrix.loc[m1, m2] = agreement
            
    comp_types = df['composition_type'].unique()
    comp_consistency = {}
    for comp in comp_types:
        df_comp = df[df['composition_type'] == comp]
        agreements = []
        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                p1 = df_comp[df_comp['model'] == m1].sort_values('filename')['top1_pred'].values
                p2 = df_comp[df_comp['model'] == m2].sort_values('filename')['top1_pred'].values
                agreements.append(np.mean(p1 == p2))
        comp_consistency[comp] = np.mean(agreements) if agreements else 0.0
        
    salience_types = df['salience'].unique()
    sal_consistency = {}
    for sal in salience_types:
        df_sal = df[df['salience'] == sal]
        agreements = []
        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                p1 = df_sal[df_sal['model'] == m1].sort_values('filename')['top1_pred'].values
                p2 = df_sal[df_sal['model'] == m2].sort_values('filename')['top1_pred'].values
                agreements.append(np.mean(p1 == p2))
        sal_consistency[sal] = np.mean(agreements) if agreements else 0.0

    return consistency_matrix, comp_consistency, sal_consistency


def compute_statistical_tests(df, results_dir):
    """
    Compute pairwise McNemar's tests and per-model Wilson CIs.
    Saves results to statistical_tests.csv and updates model_accuracy.csv.
    """
    df['is_correct'] = (df['top1_pred'] == df['class1'])
    models = sorted(df['model'].unique())
    
    # --- Pairwise McNemar's tests ---
    mcnemar_rows = []
    for m1, m2 in combinations(models, 2):
        df_m1 = df[df['model'] == m1].sort_values('filename')
        df_m2 = df[df['model'] == m2].sort_values('filename')
        
        correct_a = df_m1['is_correct'].values
        correct_b = df_m2['is_correct'].values
        
        result = compute_mcnemar_test(correct_a, correct_b)
        
        acc_a = correct_a.mean()
        acc_b = correct_b.mean()
        
        mcnemar_rows.append({
            'model_a': m1,
            'model_b': m2,
            'accuracy_a': round(acc_a, 4),
            'accuracy_b': round(acc_b, 4),
            'accuracy_diff': round(acc_a - acc_b, 4),
            'chi2': result['chi2'],
            'p_value': result['p_value'],
            'significant_at_0.05': result['p_value'] < 0.05,
            'significant_at_0.01': result['p_value'] < 0.01,
            'n_discordant': result['n_discordant'],
            'a_right_b_wrong': result['b'],
            'a_wrong_b_right': result['c'],
            'n_samples': len(correct_a)
        })
    
    mcnemar_df = pd.DataFrame(mcnemar_rows)
    mcnemar_df.to_csv(os.path.join(results_dir, "statistical_tests.csv"), index=False)
    print(f"Saved pairwise McNemar's tests ({len(mcnemar_rows)} pairs) to statistical_tests.csv")
    
    # --- Per-model Wilson CIs ---
    acc_rows = []
    for model in models:
        df_m = df[df['model'] == model]
        n = len(df_m)
        k = df_m['is_correct'].sum()
        acc = k / n
        lower, upper = compute_wilson_ci(k, n)
        
        acc_rows.append({
            'model': model,
            'is_correct': round(acc, 6),
            'n_samples': n,
            'n_correct': int(k),
            'lower_ci_95': round(lower, 4),
            'upper_ci_95': round(upper, 4),
            'ci_width': round(upper - lower, 4)
        })
    
    acc_df = pd.DataFrame(acc_rows)
    acc_df.to_csv(os.path.join(results_dir, "model_accuracy.csv"), index=False)
    print("Updated model_accuracy.csv with Wilson 95% CIs")
    
    return mcnemar_df, acc_df


def generate_reports_and_figures(results_dir):
    csv_path = os.path.join(results_dir, "benchmark_results.csv")
    if not os.path.exists(csv_path):
        print(f"Results not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    consistency_matrix, comp_consistency, sal_consistency = compute_error_consistency(df)
    
    consistency_matrix.to_csv(os.path.join(results_dir, "model_error_consistency.csv"))
    
    # --- Statistical tests ---
    mcnemar_df = None
    try:
        mcnemar_df, acc_df = compute_statistical_tests(df, results_dir)
        has_stats = True
    except ImportError:
        print("Warning: scipy not available. Skipping McNemar's tests. Install with: pip install scipy")
        has_stats = False
        # Fallback: compute basic accuracy without CIs
        df['is_correct'] = (df['top1_pred'] == df['class1'])
        acc_df = df.groupby('model')['is_correct'].mean().reset_index()
        acc_df.to_csv(os.path.join(results_dir, "model_accuracy.csv"), index=False)
    
    # --- 0. Error Consistency Heatmap ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(consistency_matrix.astype(float), annot=True, cmap='coolwarm', vmin=0, vmax=1)
    plt.title('Model-Model Error Consistency')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "consistency_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir, "consistency_heatmap.pdf"), bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.bar(comp_consistency.keys(), comp_consistency.values())
    plt.title('Composition-wise Error Consistency')
    plt.ylabel('Average Agreement')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "composition_consistency.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir, "composition_consistency.pdf"), bbox_inches='tight')
    plt.close()
    
    # --- 1. Overall Model Accuracy Bar Chart WITH Error Bars ---
    df['is_correct'] = (df['top1_pred'] == df['class1'])
    acc = df.groupby('model')['is_correct'].mean().reset_index()
    
    # Compute error bars from Wilson CIs
    error_lower = []
    error_upper = []
    for _, row in acc.iterrows():
        model = row['model']
        df_m = df[df['model'] == model]
        n = len(df_m)
        k = int(df_m['is_correct'].sum())
        p = k / n
        lo, hi = compute_wilson_ci(k, n)
        error_lower.append(p - lo)
        error_upper.append(hi - p)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(acc)), acc['is_correct'], 
                  yerr=[error_lower, error_upper],
                  capsize=5, color=sns.color_palette("muted", len(acc)),
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(acc)))
    ax.set_xticklabels(acc['model'], rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Overall Accuracy by Model (with 95% Wilson CIs)')
    ax.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for i, (bar, lo, hi) in enumerate(zip(bars, error_lower, error_upper)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + hi + 0.02,
                f'{height:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "model_accuracy.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir, "model_accuracy.pdf"), bbox_inches='tight')
    plt.close()
    
    # --- 2. Accuracy by Composition Type ---
    comp_acc = df.groupby(['model', 'composition_type'])['is_correct'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=comp_acc, x='composition_type', y='is_correct', hue='model')
    plt.title('Accuracy by Composition Type')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "composition_accuracy.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir, "composition_accuracy.pdf"), bbox_inches='tight')
    plt.close()
    
    # --- 3. Accuracy by Salience Level ---
    sal_acc = df.groupby(['model', 'salience'])['is_correct'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=sal_acc, x='salience', y='is_correct', hue='model', marker='o')
    plt.title('Accuracy by Salience Level')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "salience_accuracy.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir, "salience_accuracy.pdf"), bbox_inches='tight')
    plt.close()
    
    # --- 4. Inference Time Comparison ---
    if 'inference_time' in df.columns:
        # Only plot models that have actual timing data (not NaN)
        inf_time = df.groupby('model')['inference_time'].mean().reset_index()
        inf_time_valid = inf_time.dropna(subset=['inference_time'])
        
        if len(inf_time_valid) > 0:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=inf_time_valid, x='model', y='inference_time')
            plt.title('Average Inference Time by Model')
            plt.ylabel('Inference Time (s / image)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "inference_time.png"), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(results_dir, "inference_time.pdf"), bbox_inches='tight')
            plt.close()
        
    # --- 5. Comprehensive Radar Chart ---
    models_to_compare = df['model'].unique()
    categories = ['Overall Acc', 'AdaIN Robustness', 'Occlusion Robustness', 'High Salience Acc', 'Low Salience Acc']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, size=12)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["20%", "40%", "60%", "80%", "100%"], color="grey", size=10)
    plt.ylim(0, 1)
    
    for model in models_to_compare:
        df_m = df[df['model'] == model]
        if len(df_m) == 0: continue
        
        overall = df_m['is_correct'].mean()
        adain = df_m[df_m['composition_type'] == 'adain']['is_correct'].mean()
        occlusion = df_m[df_m['composition_type'] == 'occlusion']['is_correct'].mean()
        high_sal = df_m[df_m['salience'] == 'high']['is_correct'].mean()
        low_sal = df_m[df_m['salience'] == 'low']['is_correct'].mean()
        
        # Handle nan values if a subset is missing
        values = [
            overall if not np.isnan(overall) else 0,
            adain if not np.isnan(adain) else 0,
            occlusion if not np.isnan(occlusion) else 0,
            high_sal if not np.isnan(high_sal) else 0,
            low_sal if not np.isnan(low_sal) else 0
        ]
        values += values[:1]
        
        linewidth = 3 if model == 'RL_Attention' else 1
        alpha = 0.25 if model == 'RL_Attention' else 0.1
        ax.plot(angles, values, linewidth=linewidth, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=alpha)
        
    plt.title('Comprehensive Model Comparison (Aspects)', size=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "model_comparison_radar.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir, "model_comparison_radar.pdf"), bbox_inches='tight')
    plt.close()
    
    # --- 6. McNemar's Significance Heatmap ---
    if has_stats and mcnemar_df is not None:
        models_list = sorted(df['model'].unique())
        n_models = len(models_list)
        pval_matrix = np.ones((n_models, n_models))
        
        for _, row in mcnemar_df.iterrows():
            i = models_list.index(row['model_a'])
            j = models_list.index(row['model_b'])
            pval_matrix[i, j] = row['p_value']
            pval_matrix[j, i] = row['p_value']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create annotation labels showing p-value and significance stars
        annot_labels = np.empty_like(pval_matrix, dtype=object)
        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    annot_labels[i, j] = "—"
                else:
                    p = pval_matrix[i, j]
                    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    annot_labels[i, j] = f"{p:.3f}\n{stars}"
        
        sns.heatmap(pval_matrix, annot=annot_labels, fmt='', 
                    xticklabels=models_list, yticklabels=models_list,
                    cmap='RdYlGn', vmin=0, vmax=0.1,
                    cbar_kws={'label': 'p-value'})
        ax.set_title("McNemar's Pairwise Significance Tests\n(* p<0.05, ** p<0.01, *** p<0.001, ns = not significant)")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "mcnemar_significance.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(results_dir, "mcnemar_significance.pdf"), bbox_inches='tight')
        plt.close()
        print("Generated McNemar's significance heatmap.")
    
    print("Metrics and figures generated successfully.")

if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    generate_reports_and_figures(os.path.join(base_dir, "results"))
