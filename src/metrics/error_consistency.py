import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_error_consistency(df):
    df['is_correct'] = (df['top1_pred'] == df['class1']) | (df['top1_pred'] == df['class2'])
    
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

def generate_reports_and_figures(results_dir):
    csv_path = os.path.join(results_dir, "benchmark_results.csv")
    if not os.path.exists(csv_path):
        print(f"Results not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    consistency_matrix, comp_consistency, sal_consistency = compute_error_consistency(df)
    
    consistency_matrix.to_csv(os.path.join(results_dir, "model_error_consistency.csv"))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(consistency_matrix.astype(float), annot=True, cmap='coolwarm', vmin=0, vmax=1)
    plt.title('Model-Model Error Consistency')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "consistency_heatmap.png"))
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.bar(comp_consistency.keys(), comp_consistency.values())
    plt.title('Composition-wise Error Consistency')
    plt.ylabel('Average Agreement')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "composition_consistency.png"))
    plt.close()
    
    df['is_correct'] = (df['top1_pred'] == df['class1']) | (df['top1_pred'] == df['class2'])
    acc = df.groupby('model')['is_correct'].mean().reset_index()
    acc.to_csv(os.path.join(results_dir, "model_accuracy.csv"), index=False)
    
    print("Metrics and figures generated successfully.")

if __name__ == '__main__':
    generate_reports_and_figures("d:/gitfork/composite_vision_research/results")
