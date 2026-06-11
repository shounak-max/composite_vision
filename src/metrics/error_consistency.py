import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    # Fair metric: only class1 (primary shape) counts as correct
    df['is_correct'] = (df['top1_pred'] == df['class1'])
    acc = df.groupby('model')['is_correct'].mean().reset_index()
    acc.to_csv(os.path.join(results_dir, "model_accuracy.csv"), index=False)
    
    # 1. Overall Model Accuracy Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=acc, x='model', y='is_correct')
    plt.title('Overall Accuracy by Model')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "model_accuracy.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir, "model_accuracy.pdf"), bbox_inches='tight')
    plt.close()
    
    # 2. Accuracy by Composition Type
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
    
    # 3. Accuracy by Salience Level
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
    
    # 4. Inference Time Comparison
    if 'inference_time' in df.columns:
        inf_time = df.groupby('model')['inference_time'].mean().reset_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(data=inf_time, x='model', y='inference_time')
        plt.title('Average Inference Time by Model')
        plt.ylabel('Inference Time (s / image)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "inference_time.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(results_dir, "inference_time.pdf"), bbox_inches='tight')
        plt.close()
        
    # 5. Comprehensive Radar Chart for Model Comparison
    models_to_compare = df['model'].unique()
    categories = ['Overall Acc', 'AdaIN Robustness', 'Occlusion Robustness', 'High Salience Acc', 'Low Salience Acc']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Set the first axis to be on top
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
    
    print("Metrics and figures generated successfully.")

if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    generate_reports_and_figures(os.path.join(base_dir, "results"))
