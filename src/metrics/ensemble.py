import pandas as pd
import os

def evaluate_ensemble():
    base_dir = "d:/gitfork/composite_vision_research"
    results_file = os.path.join(base_dir, "results", "benchmark_results.csv")
    accuracy_file = os.path.join(base_dir, "results", "model_accuracy.csv")
    
    if not os.path.exists(results_file):
        print("Results file not found.")
        return

    df = pd.read_csv(results_file)
    
    # We want to ensemble the top 3 models: ViT-B/16, ConvNeXt, and RL_Attention
    models_to_ensemble = ['ViT-B/16', 'ConvNeXt', 'RL_Attention']
    df_ensemble = df[df['model'].isin(models_to_ensemble)]
    
    ensemble_results = []
    
    # Group by filename
    for filename, group in df_ensemble.groupby('filename'):
        class1 = group['class1'].iloc[0]
        class2 = group['class2'].iloc[0]
        
        # Confidence-weighted voting
        votes = {}
        for _, row in group.iterrows():
            pred = row['top1_pred']
            conf = row['top1_conf']
            if pred not in votes:
                votes[pred] = 0
            votes[pred] += conf
            
        # Hard majority voting as fallback if confidences are weird
        best_pred = max(votes.items(), key=lambda x: x[1])[0]
        
        is_correct = (best_pred == class1) or (best_pred == class2)
        ensemble_results.append(is_correct)
        
    accuracy = sum(ensemble_results) / len(ensemble_results)
    print(f"Ensemble (ViT + ConvNeXt + RL_Attention) Accuracy: {accuracy * 100:.2f}%")
    
    # Update model_accuracy.csv
    acc_df = pd.read_csv(accuracy_file)
    if 'Ensemble' in acc_df['model'].values:
        acc_df.loc[acc_df['model'] == 'Ensemble', 'is_correct'] = accuracy
    else:
        new_row = pd.DataFrame({'model': ['Ensemble'], 'is_correct': [accuracy]})
        acc_df = pd.concat([acc_df, new_row], ignore_index=True)
        
    # Sort for neatness
    acc_df = acc_df.sort_values(by='is_correct', ascending=True)
    acc_df.to_csv(accuracy_file, index=False)
    print("Updated model_accuracy.csv with Ensemble results.")

if __name__ == '__main__':
    evaluate_ensemble()
