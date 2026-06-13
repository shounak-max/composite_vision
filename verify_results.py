import pandas as pd
import os

df = pd.read_csv('results/benchmark_results.csv')
print('=== DATASET SIZE ===')
print(f'Total rows: {len(df)}')
print(f'Models: {df["model"].unique().tolist()}')
n_models = len(df["model"].unique())
print(f'Test images per model: {len(df) // n_models}')
print()

print('=== INFERENCE TIMES (Figure 8 fix) ===')
inf = df.groupby('model')['inference_time'].mean()
print(inf.to_string())
print()

print('=== OVERALL ACCURACY ===')
df['is_correct'] = (df['top1_pred'] == df['class1'])
acc = df.groupby('model')['is_correct'].mean().sort_values(ascending=False)
print(acc.to_string())
print()

print('=== SHAPE BIAS ===')
sb = pd.read_csv('results/shape_bias.csv')
print(sb.to_string(index=False))
print()

print('=== NEW FILES ===')
for f in ['shape_bias.csv', 'shape_bias.png', 'shape_bias.pdf', 
          'inference_time.png', 'model_accuracy.csv', 'model_accuracy.png']:
    path = os.path.join('results', f)
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    print(f'  {f}: {"EXISTS" if exists else "MISSING"} ({size} bytes)')
