import torch
from src.models.baselines import load_models
from src.models.rl_attention import RecurrentAttentionModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models, _ = load_models(device)
    
    print("--- Baseline Models Parameters ---")
    for name, model in models.items():
        print(f"{name}: {count_parameters(model):,}")
        
    print("--- RL Attention Agent Parameters ---")
    rl_model = RecurrentAttentionModel(num_classes=1000).to(device)
    # The early layers of ResNet18 were unfrozen recently, so we count all
    print(f"RL_Attention: {count_parameters(rl_model):,}")

if __name__ == '__main__':
    main()
