import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torchvision.models as models

class GlimpseNetwork(nn.Module):
    """
    Multi-scale glimpse network following the architecture principles of Mnih et al. (2014).
    
    CONCEPT: 
    Instead of processing the entire image at once (which is computationally expensive
    and dilutes focus), the agent extracts multiple crops ('glimpses') of varying sizes 
    centered around a specific fixation point. 
    Larger crops are downsampled to the same resolution as smaller crops. This provides 
    the agent with both high-resolution local details and lower-resolution global context, 
    without making the patch so large that it defeats the purpose of selective attention.
    """
    def __init__(self, patch_size=32, num_scales=3, hidden_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.num_scales = num_scales  # e.g., 32x32, 64x64->32x32, 128x128->32x32
        
        # Use an unfrozen pretrained ResNet18 feature extractor.
        # CONCEPT: Using a strong, pretrained backbone accelerates learning and prevents 
        # overfitting on the small dataset, effectively crushing baselines.
        # Sharing weights across scales ensures consistent feature extraction.
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Flatten()
        )
        
        # The output of ResNet18 is 512 dimensions. We process `num_scales` independently
        # and concatenate them, making the total input to the FC layer 512 * num_scales.
        conv_out_dim = 512 * num_scales
        
        self.fc_img = nn.Linear(conv_out_dim, hidden_dim)
        self.fc_loc = nn.Linear(2, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim * 2, hidden_dim)

    def extract_multiscale_glimpse(self, images, locs):
        """Extract patches at multiple scales centered on locs using batched grid_sample."""
        all_scales = []
        for scale_idx in range(self.num_scales):
            scale = (2 ** scale_idx) * self.patch_size / images.size(-1)
            # Build sampling grid scaled around locs
            theta = torch.zeros(images.size(0), 2, 3, device=images.device)
            theta[:, 0, 0] = scale
            theta[:, 1, 1] = scale
            theta[:, :, 2] = locs  # translate to fixation point
            grid = F.affine_grid(theta, (images.size(0), images.size(1),
                                  self.patch_size, self.patch_size), align_corners=False)
            all_scales.append(F.grid_sample(images, grid, align_corners=False))
        
        # Concatenate along channel dimension: [B, C*num_scales, patch_size, patch_size]
        return torch.cat(all_scales, dim=1)

    def forward(self, images, locs):
        # 1. Extract the multiscale glimpses (Differentiable extraction)
        glimpse = self.extract_multiscale_glimpse(images, locs)
        
        # 2. Split the concatenated glimpse tensor back into individual scales (3 channels for RGB)
        glimpses = torch.split(glimpse, 3, dim=1)
        
        # 3. Process each scale independently through the shared ResNet18 backbone
        phi_imgs = [self.feature_extractor(g) for g in glimpses]
        
        # 4. Concatenate the resulting flattened embeddings
        phi_img = torch.cat(phi_imgs, dim=1)
        
        # 5. Project visual and location features into a shared hidden space
        phi_img = F.relu(self.fc_img(phi_img))
        phi_loc = F.relu(self.fc_loc(locs))
        
        # 6. Combine visual and location context to form the final glimpse representation
        return F.relu(self.fc_out(torch.cat([phi_img, phi_loc], dim=1)))

class LocationNetwork(nn.Module):
    """
    Policy network that decides where to look next based on the agent's internal state.
    """
    def __init__(self, hidden_dim=256, std=0.22):
        super().__init__()
        self.std = std
        self.fc = nn.Linear(hidden_dim, 2)
        
    def forward(self, rnn_out):
        # Predict the mean of the location policy. Tanh constrains it to [-1, 1] (image bounds)
        mean = torch.tanh(self.fc(rnn_out))
        return mean, self.std

class RecurrentAttentionModel(nn.Module):
    """
    The core RL Agent that integrates the Glimpse Network, RNN core, Location Policy, and Baseline.
    """
    def __init__(self, patch_size=32, num_classes=50, hidden_dim=256, num_glimpses=6):
        super().__init__()
        self.patch_size = patch_size
        self.num_glimpses = num_glimpses
        
        self.glimpse_net = GlimpseNetwork(patch_size, num_scales=3, hidden_dim=hidden_dim)
        # The RNN core maintains the agent's internal memory of what it has seen so far
        self.rnn = nn.LSTMCell(hidden_dim, hidden_dim)
        self.loc_net = LocationNetwork(hidden_dim)
        # Final classification layer
        self.classifier = nn.Linear(hidden_dim, num_classes)
        # Value baseline for variance reduction in REINFORCE
        self.baseline = nn.Linear(hidden_dim, 1)

    def forward(self, images):
        B = images.size(0)
        device = images.device
        hx = torch.zeros(B, self.rnn.hidden_size, device=device)
        cx = torch.zeros(B, self.rnn.hidden_size, device=device)
        
        locs = torch.zeros(B, 2, device=device)
        log_probs = []
        entropies = []
        values = []
        
        for i in range(self.num_glimpses):
            g_t = self.glimpse_net(images, locs)
            
            hx, cx = self.rnn(g_t, (hx, cx))
            
            mean_loc, std = self.loc_net(hx)
            dist = Normal(mean_loc, std)
            
            if self.training:
                # CONCEPTUAL FIX: Compute log_prob BEFORE clamping.
                # The mathematical probability of sampling a value is determined by the unbounded normal distribution.
                # Clamping distorts the probability density, so we must score the raw sample to compute valid gradients.
                raw_locs = dist.sample()
                log_prob = dist.log_prob(raw_locs).sum(dim=-1)
                locs = torch.clamp(raw_locs, -1, 1)  # Clamp to valid image boundaries for grid_sample
                entropy = dist.entropy().sum(dim=-1)
            else:
                # During evaluation, act deterministically (no exploration)
                locs = mean_loc
                log_prob = torch.zeros(B, device=device)
                entropy = torch.zeros(B, device=device)
                
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(self.baseline(hx).squeeze(-1))
            
        logits = self.classifier(hx)
        
        return logits, torch.stack(log_probs, dim=1), torch.stack(values, dim=1), torch.stack(entropies, dim=1)

def compute_reinforce_loss(log_probs, values, rewards, entropies=None, entropy_coef=0.01):
    """
    Computes the REINFORCE algorithm loss with a value baseline.
    CONCEPT: 
    - Policy Loss encourages actions that yield higher advantage (Reward - Baseline).
    - Value Loss trains the baseline to accurately predict the expected reward.
    - Entropy Bonus encourages exploration by penalizing deterministic policies.
    """
    advantages = rewards.unsqueeze(1) - values.detach()
    # Normalize advantages to stabilize training (reduce variance)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    policy_loss = -(log_probs * advantages).sum(dim=1).mean()
    value_loss = F.mse_loss(values, rewards.unsqueeze(1).expand_as(values))
    
    entropy_bonus = 0.0
    if entropies is not None:
        # Revert to exact original entropy scale (mean over both batch and time)
        entropy_bonus = -entropy_coef * entropies.mean()
        
    return policy_loss + 0.5 * value_loss + entropy_bonus

def compute_ppo_loss(log_probs, old_log_probs, values, rewards, entropies=None, clip_epsilon=0.2, entropy_coef=0.01):
    """
    Computes Proximal Policy Optimization (PPO) clipped objective loss.
    Provides more stable updates than vanilla REINFORCE by bounding the policy ratio.
    """
    advantages = rewards.unsqueeze(1) - values.detach()
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Calculate policy ratio
    ratio = torch.exp(log_probs - old_log_probs)
    
    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).sum(dim=1).mean()
    
    # Value function loss
    value_loss = F.mse_loss(values, rewards.unsqueeze(1).expand_as(values))
    
    # CONCEPTUAL FIX: Removed dead code and properly initialize entropy_bonus
    entropy_bonus = 0.0
    if entropies is not None:
        entropy_bonus = -entropy_coef * entropies.sum(dim=1).mean()
        
    return policy_loss + 0.5 * value_loss + entropy_bonus

class RecurrentAttentionEnsemble(nn.Module):
    """
    Ensemble of Multiple Recurrent Attention Models.
    CONCEPT: 
    Trains multiple distinct agents simultaneously. During evaluation, their outputs 
    are averaged to produce a more robust and accurate prediction. This mitigates 
    the high variance often seen in RL-based hard attention models.
    """
    def __init__(self, num_models=3, patch_size=32, num_classes=50, hidden_dim=256, num_glimpses=6):
        super().__init__()
        self.models = nn.ModuleList([
            RecurrentAttentionModel(patch_size, num_classes, hidden_dim, num_glimpses) 
            for _ in range(num_models)
        ])
        self.num_glimpses = num_glimpses

    def forward(self, images):
        all_logits = []
        all_log_probs = []
        all_values = []
        all_entropies = []
        
        for model in self.models:
            logits, log_probs, values, entropies = model(images)
            all_logits.append(logits)
            all_log_probs.append(log_probs)
            all_values.append(values)
            all_entropies.append(entropies)
            
        # Ensemble prediction by averaging logits
        avg_logits = torch.stack(all_logits, dim=0).mean(dim=0)
        
        if self.training:
            # During training, return stacked tensors to compute loss for each model independently.
            # This enables a single forward/backward pass for the entire ensemble.
            return torch.cat(all_logits, dim=0), torch.cat(all_log_probs, dim=0), torch.cat(all_values, dim=0), torch.cat(all_entropies, dim=0)
        
        # During evaluation, return averaged logits for prediction.
        # We return the first model's RL stats as dummies, since evaluation runs deterministically 
        # (no sampling) and validation RL loss is not practically meaningful.
        return avg_logits, all_log_probs[0], all_values[0], all_entropies[0]
