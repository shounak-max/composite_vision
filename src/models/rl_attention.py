import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class GlimpseNetwork(nn.Module):
    def __init__(self, patch_size=32, hidden_dim=256):
        super().__init__()
        self.patch_size = patch_size
        import torchvision.models as models
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.conv = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
            
        conv_out_dim = 512
        self.fc_img = nn.Linear(conv_out_dim, hidden_dim)
        self.fc_loc = nn.Linear(2, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, images, locs):
        phi_img = F.relu(self.fc_img(self.conv(images)))
        phi_loc = F.relu(self.fc_loc(locs))
        return F.relu(self.fc_out(torch.cat([phi_img, phi_loc], dim=1)))

class LocationNetwork(nn.Module):
    def __init__(self, hidden_dim=256, std=0.1):
        super().__init__()
        self.std = std
        self.fc = nn.Linear(hidden_dim, 2)
        
    def forward(self, rnn_out):
        mean = torch.tanh(self.fc(rnn_out))
        return mean, self.std

class RecurrentAttentionModel(nn.Module):
    def __init__(self, patch_size=64, num_classes=50, hidden_dim=256, num_glimpses=6):
        super().__init__()
        self.patch_size = patch_size
        self.num_glimpses = num_glimpses
        
        self.glimpse_net = GlimpseNetwork(patch_size, hidden_dim)
        self.rnn = nn.LSTMCell(hidden_dim, hidden_dim)
        self.loc_net = LocationNetwork(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.baseline = nn.Linear(hidden_dim, 1)

    def extract_glimpse(self, images, locs):
        B, C, H, W = images.size()
        locs_scaled = (locs + 1.0) / 2.0
        x = (locs_scaled[:, 0] * W).long()
        y = (locs_scaled[:, 1] * H).long()
        
        glimpses = torch.zeros(B, C, self.patch_size, self.patch_size, device=images.device)
        for i in range(B):
            x0 = torch.clamp(x[i] - self.patch_size // 2, 0, W - self.patch_size)
            y0 = torch.clamp(y[i] - self.patch_size // 2, 0, H - self.patch_size)
            glimpses[i] = images[i, :, y0:y0+self.patch_size, x0:x0+self.patch_size]
        return glimpses

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
            glimpse = self.extract_glimpse(images, locs)
            g_t = self.glimpse_net(glimpse, locs)
            
            hx, cx = self.rnn(g_t, (hx, cx))
            
            mean_loc, std = self.loc_net(hx)
            dist = Normal(mean_loc, std)
            
            if self.training:
                locs = dist.rsample()
                locs = torch.clamp(locs, -1, 1)
                log_prob = dist.log_prob(locs).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)
            else:
                locs = mean_loc
                log_prob = torch.zeros(B, device=device)
                entropy = torch.zeros(B, device=device)
                
            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(self.baseline(hx).squeeze(-1))
            
        logits = self.classifier(hx)
        
        return logits, torch.stack(log_probs, dim=1), torch.stack(values, dim=1), torch.stack(entropies, dim=1)

def compute_reinforce_loss(log_probs, values, rewards):
    advantages = rewards.unsqueeze(1) - values.detach()
    policy_loss = -(log_probs * advantages).sum(dim=1).mean()
    value_loss = F.mse_loss(values, rewards.unsqueeze(1).expand_as(values))
    return policy_loss + 0.5 * value_loss

def compute_ppo_loss(log_probs, old_log_probs, values, rewards, clip_epsilon=0.2):
    advantages = rewards.unsqueeze(1) - values.detach()
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).sum(dim=1).mean()
    value_loss = F.mse_loss(values, rewards.unsqueeze(1).expand_as(values))
    return policy_loss + 0.5 * value_loss
