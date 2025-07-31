import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("./wrapper/bilateralfilter/build/lib.linux-x86_64-3.6")

def get_seg_loss(pred, label, ignore_index=255):
    
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

    return (bg_loss + fg_loss) * 0.5

# VICReg Loss
class VICRegLoss(nn.Module):
    def __init__(self, lmbda=25.0, mu=25.0, nu=1.0, gamma=1.0, eps=1e-4):
        super().__init__()
        self.lmbda = lmbda
        self.mu = mu
        self.nu = nu
        self.gamma = gamma
        self.eps = eps
        
    def forward(self, z1, z2):
        batch_size, D = z1.shape

        invariance_loss = F.mse_loss(z1, z2)

        std_z1 = torch.sqrt(z1.var(dim=0) + self.eps)
        std_z2 = torch.sqrt(z2.var(dim=0) + self.eps)
        variance_loss = 0.5 * (torch.mean(F.relu(self.gamma - std_z1)) + torch.mean(F.relu(self.gamma - std_z2)))

        z1_centered = z1 - z1.mean(dim=0)
        cov_z1 = (z1_centered.T @ z1_centered) / (batch_size - 1)
        
        z2_centered = z2 - z2.mean(dim=0)
        cov_z2 = (z2_centered.T @ z2_centered) / (batch_size - 1)
        
        off_diag_mask = ~torch.eye(D, dtype=torch.bool, device=z1.device)

        cov_loss_z1 = cov_z1[off_diag_mask].pow(2).sum() / D
        cov_loss_z2 = cov_z2[off_diag_mask].pow(2).sum() / D
        covariance_loss = 0.5 * (cov_loss_z1 + cov_loss_z2)

        weighted_loss = self.lmbda * invariance_loss + self.mu * variance_loss + self.nu * covariance_loss

        return weighted_loss, invariance_loss, variance_loss, covariance_loss