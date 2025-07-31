import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyCalculator(nn.Module):
    def __init__(self, in_channels, num_classes, d=2048):
        super().__init__()
        self.d = d
        self.in_channels = in_channels
        self.num_classes = num_classes

        projection_matrix = torch.randn(self.d, self.in_channels)
        self.register_buffer('projection_matrix', projection_matrix)

    def _project_to_hypervectors(self, features):
        z_pooled = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
        
        hypervectors = torch.einsum('bc,dc->bdc', z_pooled, self.projection_matrix)
        return hypervectors

    def _compute_class_prototypes(self, hypervectors, labels):
        device = hypervectors.device
        batch_prototypes = torch.zeros(self.num_classes, self.d, self.in_channels, device=device)

        for class_idx in range(self.num_classes):
            mask = labels[:, class_idx].bool()
            
            if mask.any():
                bundle = torch.sum(hypervectors[mask], dim=0)
                
                batch_prototypes[class_idx] = F.normalize(bundle, p=2, dim=0)
        
        return batch_prototypes

    def forward(self, features, labels):
        hypervectors = self._project_to_hypervectors(features)
        
        prototypes = self._compute_class_prototypes(hypervectors, labels)

        hypervectors_expanded = hypervectors.unsqueeze(1)
        prototypes_expanded = prototypes.unsqueeze(0)

        sim = F.cosine_similarity(hypervectors_expanded, prototypes_expanded, dim=2)
        
        gt_mask = (labels == 0)
        expanded_mask = gt_mask.unsqueeze(-1).expand_as(sim)
        masked_sim = torch.where(expanded_mask, torch.tensor(-1.0, device=sim.device), sim)
        
        max_sim_per_channel, _ = torch.max(masked_sim, dim=1)
        
        uncertainty_score = 1.0 - torch.mean(max_sim_per_channel, dim=1)
        return uncertainty_score