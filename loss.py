import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicEnsembleLossProgress(nn.Module):
    def __init__(self, 
                 criterion=nn.CrossEntropyLoss(), 
                 max_alpha=0.3, 
                 max_beta=0.1,
                 max_loss_estimate=3.0,
                 use_area_loss=True,
                 warmup_epochs=0,
                 transition_epochs=5,
                 loss_global_based=False):
        super().__init__()
        self.criterion = criterion
        self.max_alpha = max_alpha
        self.max_beta = max_beta
        self.max_loss_estimate = max_loss_estimate
        self.use_area_loss = use_area_loss
        self.warmup_epochs = warmup_epochs
        self.transition_epochs = transition_epochs
        self.epoch = 0
        self.loss_global_based = loss_global_based

    def step_epoch(self):
        self.epoch += 1

    def forward(self, outputs, labels):
        # Get base loss
        loss_fused = self.criterion(outputs["out"], labels)
        loss_global = self.criterion(outputs["distill_teacher"], labels) if "distill_teacher" in outputs else torch.tensor(0.0, device=labels.device)

        # Dynamic fusion factor (used to adjust alpha/beta)
        with torch.no_grad():
            loss_factor = 1.0 - min(loss_fused.item() / self.max_loss_estimate, 1.0)

        # Smooth transition progress ∈ [0, 1]
        progress = min(1.0, max(0.0, (self.epoch - self.warmup_epochs) / self.transition_epochs))

        # Fusion ratio: from 0.2 -> 1.0, global from 0.8 -> 0
        fused_weight = 0.2 + 0.8 * progress
        global_weight = 1.0 - fused_weight

        # Smooth adjustment of alpha/beta/gamma
        alpha = self.max_alpha * loss_factor * progress
        beta = self.max_beta * loss_factor * progress if self.use_area_loss else 0.0
        gamma = 0.1 * progress  # can be set as a hyperparameter

        # Weighted combination of base losses
        loss_total = fused_weight * loss_fused + global_weight * loss_global

        loss_area = torch.tensor(0.0, device=labels.device)
        loss_area_distill = torch.tensor(0.0, device=labels.device)

        loss_entropy = torch.tensor(0.0, device=labels.device)
        loss_diversity = torch.tensor(0.0, device=labels.device)

        if progress > 0.0:
            # Area loss + distillation loss
            if self.use_area_loss and "area_logits" in outputs and outputs["area_logits"] is not None:
                area_losses = [
                    self.criterion(area_out, labels)
                    for area_out in outputs["area_logits"]
                ]
                loss_area = sum(area_losses) / len(area_losses)

                # Distillation loss (KL)
                T = 4.0
                if "distill_teacher" in outputs and outputs["distill_teacher"] is not None: 
                    loss_area_distill = sum([
                        F.kl_div(
                            F.log_softmax(area_out / T, dim=1),
                            F.softmax(outputs["distill_teacher"] / T, dim=1),
                            reduction="batchmean"
                        ) * (T ** 2)
                        for area_out in outputs["area_logits"]
                    ]) / len(outputs["area_logits"])
               

                # === Multi-objective combination
                loss_total += beta * loss_area
                loss_total += gamma * loss_area_distill
                loss_total += 0.01 * loss_diversity    
                loss_total += 0.01 * loss_entropy      
                

                loss_total += beta * loss_area + gamma * loss_area_distill

        return {
            "loss": loss_total,
            "loss_fused": loss_fused.detach(),
            "loss_global": loss_global.detach(),
            "loss_area": loss_area.detach(),
            "loss_area_distill": loss_area_distill.detach(),

            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "fused_weight": fused_weight,
            "progress": progress
        }

class DynamicEnsembleLoss_No_Dis(nn.Module):
    def __init__(self, 
                 criterion=nn.CrossEntropyLoss(), 
                 max_alpha=0.3, 
                 max_beta=0.1,
                 max_loss_estimate=3.0,
                 use_area_loss=True,
                 warmup_epochs=0,
                 transition_epochs=5,
                 loss_global_based=False):
        super().__init__()
        self.criterion = criterion
        self.max_alpha = max_alpha
        self.max_beta = max_beta
        self.max_loss_estimate = max_loss_estimate
        self.use_area_loss = use_area_loss
        self.warmup_epochs = warmup_epochs
        self.transition_epochs = transition_epochs
        self.epoch = 0
        self.loss_global_based = loss_global_based

    def step_epoch(self):
        self.epoch += 1

    def forward(self, outputs, labels):
        # Get base loss
        loss_fused = self.criterion(outputs["out"], labels)
        loss_global = self.criterion(outputs["distill_teacher"], labels) if "distill_teacher" in outputs else torch.tensor(0.0, device=labels.device)

        # Dynamic fusion factor (used to adjust alpha/beta)
        with torch.no_grad():
            loss_factor = 1.0 - min(loss_fused.item() / self.max_loss_estimate, 1.0)

        # Smooth transition progress ∈ [0, 1]
        progress = min(1.0, max(0.0, (self.epoch - self.warmup_epochs) / self.transition_epochs))

        # Fusion ratio: from 0.2 -> 1.0, global from 0.8 -> 0
        fused_weight = 0.2 + 0.8 * progress
        global_weight = 1.0 - fused_weight

        # Smooth adjustment of alpha/beta/gamma
        alpha = self.max_alpha * loss_factor * progress
        beta = self.max_beta * loss_factor * progress if self.use_area_loss else 0.0
        gamma = 0.1 * progress  # can be set as a hyperparameter

        # Weighted combination of base losses
        loss_total = fused_weight * loss_fused + global_weight * loss_global

        loss_area = torch.tensor(0.0, device=labels.device)
        loss_area_distill = torch.tensor(0.0, device=labels.device)

        if progress > 0.0:
            # Area loss
            if self.use_area_loss and "area_logits" in outputs and outputs["area_logits"] is not None:
                area_losses = [
                    self.criterion(area_out, labels)
                    for area_out in outputs["area_logits"]
                ]
                loss_area = sum(area_losses) / len(area_losses)

                loss_total += beta * loss_area 

        return {
            "loss": loss_total,
            "loss_fused": loss_fused.detach(),
            "loss_global": loss_global.detach(),
            "loss_area": loss_area.detach(),
            "loss_area_distill": loss_area_distill.detach(),
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "fused_weight": fused_weight,
            "progress": progress
        }

class DynamicEnsembleLoss_No_Warmup(nn.Module):
    def __init__(self, 
                 criterion=nn.CrossEntropyLoss(), 
                 max_alpha=0.3, 
                 max_beta=0.1,
                 max_loss_estimate=3.0,   # Used to normalize loss (can be set empirically as initial loss)
                 use_area_loss=True,
                 warmup_epochs=0,
                 loss_global_based=True):
        super().__init__()
        self.criterion = criterion
        self.max_alpha = max_alpha
        self.max_beta = max_beta
        self.max_loss_estimate = max_loss_estimate
        self.use_area_loss = use_area_loss
        self.warmup_epochs = warmup_epochs
        self.epoch = 0
        self.loss_global_based = loss_global_based
        
    def step_epoch(self):
        self.epoch += 1

    def forward(self, outputs, labels):
        if self.loss_global_based:
            raise NotImplementedError("loss_global_based is not implemented yet.")
        else:
            """Warm up guided by Global and little fusion"""

            loss_global = self.criterion(outputs["distill_teacher"], labels)
            loss_fused = self.criterion(outputs["out"], labels)

            loss_factor = 1.0 - min(loss_fused.item() / self.max_loss_estimate, 1.0)
            alpha = self.max_alpha * loss_factor
            beta = self.max_beta * loss_factor if self.use_area_loss else 0.0

            loss_total = 0.2*loss_fused + 0.8*loss_global

            if self.epoch >= self.warmup_epochs:
                gamma = 0.1 # weight for distillation loss
                T = 4.0 # softmax temperature
                loss_total = loss_fused

                if "distill_teacher" in outputs and outputs["distill_teacher"] is not None:
                    loss_global = self.criterion(outputs["distill_teacher"], labels)
                else:
                    loss_global = torch.tensor(0.0, device=labels.device)
                loss_total += alpha * loss_global
                
                if self.use_area_loss and "area_logits" in outputs and outputs["area_logits"] is not None:
                    area_losses = [
                        self.criterion(area_out, labels)
                        for area_out in outputs["area_logits"]
                    ]
                    loss_area = sum(area_losses) / len(area_losses)

                    area_losses_distill = [
                        nn.KLDivLoss(reduction='batchmean')(
                            F.log_softmax(area_out / T, dim=1),
                            F.softmax(outputs["distill_teacher"] / T, dim=1)
                        ) * (T ** 2)
                        for area_out in outputs["area_logits"]
                    ]
                    loss_area_distill = sum(area_losses_distill) / len(area_losses_distill)

                    loss_total += beta * loss_area + gamma * loss_area_distill

                else:
                    loss_area = torch.tensor(0.0, device=labels.device)
                    loss_area_distill = torch.tensor(0.0, device=labels.device)

            else:
                loss_global = torch.tensor(0.0, device=labels.device)
                loss_area = torch.tensor(0.0, device=labels.device)
                loss_area_distill = torch.tensor(0.0, device=labels.device)
    
        return {
            "loss": loss_total,
            "loss_fused": loss_fused.detach(),
            "loss_global": loss_global.detach(),
            "loss_area": loss_area.detach(),
            "alpha": alpha,
            "beta": beta
        }

class DynamicEnsembleLossProgress_Entropy(nn.Module):
    def __init__(self, 
                 criterion=nn.CrossEntropyLoss(),
                 max_alpha=0.3, 
                 max_beta=0.1,
                 max_loss_estimate=3.0,
                 use_area_loss=True,
                 warmup_epochs=0,
                 transition_epochs=5,
                 loss_global_based=False):
        super().__init__()
        self.criterion = criterion
        self.max_alpha = max_alpha
        self.max_beta = max_beta
        self.max_loss_estimate = max_loss_estimate
        self.use_area_loss = use_area_loss
        self.warmup_epochs = warmup_epochs
        self.transition_epochs = transition_epochs
        self.epoch = 0
        self.loss_global_based = loss_global_based

    def step_epoch(self):
        self.epoch += 1

    def forward(self, outputs, labels):
        # Get base loss
        loss_fused = self.criterion(outputs["out"], labels)
        loss_global = self.criterion(outputs["distill_teacher"], labels) if "distill_teacher" in outputs else torch.tensor(0.0, device=labels.device)

        # Dynamic fusion factor (used to adjust alpha/beta)
        with torch.no_grad():
            loss_factor = 1.0 - min(loss_fused.item() / self.max_loss_estimate, 1.0)

        # Smooth transition progress ∈ [0, 1]
        progress = min(1.0, max(0.0, (self.epoch - self.warmup_epochs) / self.transition_epochs))

        # Fusion
        fused_weight = 0.5 + 0.5 * progress
        global_weight = 1.0 - fused_weight

        # Weighted combination of base losses
        loss_total = fused_weight * loss_fused + global_weight * loss_global

        # Smooth adjustment of alpha/beta/gamma
        # alpha = self.max_alpha * loss_factor * progress if self.use_area_loss else 0.0
        alpha = 0
        beta = self.max_beta * loss_factor * progress if self.use_area_loss else 0.0
        gamma = 0

        # loss_area = torch.tensor(0.0, device=labels.device)
        loss_entropy = torch.tensor(0.0, device=labels.device)
        loss_diversity = torch.tensor(0.0, device=labels.device)

        if progress > 0.0:
            # Area loss 
            if self.use_area_loss and "area_logits" in outputs and outputs["area_logits"] is not None:

                # Entropy loss (encourage softmax fusion diversity)
                def compute_entropy(weights: torch.Tensor) -> torch.Tensor:
                    epsilon = 1e-8
                    norm_weights = weights / (weights.sum() + epsilon)
                    entropy = -torch.sum(norm_weights * torch.log2(norm_weights + epsilon))
                    return entropy

                # Try to get ensemble_weights from outputs, otherwise skip entropy loss
                if "branch_weights_tensor" in outputs and outputs["branch_weights_tensor"] is not None:
                    ensemble_weights = outputs["branch_weights_tensor"]  # Already detached
                    soft_weights = F.softmax(ensemble_weights, dim=0)
                    loss_entropy = compute_entropy(soft_weights)
                else:
                    loss_entropy = torch.tensor(0.0, device=labels.device)

                # === Add multi-objective combination
                # loss_total += alpha * loss_diversity    
                loss_total += beta * loss_entropy  
                    
        return {
            "loss": loss_total,
            "loss_fused": loss_fused.detach(),
            "loss_global": loss_global.detach(),
            "loss_entropy": loss_entropy.detach(),
            "loss_diversity": loss_diversity.detach(),

            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "fused_weight": fused_weight,
            
            "progress": progress
        }
