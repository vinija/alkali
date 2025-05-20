import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class GRACELoss(nn.Module):
    """
    GRACE training loss with full NeurIPS rigor.
    Components:
    - Latent-space Preference Loss (contrastive DPO)
    - Safe–Adversarial Separation Loss
    - Unsafe–Jailbreak Merge Loss
    - Adversarial Smoothing (KL between logits)
    """

    def __init__(self,
                 margin_separation: float = 5.0,
                 margin_merge: float = 1.0,
                 lambda_sep: float = 1.0,
                 lambda_merge: float = 1.0,
                 lambda_smooth: float = 0.1,
                 beta: float = 0.8,
                 dropout_rate: float = 0.15,
                 use_hard_negatives: bool = True):
        super().__init__()
        self.margin_separation = margin_separation
        self.margin_merge = margin_merge
        self.lambda_sep = lambda_sep
        self.lambda_merge = lambda_merge
        self.lambda_smooth = lambda_smooth
        self.beta = beta
        self.dropout = nn.Dropout(p=dropout_rate)
        self.use_hard_negatives = use_hard_negatives

    def _logits(self, model_head: nn.Module, pooled: torch.Tensor) -> torch.Tensor:
        return model_head(self.dropout(pooled)).squeeze(-1)  # [B]

    def preference_loss(self,
                        h_safe: torch.Tensor,
                        h_unsafe: torch.Tensor,
                        policy_head: nn.Module,
                        ref_head: nn.Module,
                        prompt_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Relaxed β-scaled DPO contrastive loss.
        """
        logp_s = self._logits(policy_head, h_safe)
        logp_u = self._logits(policy_head, h_unsafe)

        with torch.no_grad():
            logp_s_ref = self._logits(ref_head, h_safe)
            logp_u_ref = self._logits(ref_head, h_unsafe)

        margin = logp_s - logp_u - self.beta * (logp_s_ref - logp_u_ref)
        pref_loss = -F.logsigmoid(margin).mean()

        # Optional: use smooth L1 loss as probing loss
        probing = F.smooth_l1_loss(logp_s, logp_s_ref) + F.smooth_l1_loss(logp_u, logp_u_ref)
        return pref_loss, probing

    def separation_loss(self, h_safe: torch.Tensor,
                        h_unsafe: torch.Tensor,
                        h_jb: torch.Tensor) -> torch.Tensor:
        """
        Enforces geometric distance between safe and unsafe/jailbreak clusters.
        """
        dist_su = F.pairwise_distance(h_safe, h_unsafe, p=2)
        dist_sj = F.pairwise_distance(h_safe, h_jb, p=2)

        loss_su = F.relu(self.margin_separation - dist_su).mean()
        loss_sj = F.relu(self.margin_separation - dist_sj).mean()

        return (loss_su + loss_sj) / 2

    def merge_loss(self, h_unsafe: torch.Tensor,
                   h_jb: torch.Tensor) -> torch.Tensor:
        """
        Pulls unsafe and jailbreak samples into a tight latent neighborhood.
        Uses Gaussian collapse with optional pairwise smoothing.
        """
        dist = F.pairwise_distance(h_unsafe, h_jb, p=2)
        return F.relu(dist - self.margin_merge).mean()

    def adversarial_smoothing_loss(self,
                                   h_unsafe: torch.Tensor,
                                   h_jb: torch.Tensor,
                                   policy_head: nn.Module) -> torch.Tensor:
        """
        Encourage similar logit distributions over unsafe/jailbreak pairs.
        """
        logit_u = self._logits(policy_head, h_unsafe)
        logit_j = self._logits(policy_head, h_jb)

        p = F.softmax(logit_u, dim=-1)
        q = F.softmax(logit_j, dim=-1)
        return F.kl_div(q.log(), p, reduction='batchmean')

    def forward(self,
                pooled: Dict[str, torch.Tensor],
                policy_head: nn.Module,
                ref_head: nn.Module,
                prompt_input: torch.Tensor) -> Dict[str, float]:
        """
        Main GRACE objective.
        pooled: dict with keys [safe, unsafe, jailbreak]
        """
        h_s = pooled["safe"]
        h_u = pooled["unsafe"]
        h_j = pooled["jailbreak"]

        L_pref, L_probe = self.preference_loss(h_s, h_u, policy_head, ref_head, prompt_input)
        L_sep = self.separation_loss(h_s, h_u, h_j)
        L_merge = self.merge_loss(h_u, h_j)
        L_smooth = self.adversarial_smoothing_loss(h_u, h_j, policy_head)

        L_total = (L_pref
                   + self.lambda_sep * L_sep
                   + self.lambda_merge * L_merge
                   + self.lambda_smooth * L_smooth)

        return {
            "total": L_total,
            "pref": L_pref.item(),
            "sep": L_sep.item(),
            "merge": L_merge.item(),
            "smooth": L_smooth.item(),
            "probe": L_probe.item()
        }


