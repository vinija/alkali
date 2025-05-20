import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable, Dict, List

from grace_trainer.loss import GRACELoss
from latent_pooling.attention_pooler import MultiHeadLayerwiseAttentionPooler


class GRACETrainer:
    """
    NeurIPS-grade GRACE trainer for adversarial safety alignment.
    """

    def __init__(self,
                 model: nn.Module,
                 tokenizer,
                 dataset,
                 hidden_dim: int,
                 num_layers: int,
                 batch_size: int = 8,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-5,
                 device: str = "cuda",
                 log_every: int = 10,
                 save_dir: str = "./checkpoints"):

        self.device = device
        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.batch_size = batch_size
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.pooler = MultiHeadLayerwiseAttentionPooler(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=4,
            pooling_type="mean",
            temperature=1.0,
            residual_fusion=True
        ).to(device)

        self.policy_head = nn.Linear(hidden_dim, 1).to(device)
        self.ref_head = nn.Linear(hidden_dim, 1).to(device)

        self.loss_fn = GRACELoss()
        self.optimizer = optim.AdamW(
            list(self.pooler.parameters()) + list(self.policy_head.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.log_every = log_every

    def _get_hidden(self, prompt: str, completion: str) -> List[torch.Tensor]:
        input_text = prompt.strip() + " " + completion.strip()
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        return [layer.squeeze(0) for layer in outputs.hidden_states]  # [L, S, D]

    def _get_pooled(self, prompt: str, output: str) -> torch.Tensor:
        layers = self._get_hidden(prompt, output)
        return self.pooler(layers)  # [D]

    def _embed_batch(self, batch: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
        pooled = {"safe": [], "unsafe": [], "jailbreak": []}
        prompts = []

        for p, s, u, j in zip(batch["prompt"], batch["safe"], batch["unsafe"], batch["jailbreak"]):
            pooled["safe"].append(self._get_pooled(p, s))
            pooled["unsafe"].append(self._get_pooled(p, u))
            pooled["jailbreak"].append(self._get_pooled(p, j))
            prompts.append(p)

        for k in pooled:
            pooled[k] = torch.stack(pooled[k])
        prompt_toks = torch.stack([self.tokenizer(p, return_tensors="pt").input_ids.squeeze(0).to(self.device)
                                   for p in prompts])
        return pooled, prompt_toks

    def train(self, epochs: int = 3):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(epochs):
            self.pooler.train()
            self.policy_head.train()
            self.ref_head.eval()  # frozen

            losses_epoch = []

            for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
                pooled, prompt_input = self._embed_batch(batch)

                loss_vals = self.loss_fn(pooled, self.policy_head, self.ref_head, prompt_input)
                loss = loss_vals["total"]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if step % self.log_every == 0:
                    msg = (f"[Ep {epoch} | Step {step}] "
                           f"Loss: {loss.item():.4f} | "
                           f"Pref: {loss_vals['pref']:.3f}, Sep: {loss_vals['sep']:.3f}, "
                           f"Merge: {loss_vals['merge']:.3f}, Smooth: {loss_vals['smooth']:.3f}")
                    print(msg)

                losses_epoch.append(loss.item())

            self.scheduler.step()

            # Save checkpoint
            torch.save(self.pooler.state_dict(), os.path.join(self.save_dir, f"pooler_ep{epoch}.pt"))
            torch.save(self.policy_head.state_dict(), os.path.join(self.save_dir, f"policy_ep{epoch}.pt"))

            # Drift tracking: mean-shift in safe cluster center
            safe_centers = pooled["safe"].mean(dim=0)
            print(f"[Epoch {epoch}] Safe latent center L2-norm: {safe_centers.norm():.4f}")

    def save(self, path: str):
        torch.save({
            "pooler": self.pooler.state_dict(),
            "policy": self.policy_head.state_dict()
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.pooler.load_state_dict(checkpoint["pooler"])
        self.policy_head.load_state_dict(checkpoint["policy"])
