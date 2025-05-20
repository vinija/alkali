eval_pipeline/runner.py
----------------------------------
import os
import json
import torch
import numpy as np
from typing import Dict, List
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

from avqi_metric.avqi import AVQI
from latent_pooling.attention_pooler import MultiHeadLayerwiseAttentionPooler


class GRACEEvaluator:
    """
    Full evaluation runner for AVQI + GRACE framework.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 tokenizer,
                 pooler: MultiHeadLayerwiseAttentionPooler,
                 dataset,
                 refusal_fn,
                 attack_families: List[str],
                 device: str = "cuda"):

        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.pooler = pooler.eval().to(device)
        self.dataset = dataset
        self.refusal_fn = refusal_fn
        self.families = attack_families
        self.device = device

    def _pooled(self, prompt: str, output: str) -> torch.Tensor:
        text = prompt.strip() + " " + output.strip()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = [h.squeeze(0) for h in out.hidden_states]
        return self.pooler(hidden_states)

    def evaluate(self, save_path: str = "results/"):
        os.makedirs(save_path, exist_ok=True)
        cluster_vectors = defaultdict(list)
        per_attack_asr = defaultdict(lambda: [0, 0])  # hits, total
        refusal_failures = []

        print(f"[*] Running evaluation on {len(self.dataset)} samples...")
        for sample in tqdm(self.dataset):
            prompt = sample["prompt"]
            y_s = sample["safe"]
            y_u = sample["unsafe"]
            y_j = sample["jailbreak"]
            family = sample.get("attack_family", "unknown")

            h_s = self._pooled(prompt, y_s)
            h_u = self._pooled(prompt, y_u)
            h_j = self._pooled(prompt, y_j)

            cluster_vectors["safe"].append(h_s)
            cluster_vectors["unsafe"].append(h_u)
            cluster_vectors["jailbreak"].append(h_j)

            # Evaluate refusal on jailbreak
            generated = y_j.strip()
            if not self.refusal_fn(prompt + " " + generated):
                per_attack_asr[family][0] += 1
                refusal_failures.append((prompt, generated))
            per_attack_asr[family][1] += 1

        # Compute AVQI
        avqi = AVQI()
        avqi.add_cluster("safe", cluster_vectors["safe"])
        avqi.add_cluster("unsafe", cluster_vectors["unsafe"])
        avqi.add_cluster("jailbreak", cluster_vectors["jailbreak"])
        score = avqi.compute_raw_avqi()
        decomp = avqi.compute_decomposition()

        # ASR per attack family
        attack_asr = {
            fam: {
                "ASR": round(hits / total * 100, 2),
                "hits": hits,
                "total": total
            } for fam, (hits, total) in per_attack_asr.items()
        }

        # Centroid shift
        c_safe = avqi.centroids["safe"]
        centroid_norm = c_safe.norm().item()

        result = {
            "AVQI_score": round(score, 4),
            "Centroid_L2_safe": round(centroid_norm, 4),
            "AVQI_decomposition": decomp,
            "Attack_ASR": attack_asr,
            "Failure_examples": refusal_failures[:10]
        }

        # Save to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"eval_{timestamp}.json"
        with open(os.path.join(save_path, fname), 'w') as f:
            json.dump(result, f, indent=2)

        print(f"[✓] Evaluation complete. AVQI: {score:.4f} | Results saved to {fname}")
        return result
