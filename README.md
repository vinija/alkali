# 🛡️ AVQI + GRACE

**Adversarial Vulnerability Quality Index (AVQI)** + **GRACE (Geometric Representation-Aware Contrastive Enhancement)**  
_A EMNLP 2025 Submission – Towards Robust Safety Alignment in LLMs_

---

## 🔍 Overview

This repository implements a robust adversarial safety alignment framework for LLMs based on:
- **GRACE**: A contrastive, geometry-aware fine-tuning procedure operating in the **latent space**
- **AVQI**: A principled metric for **intrinsic vulnerability diagnosis** using latent cluster geometry

The pipeline is benchmarked using:
- **ALKALI Benchmark**: 9,000 adversarial prompts, across 3 macro and 15 fine-grained attack families
- **21 Open and Proprietary LLMs** tested under unified safety evaluation

---

## 🌐 Architecture

```text
                    ┌────────────┐
                    │  ALKALI    │───┐
                    └────────────┘   │
                                    ▼
   ┌────────────┐    prompt + adv completions   ┌────────────┐
   │ Frozen LLM │──────────────────────────────▶│ Pooler     │
   └────────────┘   output_hidden_states        └────────────┘
                                            ▲             │
                                            │             ▼
                                        ┌────────┐   ┌────────────┐
                                        │ Policy │   │ Ref Policy │
                                        └────────┘   └────────────┘
                                             ▲         ▲
                                             │         │
                                     ┌────────────────────────┐
                                     │    GRACE Loss Module   │
                                     └────────────────────────┘

                                    (Contrastive preference + merge + separation + KL smoothing)
![image](https://github.com/user-attachments/assets/09108d84-a118-49f6-8b4e-6e9f96de0bc1)
