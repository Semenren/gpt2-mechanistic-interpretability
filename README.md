# 🔍 GPT-2 Mechanistic Interpretability Sandbox


[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![Tool: TransformerLens](https://img.shields.io/badge/Tools-TransformerLens-orange.svg)](https://github.com/transformerlensorg/TransformerLens)

An experimental research repository focused on **Mechanistic Interpretability**, causal tracing, and representation steering within autoregressive transformer architectures. The primary objective is to reverse-engineer information routing mechanisms, map induction sub-circuits, and design target activation interventions to suppress undesirable model tendencies (e.g., sycophancy).

---

## 🧬 Core Focus: Induction Circuits Mapping

Induction circuits are structural sub-graphs within transformers responsible for in-context learning and sequence replication patterns ($[A][B] \dots [A] \rightarrow [B]$). This sandbox isolates and reverse-engineers these mechanisms inside `gpt2-small` (12-layer, 117M parameters).

### Key Methodologies Implemented:
1. **Activation Patching (Causal Tracing):** Systematically replacing internal activations (hidden states, MLP outputs, or attention head outputs) of a model running a corrupted prompt with saved activations from a clean prompt to isolate where the causal compute happens.
2. **Residual Stream Intervention:** Intercepting vectors within the residual stream to track how information updates propagate from layer to layer.
3. **Component Ablation (Zero/Mean):** Dynamically modifying the model's forward pass by zeroing out specific attention heads (`L6H9`, `L10H1`) to observe their direct impact on the logit difference between target tokens and alternative options.

---

## ⚡ Key Findings & Evaluation

Through causal localization, specific behavior circuits were identified and evaluated:

* **Logit Difference Drift:** Zero-ablation of mid-level attention heads (`Layer 6, Head 9` and `Layer 10, Head 1`) resulted in a significant drop in target token logit sequence predictability (e.g., dropping clean Logit Difference from **6.48** down to **6.13** under controlled distributions).
* **Sycophancy Mitigation:** Implemented a real-time vector hook inside late-stage MLP layers. By calculating the projection of hidden states onto the user-praise alignment subspace, the system applies an inverse steering vector to suppress sycophantic generation patterns without destabilizing baseline perplexity.

---

## 📐 Mathematical Metrics

To quantify the strength of internal circuits, the evaluation engine measures:

* **Logit Attenuation Index ($LAI$):**
  $$LAI = \text{LogitDiff}_{\text{clean}} - \text{LogitDiff}_{\text{ablated}}$$
* **Activation Projection Vector ($V_{\text{steer}}$):**
  $$V_{\text{steer}} = h_l - \alpha \cdot \frac{h_l \cdot \vec{d}}{\|\vec{d}\|^2} \vec{d}$$
  *Where $h_l$ is the hidden state at layer $l$, $\vec{d}$ is the isolated sycophancy direction matrix, and $\alpha$ is the structural steering coefficient.*

---

## 📦 Repository Structure

```text
├── notebooks/
│   └── gpt2_induction_circuits.ipynb 
├── .gitignore
├── LICENSE
└── README.md    
