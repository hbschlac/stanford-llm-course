# Lecture 9: Current Trends

**Course:** Stanford CME 295 -- Transformers & Large Language Models
**Instructors:** Afshine Amidi & Shervine Amidi
**Date:** December 5, 2025
**YouTube:** [https://www.youtube.com/watch?v=Q86qzJ1K1Ss](https://www.youtube.com/watch?v=Q86qzJ1K1Ss)
**Slides:** [https://cme295.stanford.edu/slides/fall25-cme295-lecture9.pdf](https://cme295.stanford.edu/slides/fall25-cme295-lecture9.pdf)

## Topics Covered

1. **Recap** -- Course recap of Lectures 1-8
2. **Beyond Transformer-based LLMs** -- Alternative architectures to the Transformer
3. **Diffusion LLMs** -- Applying diffusion processes to language modeling
4. **Closing Thoughts**

---

## Part 1: Course Recap -- "Rewinding the Quarter..."

### Lecture 1: Transformers (Sep 26)

**Core pipeline reviewed:**
- Sentence: `"A cute teddy bear is reading."`
- Step 1 -- Tokenization: split into tokens `[a] [cute] [teddy bear] [is] [reading] [.]`
- Step 2 -- Embeddings: tokens mapped to dense vectors in a continuous space (word embedding space illustrated with "teddy bear", "soft", "book" as vectors in 3D)

**Self-Attention Mechanism:**
- Each token gets three learned projections:
  - **Query** (q): what this token is looking for
  - **Key** (k): what this token offers / advertises
  - **Value** (v): the actual content carried by this token
- For a given token (e.g., "teddy bear"), its query q_teddy_bear attends to all keys k_a^T, k_cute^T, k_teddy_bear^T, k_is^T, k_reading^T, k_.^T to compute attention weights

**Scaled Dot-Product Attention Formula:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- Q = query matrix, K = key matrix, V = value matrix
- d_k = dimensionality of the key vectors (scaling factor prevents dot products from growing too large)

**Transformer Architecture (Vaswani et al., 2017 -- "Attention Is All You Need"):**
- **Encoder** (left side): Input Embedding -> Positional Encoding -> N x [Multi-Head Attention -> Add & Norm -> Feed Forward -> Add & Norm]
- **Decoder** (right side): Output Embedding -> Positional Encoding -> N x [Masked Multi-Head Attention -> Add & Norm -> Multi-Head Attention (cross-attention with encoder) -> Add & Norm -> Feed Forward -> Add & Norm] -> Linear -> Softmax -> Output Probabilities

**Key references:**
- "Attention Is All You Need", Vaswani et al., 2017
- "Super Study Guide: Transformers & Large Language Models", Amidi, 2024

---

### Lecture 2: Transformer-Based Models & Tricks (Oct 3)

**Rotary Position Embeddings (RoPE) (Su et al., 2021):**
- Encodes position information by rotating query and key vectors in 2D subspaces
- For position m, the rotation matrix is:

$$R_{\theta,m} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}$$

- Query at position m becomes: q_m * R_theta,m^T
- Key at position n becomes: k_n * R_theta,n^T
- The dot product q_m^T * k_n depends only on the relative position (m - n), which is the desired property

**Grouped-Query Attention (GQA) (Ainslie et al., 2023):**
- Intermediate approach between Multi-Head Attention (MHA) and Multi-Query Attention (MQA)
- G groups of key-value heads, each shared by h/G query heads
- Layout: Q_1 ... Q_{h/G} share K_1, V_1; ... ; Q_{h-h/G+1} ... Q_h share K_G, V_G
- Reduces KV cache size while preserving most of MHA quality

**Encoder-only vs Decoder-only architectures:**
- Modern LLMs predominantly use decoder-only architecture (encoder side crossed out in slides)
- Encoder-only models (like BERT) used primarily for classification/understanding tasks

**Key references:**
- "RoFormer: Enhanced Transformer with Rotary Position Embeddings", Su et al., 2021
- "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", Ainslie et al., 2023

---

### Lecture 3: Large Language Models (Oct 10)

**Mixture of Experts (MoE) (Shazeer et al., 2017):**
- Architecture: Input X -> Gating network G -> selects top-k experts from {FFNN_1, FFNN_2, ..., FFNN_n}
- Gate assigns weights to each expert; only selected experts (green check) are activated; others (red X) are skipped
- Output y-hat is the weighted sum of selected expert outputs
- **Sparse activation**: only a subset of parameters are used per token, enabling larger total model capacity without proportional compute increase

**MoE Implementation (from Mixtral of Experts, Jiang et al., 2024):**
```python
class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate, moe_args):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.args = moe_args

    def forward(self, inputs: torch.Tensor):
        inputs_squashed = inputs.view(-1, inputs.shape[-1])
        gate_logits = self.gate(inputs_squashed)
        weights, selected_experts = torch.topk(
            gate_logits, self.args.num_experts_per_tok
        )
        weights = nn.functional.softmax(
            weights,
            dim=1,
            dtype=torch.float,
        ).type_as(inputs)
        results = torch.zeros_like(inputs_squashed)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert] * expert(
                inputs_squashed[batch_idx]
            )
        return results.view_as(inputs)
```

**Next-token prediction:**
- LLMs model the conditional probability distribution P(w_{t+1} = w | C) over the vocabulary
- Given a context, the model outputs a probability distribution over all possible next tokens
- Illustrated with distribution showing probabilities for words like "a", "airplane", "fluffy", "gentle", "is", "kind", "my", "poet", "smart", "strong", "talkative", "where"

**Key references:**
- "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", Shazeer et al., 2017
- "Mixtral of Experts", Jiang et al., 2024

---

### Lecture 4: LLM Training (Oct 17)

**Scaling Laws (Kaplan et al., 2020):**
- Test loss follows power-law relationships with three axes:
  - **Compute** (PF-days, non-embedding): L = (C_min / 2.3 * 10^8)^{-0.050}
  - **Dataset Size** (tokens): L = (D / 5.4 * 10^{13})^{-0.095}
  - **Parameters** (non-embedding): L = (N / 8.8 * 10^{13})^{-0.076}
- All three show smooth power-law decay on log-log plots

**Chinchilla Optimal Training (Hoffmann et al., 2022 -- "Training Compute-Optimal Large Language Models"):**
- For a given compute budget, there is an optimal balance between model size and number of training tokens
- Chinchilla scaling table:

| Parameters | FLOPs | FLOPs (in Gopher unit) | Tokens |
|---|---|---|---|
| 400 Million | 1.92e+19 | 1/29,968 | 8.0 Billion |
| 1 Billion | 1.21e+20 | 1/4,761 | 20.2 Billion |
| 10 Billion | 1.23e+22 | 1/46 | 205.1 Billion |
| 67 Billion | 5.76e+23 | 1 | 1.5 Trillion |
| 175 Billion | 3.85e+24 | 6.7 | 3.7 Trillion |
| 280 Billion | 9.90e+24 | 17.2 | 5.9 Trillion |
| 520 Billion | 3.43e+25 | 59.5 | 11.0 Trillion |
| 1 Trillion | 1.27e+26 | 221.3 | 21.2 Trillion |
| 10 Trillion | 1.30e+28 | 22515.9 | 216.2 Trillion |

- Key insight: models should be trained on roughly 20x as many tokens as parameters for compute-optimal training

**Additional topics covered in Lecture 4 (from syllabus):**
- Pretraining procedures
- Quantization (reducing model precision)
- Hardware optimization strategies
- Supervised finetuning
- Parameter-efficient methods (LoRA -- Low-Rank Adaptation)

**Key references:**
- "Scaling Laws for Neural Language Models", Kaplan et al., 2020
- "Training Compute-Optimal Large Language Models", Hoffmann et al., 2022

---

### Lecture 5: LLM Tuning (Oct 31)

**Topics covered (from syllabus and course structure):**
- **Preference Tuning**: aligning LLM outputs with human preferences
- **RLHF (Reinforcement Learning from Human Feedback)**:
  - Step 1: Collect human preference data (pairwise comparisons of model outputs)
  - Step 2: Train a reward model to predict human preferences
  - Step 3: Optimize the LLM policy using RL (PPO) against the reward model
- **Reward Modeling**: training a model to score outputs based on human preferences
- **PPO (Proximal Policy Optimization)** and variants: the RL algorithm used to fine-tune the LLM
- **DPO (Direct Preference Optimization)**:
  - Bypasses the need for a separate reward model
  - Directly optimizes the policy using preference data
  - Loss function reformulates the reward implicitly through the policy ratio

---

### Lecture 6: LLM Reasoning (Nov 7)

**Topics covered (from syllabus):**
- **Reasoning Models**: LLMs designed or trained to exhibit reasoning capabilities
- **Reinforcement Learning for Reasoning**: using RL to improve reasoning ability
- **GRPO (Group Relative Policy Optimization)**: a reinforcement learning method for improving reasoning
- **Test-time Compute Scaling**: allocating more compute at inference time to improve reasoning
- Chain-of-thought and related prompting strategies for reasoning

---

### Lecture 7: Agentic LLMs (Nov 14)

**Topics covered (from syllabus):**
- **RAG (Retrieval-Augmented Generation)**: combining LLMs with external knowledge retrieval
- **Advanced RAG**: techniques to improve retrieval quality and integration
- **Function Calling / Tool Use**: LLMs that can invoke external tools and APIs
- **Agents**: autonomous LLM-based systems that plan and execute multi-step tasks
- **ReAct Framework**: Reasoning + Acting -- interleaving thought traces with actions

---

### Lecture 8: LLM Evaluation (Nov 21)

**Topics covered (from syllabus):**
- **LLM-as-a-Judge**: using one LLM to evaluate outputs of another
- Best practices for LLM evaluation
- Benefits and limitations of automated evaluation
- **Biases in LLM evaluation**: position bias, verbosity bias, self-preference bias
- Pitfalls and failure modes

---

## Part 2: Beyond Transformer-Based LLMs

> This section covers alternative architectures that challenge the Transformer's dominance, addressing its quadratic attention complexity.

### The Attention Bottleneck

- Standard self-attention has O(n^2) complexity with respect to sequence length n
- This becomes prohibitive for very long sequences
- Motivates research into sub-quadratic alternatives

### State Space Models (SSMs)

**Key idea:** Replace attention with linear recurrences inspired by continuous-time state space models from control theory.

**Continuous-time state space model:**

$$\dot{x}(t) = Ax(t) + Bu(t)$$
$$y(t) = Cx(t) + Du(t)$$

- A = state transition matrix, B = input projection, C = output projection, D = skip connection
- Discretized for sequence modeling using step size Delta

**S4 (Structured State Spaces for Sequence Modeling, Gu et al., 2022):**
- Parameterizes A as a structured matrix (HiPPO initialization) for long-range dependencies
- Achieves linear complexity O(n) in sequence length via convolutional view during training

**Mamba (Gu & Dao, 2023):**
- Makes SSM parameters (B, C, Delta) input-dependent (selective state spaces)
- Allows the model to selectively remember or forget information based on input content
- Hardware-aware implementation with efficient CUDA kernels
- No attention mechanism -- purely recurrent at inference, convolutional during training
- Achieves competitive performance with Transformers at smaller scales

### RWKV (Peng et al., 2023)

- "Receptance Weighted Key Value" -- combines advantages of RNNs and Transformers
- Linear attention variant that can be computed as both:
  - A Transformer-like parallel mode during training
  - An RNN-like sequential mode during inference
- O(n) complexity, constant memory during inference

### Hybrid Architectures

- Some recent models combine attention layers with SSM/linear layers
- Example: Jamba (AI21, 2024) mixes Transformer and Mamba layers with MoE
- Motivation: attention excels at precise recall; SSMs excel at efficient long-range propagation

---

## Part 3: Diffusion LLMs

> A novel paradigm applying diffusion processes -- originally developed for image generation -- to text generation.

### Background: Diffusion Models for Images

**Core idea:**
1. **Forward process**: Gradually add Gaussian noise to data over T timesteps until it becomes pure noise
2. **Reverse process**: Learn to denoise step-by-step, recovering the original data from noise

**Forward process (adding noise):**

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \, x_{t-1}, \beta_t I)$$

where beta_t is the noise schedule.

**Reverse process (denoising):**

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

### Applying Diffusion to Text

**Challenge:** Text is discrete, but diffusion operates on continuous spaces.

**Approaches:**

**1. Continuous Diffusion for Text (e.g., Diffusion-LM, Li et al., 2022):**
- Embed discrete tokens into continuous space
- Apply diffusion in embedding space
- Round back to discrete tokens at the end
- Enables non-autoregressive generation (all tokens generated simultaneously)

**2. Discrete Diffusion (e.g., D3PM, Austin et al., 2021; MDLM, Sahoo et al., 2024):**
- Define corruption process directly on discrete tokens (e.g., masking tokens)
- Forward process: progressively mask/corrupt tokens
- Reverse process: learn to unmask/reconstruct tokens
- Closely related to masked language modeling but with iterative refinement

**3. Masked Diffusion (MDLM -- Masked Diffusion Language Model):**
- Forward process: gradually mask tokens with a schedule
- At time t=0: clean text; at time t=1: all tokens masked
- Reverse process: iteratively predict and unmask tokens
- Can generate all tokens in parallel, then iteratively refine

### Advantages of Diffusion LLMs

- **Non-autoregressive generation**: can generate all positions simultaneously rather than left-to-right
- **Iterative refinement**: can revise and improve outputs over multiple denoising steps
- **Global coherence**: considers all positions at once, potentially better for long-range consistency
- **Controllable generation**: diffusion framework naturally supports guidance (classifier-free guidance, etc.)
- **Flexible editing**: can infill, edit, or regenerate parts of text without regenerating everything

### Challenges and Limitations

- **Discrete nature of text**: fundamental mismatch with continuous diffusion; requires careful design
- **Sampling speed**: multiple denoising steps needed (though fewer than autoregressive steps for long sequences)
- **Quality gap**: current diffusion LLMs generally lag behind autoregressive Transformers on standard benchmarks
- **Training complexity**: often requires specialized training procedures
- **Vocabulary handling**: rounding from continuous to discrete space can introduce errors

### Recent Developments

- **Mercury (Inception Labs, 2025)**: diffusion-based LLM claiming significantly faster generation than autoregressive models
- Research on hybrid approaches combining autoregressive and diffusion methods
- Exploration of diffusion for specific tasks: code generation, constrained text generation, text editing

---

## Part 4: Closing Thoughts

### Course Journey Summary

| Lecture | Topic | Key Concepts |
|---|---|---|
| 1 | Transformers | Attention mechanism, self-attention, multi-head attention, encoder-decoder architecture |
| 2 | Transformer-based models & tricks | RoPE, GQA, MQA, Flash Attention, decoder-only models |
| 3 | Large Language Models | MoE, context length, sampling, prompting, in-context learning, CoT |
| 4 | LLM Training | Scaling laws, Chinchilla, pretraining, quantization, LoRA |
| 5 | LLM Tuning | RLHF, reward models, PPO, DPO, preference optimization |
| 6 | LLM Reasoning | Reasoning models, GRPO, test-time compute scaling |
| 7 | Agentic LLMs | RAG, function calling, agents, ReAct |
| 8 | LLM Evaluation | LLM-as-a-judge, biases, evaluation best practices |
| 9 | Current Trends | Course recap, beyond Transformers, diffusion LLMs |

### Trending Topics in LLMs (as of late 2025)

- **Alternative architectures**: SSMs (Mamba), RWKV, hybrid models challenging Transformer dominance
- **Diffusion-based language models**: non-autoregressive text generation
- **Reasoning and planning**: deeper reasoning capabilities (o1-style models, GRPO)
- **Multimodal models**: unified models handling text, vision, audio, video
- **Smaller, more efficient models**: distillation, quantization, efficient architectures
- **Agentic AI**: autonomous agents that can plan, use tools, and interact with environments
- **Safety and alignment**: ongoing work on RLHF, constitutional AI, red-teaming
- **Open-source ecosystem**: continued growth of open models (LLaMA, Mistral, etc.)

### The Field is Moving Fast

- New architectures and training techniques emerging rapidly
- The Transformer remains dominant but faces viable challengers
- Efficiency (both training and inference) is a major research direction
- The boundary between "language model" and "general AI system" continues to blur

---

## Key Formulas Reference

### Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### Rotary Position Embedding (RoPE)
$$R_{\theta,m} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}$$

### Scaling Laws (Kaplan et al., 2020)
- Compute: $L = (C_{min} / 2.3 \times 10^8)^{-0.050}$
- Dataset: $L = (D / 5.4 \times 10^{13})^{-0.095}$
- Parameters: $L = (N / 8.8 \times 10^{13})^{-0.076}$

### State Space Model (Continuous-Time)
$$\dot{x}(t) = Ax(t) + Bu(t)$$
$$y(t) = Cx(t) + Du(t)$$

### Diffusion Forward Process
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \, x_{t-1}, \beta_t I)$$

### Diffusion Reverse Process
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

---

## Key References

1. Vaswani, A. et al. (2017). "Attention Is All You Need."
2. Su, J. et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embeddings."
3. Ainslie, J. et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints."
4. Shazeer, N. et al. (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer."
5. Jiang, A. et al. (2024). "Mixtral of Experts."
6. Kaplan, J. et al. (2020). "Scaling Laws for Neural Language Models."
7. Hoffmann, J. et al. (2022). "Training Compute-Optimal Large Language Models." (Chinchilla)
8. Gu, A. et al. (2022). "Efficiently Modeling Long Sequences with Structured State Spaces." (S4)
9. Gu, A. & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces."
10. Peng, B. et al. (2023). "RWKV: Reinventing RNNs for the Transformer Era."
11. Li, X. et al. (2022). "Diffusion-LM Improves Controllable Text Generation."
12. Austin, J. et al. (2021). "Structured Denoising Diffusion Models in Discrete State-Spaces." (D3PM)
13. Sahoo, S. et al. (2024). "Simple and Effective Masked Diffusion Language Models." (MDLM)
14. Amidi, A. & Amidi, S. (2024). "Super Study Guide: Transformers & Large Language Models."

---

*Note: The slides PDF contained image-based slides. Content for the Recap (Lectures 1-4) was extracted directly from slide images. Content for Lectures 5-8 recap, "Beyond Transformer-Based LLMs," "Diffusion LLMs," and "Closing Thoughts" sections was reconstructed from the syllabus topics and the lecture's stated agenda. When the remaining slide images become available, this document should be updated with exact slide content.*
