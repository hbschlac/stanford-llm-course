# CME 295: Large Language Models — Lecture 4: LLM Training

**Course:** Stanford CME 295 — Large Language Models
**Video:** [YouTube — Lecture 4](https://www.youtube.com/watch?v=VlA_jt_3Qc4)
**Slides:** [PDF](https://cme295.stanford.edu/slides/fall25-cme295-lecture4.pdf)

**Topics Covered:**
- Pretraining objective (next-token prediction / causal language modeling)
- Training data sources and preprocessing
- Tokenizer training (BPE)
- Hardware for training (GPUs, FLOPS, memory hierarchy)
- Flash Attention
- Mixed precision training (FP32, FP16, BF16)
- Distributed training (data parallelism, model parallelism, ZeRO, FSDP)
- Supervised fine-tuning (SFT) and instruction tuning
- Parameter-efficient fine-tuning (LoRA, QLoRA)
- Evaluation and benchmarks

---

## 1. Pretraining

### The Training Objective

The core objective for pretraining an LLM is **next-token prediction**, also called **causal language modeling (CLM)**. Given a sequence of tokens, the model learns to predict the next token:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t \mid x_1, x_2, \ldots, x_{t-1}; \theta)$$

where:
- $x_1, x_2, \ldots, x_T$ is the token sequence
- $\theta$ represents the model parameters
- The loss is the negative log-likelihood summed over all positions

In plain language: the model reads text from left to right and tries to guess what word comes next, over and over, billions of times. This simple objective is what gives LLMs their powerful language understanding.

### Training Data Sources

Pretraining data comes from a mixture of large-scale text corpora:

| Source | Description | Example |
|--------|-------------|---------|
| Web crawl data | Massive internet scrapes | Common Crawl, C4, RefinedWeb |
| Books | Long-form structured text | Books3, Project Gutenberg |
| Wikipedia | Encyclopedic knowledge | Full Wikipedia dumps |
| Code | Programming languages | GitHub, The Stack |
| Scientific papers | Academic literature | arXiv, Semantic Scholar |
| Conversational data | Dialogue and forums | Reddit, StackOverflow |

**Scale matters.** Modern LLMs train on trillions of tokens:

| Model | Training Tokens |
|-------|----------------|
| GPT-3 (2020) | 300B |
| LLaMA 2 (2023) | 2T |
| LLaMA 3 (2024) | 15T+ |

### Data Preprocessing and Filtering

Raw web data is noisy. Preprocessing pipelines typically include:

1. **Language filtering** -- keeping only target language(s) (e.g., English)
2. **Deduplication** -- removing exact and near-duplicate documents to prevent the model from memorizing repeated content
3. **Quality filtering** -- using heuristics or classifiers to remove low-quality pages (spam, boilerplate, gibberish)
4. **Toxicity and PII filtering** -- removing harmful content and personally identifiable information
5. **Domain balancing** -- controlling the mix of sources (more code, more books, etc.) to influence the model's strengths

### Tokenizer Training (BPE)

Before training the model, you need a **tokenizer** to convert raw text into tokens (integer IDs). The most common approach is **Byte Pair Encoding (BPE)**.

**How BPE works:**
1. Start with individual characters (or bytes) as the initial vocabulary
2. Count all adjacent pairs of tokens in the training corpus
3. Merge the most frequent pair into a new token
4. Repeat until you reach the desired vocabulary size (typically 32K-128K tokens)

**Example:**
- Starting: `l o w e r`
- After merging frequent pairs: `lo w er` then `low er` then `lower`

**Why BPE?**
- Handles any text (no out-of-vocabulary problem when using byte-level BPE)
- Common words become single tokens (efficient)
- Rare words get split into subword pieces (still representable)
- Vocabulary size is a tunable hyperparameter

Common tokenizers used by major models:
| Model | Tokenizer | Vocabulary Size |
|-------|-----------|----------------|
| GPT-2 | BPE | 50,257 |
| LLaMA | SentencePiece (BPE) | 32,000 |
| LLaMA 3 | tiktoken (BPE) | 128,256 |
| GPT-4 | tiktoken (BPE) | ~100K |

---

## 2. Hardware for Training

### GPUs: The Workhorse of LLM Training

LLMs are trained almost exclusively on **GPUs** (Graphics Processing Units), specifically NVIDIA's data center GPUs. The key specs that matter:

- **FLOPS** (Floating Point Operations Per Second) -- how fast it can do math
- **VRAM** (Video RAM / GPU Memory) -- how much data it can hold on-chip
- **Memory Bandwidth** -- how fast data can move between memory and compute units

### GPU Memory Hierarchy

Understanding GPU memory is critical for training optimization:

| Memory Type | Speed | Size | Description |
|-------------|-------|------|-------------|
| **SRAM** (on-chip) | Very fast (~19 TB/s) | Very small (~20 MB) | Used for active computation |
| **HBM** (High Bandwidth Memory) | Fast (~1.5-3.35 TB/s) | Large (40-80 GB) | Main GPU memory (VRAM) |

The key bottleneck: **moving data between HBM and SRAM is often slower than actually doing the computation**. This is why memory-aware algorithms like Flash Attention matter so much.

### NVIDIA H100 Specifications

The H100 is the flagship GPU for LLM training as of 2024-2025:

| Specification | H100 SXM | H100 NVL |
|--------------|----------|----------|
| FP64 | 34 teraFLOPS | 30 teraFLOPS |
| FP64 Tensor Core | 67 teraFLOPS | 60 teraFLOPS |
| FP32 | 67 teraFLOPS | 60 teraFLOPS |
| TF32 Tensor Core | 989 teraFLOPS | 835 teraFLOPS |
| BF16 Tensor Core | 1,979 teraFLOPS | 1,671 teraFLOPS |
| FP16 Tensor Core | 1,979 teraFLOPS | 1,671 teraFLOPS |
| FP8 Tensor Core | 3,958 teraFLOPS | 3,341 teraFLOPS |
| INT8 Tensor Core | 3,958 TOPS | 3,341 TOPS |
| GPU Memory | 80 GB | 94 GB |
| Memory Bandwidth | 3.35 TB/s | 3.9 TB/s |
| Interconnect | NVLink: 900 GB/s | NVLink: 600 GB/s |

**Key takeaway: Lower precision = faster processing.** FP8 delivers roughly 60x more throughput than FP64. This is why mixed precision and quantization are so important.

---

## 3. Flash Attention

### The Problem with Standard Attention

The standard self-attention computation follows this sequence:

$$\text{Attention}(Q, K, V) = \text{softmax}(QK^T) \cdot V$$

Step by step, the standard approach does:
1. **LOAD** Q, K from HBM by blocks
2. **Compute** $S = QK^T$ on chip
3. **WRITE** S back to HBM (slow!)
4. **READ** S from HBM (slow!)
5. **Compute** $P = \text{softmax}(S)$ on chip
6. **WRITE** P back to HBM (slow!)
7. **LOAD** P, V from HBM
8. **Compute** $O = PV$ on chip
9. **WRITE** O to HBM

The problem: the intermediate matrices S and P are huge ($N \times N$ where N is the sequence length), and we keep writing them to slow HBM and reading them back. This creates a memory I/O bottleneck.

### Flash Attention: Two Key Ideas

**Idea 1: Tiling -- minimize reads/writes to HBM via SRAM**

Instead of computing the full $S = QK^T$ matrix, Flash Attention:
- Splits Q, K, V into small **blocks** (tiles)
- Loads blocks of Q, K, V into fast **SRAM**
- Computes a block of the output O directly
- Writes only the final result block to HBM
- Never materializes the full $N \times N$ attention matrix

The mathematical trick that enables this: you do not need to compute the full $S = QK^T$ before applying softmax. The softmax can be decomposed blockwise:

$$\text{softmax}([S_1, \ldots, S_n]) = [\alpha_1 \cdot \text{softmax}(S_1), \ldots, \alpha_n \cdot \text{softmax}(S_n)]$$

where $\alpha_i$ are rescaling factors that account for the global maximum.

**Idea 2: Recompute instead of storing**

During the backward pass, instead of storing S and P in HBM (which takes a lot of memory), Flash Attention **recomputes** them on the fly using tiling through SRAM. This trades a small amount of extra compute for a large memory savings.

The counterintuitive result: **more FLOPs, but less runtime** -- because the bottleneck was memory I/O, not compute.

### Flash Attention Results

| Metric | Standard Attention | Flash Attention |
|--------|-------------------|-----------------|
| GFLOPs | 66.6 | 75.2 |
| HBM Read/Write (GB) | 40.3 | **4.4** |
| Runtime (ms) | 41.7 | **7.3** |

Flash Attention uses slightly more FLOPs but reduces HBM traffic by ~9x and runtime by ~5.7x. It computes the **exact same result** -- no approximation.

**Reference:** "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", Dao et al., 2022.

---

## 4. Mixed Precision Training

### Precision of Numbers

Every number in a computer is stored using a fixed number of bits. A floating-point number has three components:

| Component | Description | Example |
|-----------|-------------|---------|
| **Sign** | Positive or negative (1 bit) | 0 = +1, 1 = -1 |
| **Exponent** | Controls magnitude (range) | 8 bits can represent $2^{122}$ to $2^{127}$ |
| **Mantissa** | Controls precision (granularity after decimal point); also called significand or fraction | e.g., 1.75 |

### Floating Point Formats Comparison

| Format | Sign Bits | Exponent Bits | Mantissa Bits | Total Bits |
|--------|-----------|---------------|---------------|------------|
| **FP16** (Float 16) | 1 | 5 | 10 | 16 |
| **FP32** (Float 32) | 1 | 8 | 23 | 32 |
| **FP64** (Float 64) | 1 | 11 | 52 | 64 |
| **BF16** (Brain Float 16) | 1 | 8 | 7 | 16 |

**BF16 vs FP16:** Both use 16 bits, but BF16 has more exponent bits (8 vs 5) and fewer mantissa bits (7 vs 10). This means:
- BF16 has the **same range** as FP32 (same number of exponent bits)
- BF16 has **less precision** than FP16 (fewer mantissa bits)
- BF16 is preferred for training because the range matters more -- gradients can be very large or very small, and overflow/underflow causes training instability

### How Mixed Precision Training Works

**Objective:** Speed up training and decrease memory requirements by using lower-precision arithmetic where possible, while keeping critical computations in full precision.

The approach uses different precisions at different stages:

1. **Forward pass** -- Activations computed in **low precision** (FP16 or BF16)
   - Input x (stored in FP32) flows through function f, producing output y-hat in FP16
   - This is faster because lower precision = higher throughput on GPU tensor cores

2. **Backward pass** -- Gradient updates computed in **low precision** (FP16 or BF16)
   - Gradients with respect to model parameters are computed in FP16
   - This saves memory and speeds up the backward pass

3. **Weight update** -- Master copy of weights kept in **high precision** (FP32)
   - The optimizer maintains a full-precision (FP32) copy of the weights
   - Updates are applied in FP32 to avoid accumulation of rounding errors
   - The updated FP32 weights are then cast back to FP16 for the next forward pass

**Why is this necessary?** If you accumulate many small gradient updates in FP16, rounding errors build up and the model fails to train properly. Keeping the master weights in FP32 prevents this.

**Loss scaling** (for FP16): Because FP16 has limited range, small gradient values can underflow to zero. Loss scaling multiplies the loss by a large factor before backpropagation, then divides the gradients by the same factor before the weight update. BF16 largely avoids this problem because it has the same range as FP32.

**Reference:** "Mixed Precision Training", Micikevicius et al., 2017.

---

## 5. Distributed Training

Training large LLMs requires distributing computation across many GPUs. There are several complementary strategies.

### Data Parallelism

The simplest distributed training strategy:

1. **Replicate** the entire model on each GPU
2. **Split** each mini-batch of data across GPUs (each GPU gets a different shard of data)
3. Each GPU computes **forward and backward passes** on its data shard independently
4. **Synchronize gradients** across all GPUs (typically using all-reduce)
5. Each GPU **updates its local copy** of the model with the averaged gradients

**Limitation:** Every GPU must hold a full copy of the model. For a 70B parameter model in FP32, that is ~280 GB -- far more than any single GPU can hold (H100 has 80 GB).

### Model Parallelism

When a model is too large for a single GPU, you split the model itself across GPUs. There are two main approaches:

**Tensor Parallelism:**
- Splits individual layers (matrices) across GPUs
- For example, a large weight matrix $W$ is split column-wise across 4 GPUs, each holding $W_1, W_2, W_3, W_4$
- Each GPU computes a partial result, then results are combined
- Requires high-bandwidth interconnect (NVLink) because GPUs communicate at every layer

**Pipeline Parallelism:**
- Splits the model by layers across GPUs
- GPU 1 holds layers 1-10, GPU 2 holds layers 11-20, etc.
- Data flows through GPUs sequentially like a pipeline
- Uses micro-batching to keep all GPUs busy (one GPU processes micro-batch 2 while the next GPU processes micro-batch 1)
- Problem: **pipeline bubbles** -- GPUs sit idle waiting for data from the previous stage

### ZeRO Optimization (Zero Redundancy Optimizer)

ZeRO eliminates memory redundancy in data parallelism. It has three stages:

| Stage | What is Partitioned | Memory Savings |
|-------|-------------------|----------------|
| **ZeRO-1** | Optimizer states (e.g., Adam moments) | ~4x reduction |
| **ZeRO-2** | Optimizer states + gradients | ~8x reduction |
| **ZeRO-3** | Optimizer states + gradients + parameters | ~Nd reduction (N = number of GPUs) |

**How it works (ZeRO-3 / FSDP):**
- Instead of replicating the full model on every GPU, each GPU only stores a **shard** (1/N) of the parameters, gradients, and optimizer states
- When a GPU needs a parameter for computation, it **gathers** the full parameter from all GPUs (all-gather)
- After the backward pass, each GPU only keeps the gradient shard it owns
- This distributes memory evenly while still achieving data parallelism

### FSDP (Fully Sharded Data Parallelism)

FSDP is PyTorch's implementation of ZeRO Stage 3:

- Shards model parameters, gradients, and optimizer states across all GPUs
- Uses all-gather to reconstruct parameters when needed for computation
- Uses reduce-scatter to aggregate and distribute gradients
- Enables training models that are much larger than what fits on a single GPU
- Trades communication overhead for memory efficiency

**In practice,** most large-scale training combines multiple strategies:
- Tensor parallelism within a node (uses fast NVLink)
- Pipeline parallelism across nodes
- Data parallelism (FSDP) across the full cluster

---

## 6. Supervised Fine-Tuning (SFT)

### Why Fine-Tune?

A pretrained LLM has "basic knowledge" about language, code, and the world -- but it does not know how to be a helpful assistant. When you ask a pretrained model:

> "Can I put my teddy bear in the washer?"

It might respond: *"Teddy bears are often made of materials like polyester and cotton, with plastic eyes and sometimes small accessories."*

This is a text completion, not a helpful answer. The model continues the text statistically rather than responding to the user's intent.

### The Remedy: Fine-Tuning

Fine-tuning takes the pretrained model and trains it further on a curated dataset to produce desired behaviors:

```
Initialized model --> Pretraining --> Model with "basic knowledge" --> Fine-tuning --> Model tuned for specific tasks
```

### SFT = Supervised Fine-Tuning

**Idea:** Change the model's behavior by **tuning its weights** on paired examples.

**Strategy:**
1. Collect pairs of inputs/outputs with desired behavior (aka SFT data)
2. Train using the same next-token prediction objective, but *given the input*

**Special case -- Instruction Tuning:** SFT on instruction-following data. The goal is to "graduate" the model to being a helpful assistant.

### Instruction Tuning Data

The training data consists of diverse instruction-response pairs across many task types:

| Task Type | Example Input | Example Output |
|-----------|--------------|----------------|
| Story writing | "Write a short story about a teddy bear who likes to read poetry." | "Once upon a time, a bear, Teddy, stumbled upon verses from Attar..." |
| Lists generation | "List three fun activities a teddy bear might do on a rainy day." | "Sure! 1. Read poetry with friends. 2. Be cute. 3. Hug its owner tightly." |
| Poem creation | "Create a poem about my cute teddy bear." | "Soft and cuddly, full of charm, / Always keeps me safe from harm..." |
| Explanation | "Explain why a teddy bear is a great friend." | "A teddy bear is a great friend because it provides comfort and companionship..." |

### Objective Function

The same next-token prediction loss, but applied **only to the output tokens** (the model is conditioned on the input but only trained to predict the response):

$$\mathcal{L}_{\text{SFT}} = -\sum_{t \in \text{output}} \log P(x_t \mid x_{<t}; \theta)$$

The input tokens provide context but their prediction loss is typically masked.

### Data Mixtures

SFT data can include both human-written and synthetic data:
- Assistant dialogs
- Synthetic instructions
- Maths, reasoning, code
- Safety alignment data
- And more...

**Size:** Thousands to millions of examples.

| Model | SFT Size (# examples) |
|-------|----------------------|
| GPT-3 | 13 thousand |
| LLaMA 3 | 10 million |

### Result After Instruction Tuning

After instruction tuning, the same question gets a helpful response:

> "Can I put my teddy bear in the washer?"

Instruction-tuned model: *"No, it might get damaged. Try hand washing instead."*

**References:**
- "Finetuned language models are zero-shot learners", Wei et al., 2022.
- "Language Models are Few-Shot Learners", Brown et al., 2020.
- "The Llama 3 Herd of Models", LLaMA team, 2024.

### Challenges of SFT

- Very **high-quality** data needed
- Sensitive to **prompt distribution** -- performance depends on how similar test prompts are to training prompts
- **Generalization** -- hard to cover all possible use cases
- Difficult to **evaluate** -- how do you measure "helpfulness"?
- Computationally **expensive** -- still requires training all model parameters

---

## 7. Evaluation and Benchmarks

### Standard Benchmarks

Fine-tuned models are evaluated across multiple dimensions:

| Benchmark | What It Measures |
|-----------|-----------------|
| **MMLU** | General knowledge (57 subjects, multiple choice) |
| **ARC-Challenge** | Basic reasoning |
| **GSM8K** | Math reasoning (grade school math) |
| **HumanEval** | Code generation |

**Validity note:** It is recommended to train on the test *task* (not the test *data*) to compare across models. Training on the actual test data confounds evaluation and emergence signals.

**Reference:** "Training on the Test Task Confounds Evaluation and Emergence", Dominguez-Olmedo et al., 2024.

### "Real-Life" Evaluation: Chatbot Arena

Websites like **Chatbot Arena** (lmarena.ai) run A/B tests for user prompts:
- Users submit prompts and get responses from two anonymous models
- They vote for the better response
- Models are ranked using an Elo rating system

**Benefits:** Puts a number on "vibes" -- measures real user preference.

**Outstanding challenges:**
- Unequal exposure of models / "cold start" problem
- Easy to "rig" -- models can be optimized for arena-style prompts
- Users cannot accurately assess important aspects (e.g., factuality)
- Personal preference bias (non-representative distribution)
- Safety penalization

Bottom line: **evaluation is a hard problem in itself!**

**Reference:** "Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards", Huang et al., 2025.

---

## 8. Parameter-Efficient Fine-Tuning (PEFT)

### The Problem with Full Fine-Tuning

Full SFT is resource-intensive:
- You need to update **all** parameters in the model
- You need to store gradients and optimizer states for every parameter
- A 70B model requires hundreds of GB of GPU memory for training
- Not everyone has access to clusters of expensive GPUs

### LoRA (Low-Rank Adaptation)

**Core idea:** Instead of updating the full weight matrix during fine-tuning, approximate the weight update as the product of two small, low-rank matrices.

**The math:**

$$W = W_0 + B \times A$$

where:
- $W$ is the effective weight matrix used during inference
- $W_0$ is the original pretrained weight matrix (**frozen** -- not updated)
- $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ are small trainable matrices
- $r$ is the **rank** -- a small number (e.g., 4, 8, 16, 64), much smaller than d or k

**Why it works:**
- Research suggests that the weight updates during fine-tuning have low intrinsic rank -- you do not need to update every parameter independently
- Only a **fraction** of the total parameters need to be trained, with similar performance to full fine-tuning

**Concrete savings example:**
- Full weight matrix $W$: $d \times k$ parameters (e.g., 4096 x 4096 = 16.7M)
- LoRA matrices $B$ and $A$: $d \times r + r \times k$ parameters (e.g., 4096 x 16 + 16 x 4096 = 131K)
- That is roughly **0.8% of the original parameters** for rank 16

### Finetuning Evolution: Before vs After

**Before (full fine-tuning):** Optimize the full weight matrix $W$ directly -- every parameter is trainable.

**After (LoRA):** Freeze $W_0$, only train small matrices $A$ and $B$ that represent the update. The output is $W_0 x + B A x$.

### Benefit of LoRA: Swap Matrices = Swap Tasks

One of the most powerful properties of LoRA is **modularity**. Since the base model $W_0$ stays the same, you can:

- Train $B_{\text{spam}}, A_{\text{spam}}$ for spam detection
- Train $B_{\text{sentiment}}, A_{\text{sentiment}}$ for sentiment extraction
- Train $B_{\text{translation}}, A_{\text{translation}}$ for translation

To switch tasks, you just swap the small LoRA matrices -- no need to reload the entire model. This makes serving multiple tasks from a single base model very efficient.

### Where to Apply LoRA

**Originally** (Hu et al., 2021): Applied to the attention weight matrices ($W_Q, W_K, W_V, W_O$) in the Masked Multi-Head Attention layer.

**Updated guidance** (Schulman et al., 2025, "LoRA Without Regret"): The **Feed-Forward layer** is the most important location to apply LoRA. Current best practice is to apply LoRA to **both** attention and feed-forward layers.

### Training Dynamics with LoRA

Beware of differences from full fine-tuning (these are empirical findings):
- LoRA needs a **higher learning rate** than full fine-tuning
- LoRA does **poorly with large batch sizes** compared to full fine-tuning

**Reference:** "LoRA Without Regret", Schulman et al., 2025.

### Other PEFT Methods

LoRA is the most popular, but other methods exist:
- **Prefix tuning** -- prepend trainable tokens to the input at every layer
- **Adapters** -- insert small trainable modules between existing layers
- **Prompt tuning** -- learn soft prompt embeddings prepended to the input

---

## 9. QLoRA (Quantized LoRA)

### The Idea

**QLoRA** combines quantization with LoRA to further reduce memory requirements:
- **Quantize** all frozen weights ($W_0$) to 4-bit precision (NF4) -- this is just for storage
- Keep the LoRA matrices ($B$ and $A$) in **full precision** (FP16/BF16)
- All **computations** are done in full precision (weights are dequantized on the fly)

### Efficient Quantization: NF4

**Trick:** Use 4-bit NormalFloat (NF4) to optimally split the quantization space.

Standard INT8 quantization uses **uniform distribution cutoffs** -- evenly spaced bins. But neural network weights follow an approximately **normal distribution**, so most values cluster near zero.

NF4 uses **Normal quantile-based cutoffs** -- placing bin boundaries based on the quantiles of a normal distribution. This means more bins where the data is dense (near zero) and fewer where it is sparse (the tails). Result: better quality for the same number of bits.

### Double Quantization

QLoRA introduces a **double quantization** trick to save even more memory:

1. **Single quantization:** Weights are quantized from FP32 to NF4 (4-bit). The quantization constants (scale factors for each block of weights) are stored in FP32.
2. **Double quantization:** The quantization constants themselves are also quantized from FP32 to FP8 (8-bit), with their own quantization constants stored in FP32.

This reduces the memory overhead of storing quantization metadata by an additional ~6%.

### QLoRA Results

Results reported on LLaMA 65B:
- **~16x VRAM savings** during fine-tuning
- Double quantization trick saves an extra ~6% memory
- VRAM savings enable fine-tuning on smaller GPUs, and faster
- Better trade-off between memory resources and quality

**Reference:** "QLoRA: Efficient Finetuning of Quantized LLMs", Dettmers et al., 2023.

---

## 10. The Full LLM Lifecycle

### Summary Pipeline

The complete lifecycle of building an LLM consists of three stages:

```
Initialized model
       |
       v
  [Pretraining]          -->  Model with "basic knowledge" about language, code, etc.
       |
       v
  [Finetuning (SFT)]     -->  Model tuned for specific tasks
       |
       v
  [Preference Tuning]    -->  Model does not misbehave as much
```

The combination of fine-tuning and preference tuning is often called **"Alignment"** of the model. Preference tuning (covered in Lecture 5) uses techniques like RLHF (Reinforcement Learning from Human Feedback) and DPO (Direct Preference Optimization) to teach the model to produce outputs that humans prefer.

---

## Key References

| Topic | Paper | Authors | Year |
|-------|-------|---------|------|
| Flash Attention | "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" | Dao et al. | 2022 |
| Mixed Precision | "Mixed Precision Training" | Micikevicius et al. | 2017 |
| Instruction Tuning | "Finetuned language models are zero-shot learners" | Wei et al. | 2022 |
| LoRA | "LoRA: Low-Rank Adaptation of Large Language Models" | Hu et al. | 2021 |
| LoRA Best Practices | "LoRA Without Regret" | Schulman et al. | 2025 |
| QLoRA | "QLoRA: Efficient Finetuning of Quantized LLMs" | Dettmers et al. | 2023 |
| Evaluation Challenges | "Training on the Test Task Confounds Evaluation and Emergence" | Dominguez-Olmedo et al. | 2024 |
| Arena Manipulation | "Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards" | Huang et al. | 2025 |
| Float Representations | "Super Study Guide: Transformers and Large Language Models" | Amidi et al. | 2024 |
| GPT-3 | "Language Models are Few-Shot Learners" | Brown et al. | 2020 |
| LLaMA 3 | "The Llama 3 Herd of Models" | LLaMA team | 2024 |
| H100 GPU | "NVIDIA H100 Tensor Core GPU" | NVIDIA | -- |
