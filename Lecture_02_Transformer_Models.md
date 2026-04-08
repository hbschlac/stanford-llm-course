# CME 295: Transformers & Large Language Models -- Lecture 2

| Field | Value |
|-------|-------|
| **Course** | CME 295, Stanford University (ICME) |
| **Instructors** | Afshine Amidi & Shervine Amidi |
| **Lecture Date** | October 3, 2025 |
| **YouTube** | [https://www.youtube.com/watch?v=yT84Y5zCnaA](https://www.youtube.com/watch?v=yT84Y5zCnaA) |
| **Slides** | [https://cme295.stanford.edu/slides/fall25-cme295-lecture2.pdf](https://cme295.stanford.edu/slides/fall25-cme295-lecture2.pdf) |
| **Topics** | Position Embeddings, Layer Normalization, Attention Approximation, Transformer-based Models, BERT Deep Dive |

---

## Table of Contents

1. [Recap of Last Episode](#1-recap-of-last-episode)
2. [Position Embeddings](#2-position-embeddings)
   - [Need for Position Information](#21-need-for-position-information)
   - [Hardcoded (Sinusoidal) Position Embeddings](#22-hardcoded-sinusoidal-position-embeddings)
   - [From Absolute to Relative Position Info](#23-from-absolute-to-relative-position-info)
   - [Linear Bias in Attention Layer (ALiBi, T5)](#24-linear-bias-in-attention-layer-alibi-t5)
   - [RoPE -- Rotary Position Embeddings](#25-rope----rotary-position-embeddings)
3. [Layer Normalization](#3-layer-normalization)
   - [Layer Normalization (LN)](#31-layer-normalization-ln)
   - [Post-Norm vs Pre-Norm](#32-post-norm-vs-pre-norm)
   - [RMSNorm](#33-rmsnorm)
4. [Attention Approximation](#4-attention-approximation)
   - [Sparse Attention: Longformer](#41-sparse-attention-longformer)
   - [Sliding Window Attention (SWA)](#42-sliding-window-attention-swa)
   - [Sharing Attention Heads: MHA, MQA, GQA](#43-sharing-attention-heads-mha-mqa-gqa)
5. [Transformer-based Models](#5-transformer-based-models)
6. [BERT Deep Dive](#6-bert-deep-dive)
   - [Architecture Overview](#61-architecture-overview)
   - [Pretraining: Masked Language Model (MLM)](#62-pretraining-masked-language-model-mlm)
   - [Pretraining: Next Sentence Prediction (NSP)](#63-pretraining-next-sentence-prediction-nsp)
   - [Finetuning](#64-finetuning)
   - [Takeaways and Shortcomings](#65-takeaways-and-shortcomings)
   - [Trick: Distillation / DistilBERT](#66-trick-distillation--distilbert)
   - [Variant: RoBERTa](#67-variant-roberta)
7. [Key References](#7-key-references)

---

## 1. Recap of Last Episode

The lecture begins with a recap of the attention mechanism from Lecture 1.

**Self-attention mechanism:** Each token in a sentence has value vectors (v), key vectors (k), and query vectors (q). For a given query token (e.g., "teddy bear"), the attention mechanism computes how much to attend to every other token by comparing the query against all keys.

**Scaled dot-product attention formula:**

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

Where:
- `Q` = query matrix
- `K` = key matrix
- `V` = value matrix
- `d_k` = dimension of key vectors (scaling factor to prevent large dot products)

**Multi-Head Attention (MHA) architecture** (from the original Transformer paper):
- Input Q, K, V go through separate Linear projections
- Scaled Dot-Product Attention is computed in parallel across `h` heads
- Outputs are concatenated and passed through a final Linear layer

**Attention map example:** Showed anaphora resolution -- two attention heads in layer 5 of 6 that resolve the word "its" to "The Law" and "application," demonstrating how attention heads learn different linguistic relationships.

**Suggested reading:** "Attention Is All You Need" (Vaswani et al., 2017) -- the original Transformer paper.

**Lecture outline (agenda):**
1. Position embeddings
2. Layer normalization
3. Attention approximation
4. Transformer-based models
5. BERT deep dive

---

## 2. Position Embeddings

### 2.1 Need for Position Information

**Motivation:** The self-attention mechanism with direct connections between tokens "loses" position information. When computing attention, the model treats all tokens equally regardless of their position in the sequence -- a bag-of-words-like behavior.

**Idea:** Add learned position-specific embedding to each token vector. This is shown in the original Transformer architecture diagram where "Positional Encoding" is added to both input and output embeddings before entering the encoder/decoder stacks.

**Limitation of learned position embeddings:** Need to retrain for longer sequences (the model cannot generalize to sequence lengths not seen during training).

### 2.2 Hardcoded (Sinusoidal) Position Embeddings

**Idea:** Hardcode the values of position embeddings using sinusoidal functions instead of learning them.

**Formulas:**

```
PE_{m, 2i}   = sin(m / 10000^(2i/d_model))
PE_{m, 2i+1} = cos(m / 10000^(2i/d_model))
```

Equivalently, defining the frequency:

```
omega_i = 10000^(-2i / d_model)
```

Then:

```
PE_{m, 2i}   = sin(omega_i * m)
PE_{m, 2i+1} = cos(omega_i * m)
```

Where:
- `m` = position in the sequence
- `i` = dimension index (each pair of dimensions uses sin for even index 2i and cos for odd index 2i+1)
- `d_model` = model dimension

**Key property -- relative position encoding via dot product:**

Using the trigonometric identity:

```
cos(a - b) = cos(a)cos(b) + sin(a)sin(b)
```

We can show:

```
cos(omega_i(m - n)) = cos(omega_i * m)cos(omega_i * n) + sin(omega_i * m)sin(omega_i * n)
```

Therefore the inner product of two position embeddings is:

```
<PE_m, PE_n> = ... + cos(omega_i(m - n)) + ...
```

Which means:

```
<PE_m, PE_n> = f(m - n)
```

The dot product between position embeddings depends only on the **relative distance** (m - n), not on the absolute positions -- a desirable property.

**Visualizations shown:**
- **Value of embeddings:** Heatmap showing sinusoidal patterns across positions (x-axis) and dimensions (y-axis) -- alternating red/blue bands
- **Similarity between positions:** Heatmap showing that nearby positions have higher similarity (diagonal pattern), with similarity decreasing as distance increases

**Benefits:** Can extend to any sequence length (no retraining needed, unlike learned embeddings).

### 2.3 From Absolute to Relative Position Info

**Motivation:** We actually care about **relative position** when used **with the attention layer**. Instead of adding position info to the input embeddings, we can modify the attention mechanism itself to incorporate positional information.

**Idea:** Change the attention layer instead of the input embeddings.

### 2.4 Linear Bias in Attention Layer (ALiBi, T5)

**Idea:** Add a bias term to the query-key attention scores.

**Modified attention formula:**

```
softmax( (<q_m, k_n> / sqrt(d_k)) + bias(m, n) )
```

Where `m` and `n` are the positions of the query and key tokens respectively.

**T5 bias:** The bias is **learned per head**, using a bucketed function:

```
bias(m, n) = beta_{bucket(m-n)}
```

The distances are bucketed (grouped into ranges) and a learnable parameter beta is used for each bucket.

**ALiBi (Attention with Linear Biases):** The bias is **linear**, **deterministic**, and **not bounded**:

```
bias(m, n) = mu * (n - m)
```

Where `mu` is a fixed slope parameter (set per attention head, not learned).

**References:**
- "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (Raffel et al., 2023) -- for T5 bias
- "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (Press et al., 2021) -- for ALiBi

### 2.5 RoPE -- Rotary Position Embeddings

**RoPE = Rotary Position Embeddings** -- the default choice for modern LLMs.

**Idea:** Rotate query and key vectors with a rotation matrix based on their position.

**Rotation matrix (2D case):**

```
R_{theta,m} = | cos(m*theta)  -sin(m*theta) |
              | sin(m*theta)   cos(m*theta) |
```

The query at position `m` is rotated by angle `m*theta`, and the key at position `n` is rotated by angle `n*theta`.

**Visualization:** Showed vectors q_m and k_n being rotated in 2D space by their respective position-dependent angles m*theta and n*theta, producing q_m * R_{theta,m}^T and k_n * R_{theta,n}^T.

**Extension to dimension d > 2:** Rotate every block of dimension 2 independently. The full rotation matrix is block-diagonal:

```
R_{theta,m} = | block_1    0       ...    0       |
              | 0          ...     ...    ...     |
              | ...        ...     ...    0       |
              | 0          ...     0    block_{d_k/2} |
```

Where each block_i is a 2x2 rotation matrix:

```
block_i = | cos(m*theta_i)  -sin(m*theta_i) |
          | sin(m*theta_i)   cos(m*theta_i) |
```

**Benefits -- relative distance nicely captured:**

```
q_m * k_n^T = x_m * W_q * R_{theta, n-m} * W_k^T * x_n^T
```

The attention score between positions m and n depends on the **relative distance (n-m)** through the rotation matrix, which is the desired property.

**Attention weight has long-term decay:** The relative upper bound of A(q_m * k_n^T) decays as |m - n| increases, meaning tokens far apart naturally attend to each other less -- a desirable inductive bias.

**Reference:** "RoFormer: Enhanced Transformer with Rotary Position Embeddings" (Su et al., 2021)

---

## 3. Layer Normalization

### 3.1 Layer Normalization (LN)

**LN = Layer Normalization** -- applied at multiple points in the Transformer architecture (highlighted as "Norm" in the Add & Norm blocks after both Multi-Head Attention and Feed Forward layers).

**Formula:**

```
LN(x) = gamma * x_hat + beta
```

Where the normalized input is:

```
x_hat = (x - mu) / sqrt(sigma^2 + epsilon)
```

With statistics computed across the feature dimension:

```
mu = (1/d) * sum_{i=1}^{d} x_i

sigma^2 = (1/d) * sum_{i=1}^{d} (x_i - mu)^2
```

- `gamma` and `beta` are learnable scale and shift parameters
- `epsilon` is a small constant for numerical stability
- `d` is the feature dimension
- Statistics are computed **per-token** across the feature dimension (not across the batch)

**Benefits:** Helps with training stability and convergence.

**References:**
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Layer Normalization" (Ba et al., 2016)

### 3.2 Post-Norm vs Pre-Norm

**Post-Norm** (original Transformer):
- LayerNorm is applied **after** the residual connection
- Formula: `Output = LayerNorm(x + SubLayer(x))`
- Flow: x_l -> Multi-Head Attention -> addition (residual) -> Layer Norm -> FFN -> addition (residual) -> Layer Norm -> x_{l+1}

**Pre-Norm** (modern preference):
- LayerNorm is applied **before** the sublayer, inside the residual path
- Formula: `Output = x + SubLayer(LayerNorm(x))`
- Flow: x_l -> Layer Norm -> Multi-Head Attention -> addition (residual) -> Layer Norm -> FFN -> addition (residual) -> x_{l+1}

**Modern default:** Pre-Norm is the standard nowadays. It leads to more stable training, especially for deep networks.

**Reference:** "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)

### 3.3 RMSNorm

Modern architectures use **Pre-Norm + RMSNorm** (instead of standard Layer Normalization).

**RMSNorm simplification:**

Standard LayerNorm uses:
```
gamma * x_hat + beta
```

RMSNorm simplifies this to:
```
gamma * x / RMS(x)
```

Where:
```
RMS(x) = sqrt( (1/d) * sum_{i=1}^{d} x_i^2 )
```

Key differences from standard LN:
- No mean subtraction (removes the re-centering step)
- No beta (bias/shift) parameter
- Uses root mean square instead of standard deviation
- Computationally cheaper while maintaining similar performance

**Reference:** "Root Mean Square Layer Normalization" (Zhang et al., 2019)

---

## 4. Attention Approximation

### 4.1 Sparse Attention: Longformer

Standard self-attention is O(n^2) in sequence length, which becomes prohibitive for long sequences.

**Longformer** introduces sparse attention patterns:
- Instead of attending to all tokens (full attention matrix), each token only attends to a subset
- Showed the full attention matrix (all cells filled) vs. sparse version (many cells empty)

**Reference:** "Longformer: The Long-document Transformer" (Beltagy et al., 2020)

### 4.2 Sliding Window Attention (SWA)

**SWA = Sliding Window Attention**

**Idea:** Each token only attends to a local window of nearby tokens, rather than all tokens in the sequence.

**Variations include:**
- **Interleaving local and global attention layers** -- some layers use full attention, others use sliding window
- Used in **Mistral 7B** -- showed diagram of layers with alternating window sizes across the token sequence

**Analogy:** Similar to the "receptive field" concept in computer vision / convolutional neural networks -- stacking multiple layers of local attention allows information to propagate to distant tokens, just as stacking convolution layers increases the effective receptive field.

**Reference:**
- Mistral 7B announcement, 2023
- "VIP Cheatsheet: Convolutional Neural Networks" (Amidi, 2018) -- for receptive field analogy

### 4.3 Sharing Attention Heads: MHA, MQA, GQA

**Core idea:** Share key/value attention heads within groups of queries to reduce compute and memory.

The parameter `G` controls the number of key-value groups:

| G Value | Name | Description | Diagram |
|---------|------|-------------|---------|
| **G = 1** | **Multi-Query Attention (MQA)** | All query heads share a **single** K and V | One K,V pair shared by Q_1 ... Q_h (x1) |
| **1 < G < h** | **Grouped-Query Attention (GQA)** | Query heads are divided into G groups, each group shares one K,V | G groups, each with K,V and Q_1 ... Q_{h/G} (xG) |
| **G = h** | **Multi-Head Attention (MHA)** | Each query head has its own K and V (standard) | h independent {Q, K, V} triplets (xh) |

**Spectrum:** MQA (most parameter sharing, fastest) <-> GQA (middle ground) <-> MHA (no sharing, most expressive)

**Reference:** "Super Study Guide: Transformers & Large Language Models" (Amidi et al., 2024)

---

## 5. Transformer-based Models

**3 categories of Transformer-based models:**

| Category | Task Type | Examples | Architecture |
|----------|-----------|----------|-------------|
| **Encoder-decoder** | Text to text | T5, mT5, ByT5 | Full encoder + decoder stack |
| **Encoder-only** | Projection of embedding for class prediction (e.g., sentiment extraction) | BERT, DistilBERT, RoBERTa | Encoder stack only (decoder removed) |
| **Decoder-only** | Text to text | GPT series | Decoder stack only (encoder removed) |

**Historical context:**
- **Encoder-only models** were popular in ~2018-2022
- **Decoder-only models** are popular now (current era)

**Reference:** Figure adapted from "Attention Is All You Need" (Vaswani et al., 2017)

---

## 6. BERT Deep Dive

### 6.1 Architecture Overview

**BERT = Bidirectional Encoder Representations from Transformers**

**Key properties:**
- Encoder-only architecture
- Uses **bidirectional** (non-causal / non-masked) attention -- each token can attend to all other tokens in both directions
- Pre-trained on large unlabeled corpus, then fine-tuned on downstream tasks

**Two-phase approach:**
1. **Pretraining** -- learn general language representations
2. **Finetuning** -- adapt to specific downstream tasks

**Tokenization:** Uses **WordPiece** tokenizer
- Example: "teddy bear is reading" may be tokenized into subword units
- Special tokens: `[CLS]` (classification token at start), `[SEP]` (separator between segments), `[PAD]` (padding)

**BERT model specifications:**

| Specification | BERT Base | BERT Large |
|---------------|-----------|------------|
| N (layers) | 12 | 24 |
| d_model | 768 | 1024 |
| h (attention heads) | 12 | 16 |
| Total parameters | 110M | 340M |
| Context window | 512 | 512 |

**Reference:** "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2019)

### 6.2 Pretraining: Masked Language Model (MLM)

**Goal:** Predict masked tokens from context (bidirectional).

**Procedure:**
1. Take input text and randomly select ~15% of tokens for masking
2. Of the selected tokens:
   - 80% are replaced with `[MASK]` token
   - 10% are replaced with a random token
   - 10% are left unchanged
3. The model predicts the original token at each masked position

**Example:** "a cute teddy bear is reading" with "cute" and "reading" masked:
- Input: `a [MASK] teddy bear is [MASK]`
- Target: predict "cute" and "reading"

**Architecture for MLM:** The encoder output for each masked position is fed through a Feed-Forward Network (FFN) + Softmax over the vocabulary to predict the original token. The loss is computed only on the masked positions.

**Loss function:** Cross-entropy on masked tokens only.

**Reference:** "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2019)

### 6.3 Pretraining: Next Sentence Prediction (NSP)

**Goal:** Predict whether sentence B naturally follows sentence A.

**Procedure:**
1. Take pairs of sentences (A, B)
2. 50% of the time, B is the actual next sentence (label: IsNext)
3. 50% of the time, B is a random sentence (label: NotNext)
4. The model predicts the label using the `[CLS]` token representation

**Input format:**
```
[CLS] Sentence A [SEP] Sentence B [SEP]
```

**Architecture:** The `[CLS]` token output goes through an FFN and sigmoid/softmax for binary classification.

### 6.4 Finetuning

**Finetuning process for sentiment extraction (example):**

Step-by-step construction (shown as animated slides):

1. **Tokenize** the input: `[CLS] this teddy bear is so cute ! [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]`
2. **Embed** each token (token embedding)
3. **Add position embeddings** (positions 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
4. **Add segment embeddings** (Segment A for tokens up to [SEP], Segment B for [PAD] tokens)
5. Combine into **position- and segment-aware embedding**
6. All token embeddings are computed
7. Pass through the **pretrained BERT encoder**
8. **Sentiment extraction task:** Take the `[CLS]` token output, pass through an FFN with label "SE" (Sentiment Extraction), output a prediction (e.g., value of 1 = positive sentiment)

**Input representation = Token Embedding + Position Embedding + Segment Embedding**

### 6.5 Takeaways and Shortcomings

**Benefits:**
- State of the art results (at time of publication)
- ~True contextual representation of words (bidirectional context)
- Adaptable to many classification tasks

**Applications:** Widely used in the industry for anything related to encoding (search, classification, NER, etc.)

**Limitations:**
- Context window size is limited (512 tokens)
- Computationally expensive: hard sell for low-latency/cost-sensitive applications
- Training paradigm is complex: MLM/NSP + finetuning (two-stage process)

### 6.6 Trick: Distillation / DistilBERT

**Knowledge Distillation concept:**

Quote: "The **soft targets** contain almost all the **knowledge**." (Hinton et al., 2014 -- "Dark Knowledge")

**Distillation setup:**
- **Teacher model (T):** Large pretrained model that produces output distribution y_hat_T
- **Student model (S):** Smaller model that produces output distribution y_hat_S
- **Loss (L):** KL divergence between teacher and student output distributions

**KL Divergence formula:**

```
KL(y_hat_T || y_hat_S) = sum_i y_hat_T^(i) * log(y_hat_T^(i) / y_hat_S^(i))
```

The student learns to match the teacher's soft probability distribution, not just the hard labels.

**DistilBERT results:**
- BERT Base (N=12 layers) -> DistilBERT (N=6 layers)
- **~1.6x faster** inference
- **~97% performance** retained

**Reference:** "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter" (Sanh et al., 2019)

### 6.7 Variant: RoBERTa

**Goal:** Thorough analysis of directions of optimization for BERT pretraining.

**Modeling changes:**
- Removing NSP/segment encodings: **~no effect!** (*DistilBERT had also removed it*)
- Static masking -> **dynamic masking** across epochs (re-randomize which tokens are masked each epoch)

**Data changes:**
- **Richer data:** 16 GB -> 160 GB of pretraining corpus size
- **Train for longer:** 1M steps of batch size 256 vs. 500k of batch size 8k

**Result:** +4% across benchmarks with the same architecture. Shows that BERT was significantly undertrained and that training recipe matters as much as architecture.

**Reference:** "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (Liu et al., 2019)

---

## 7. Key References

| Paper | Authors | Year |
|-------|---------|------|
| Attention Is All You Need | Vaswani et al. | 2017 |
| Transformer Architecture: The Positional Encoding | Kazemnejad | 2019 |
| RoFormer: Enhanced Transformer with Rotary Position Embeddings | Su et al. | 2021 |
| Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (ALiBi) | Press et al. | 2021 |
| Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5) | Raffel et al. | 2023 |
| Layer Normalization | Ba et al. | 2016 |
| On Layer Normalization in the Transformer Architecture | Xiong et al. | 2020 |
| Root Mean Square Layer Normalization | Zhang et al. | 2019 |
| Longformer: The Long-document Transformer | Beltagy et al. | 2020 |
| BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | Devlin et al. | 2019 |
| DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter | Sanh et al. | 2019 |
| RoBERTa: A Robustly Optimized BERT Pretraining Approach | Liu et al. | 2019 |
| Dark Knowledge (Distillation) | Hinton et al. | 2014 |
| Super Study Guide: Transformers & Large Language Models | Amidi et al. | 2024 |
| VIP Cheatsheet: Convolutional Neural Networks | Amidi | 2018 |
| Mistral 7B announcement | Mistral AI | 2023 |

---

## Key Formulas Summary

### Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

### Sinusoidal Position Embeddings
```
PE_{m, 2i}   = sin(omega_i * m)
PE_{m, 2i+1} = cos(omega_i * m)
where omega_i = 10000^(-2i / d_model)
```

### RoPE Rotation Matrix (2D)
```
R_{theta,m} = | cos(m*theta)  -sin(m*theta) |
              | sin(m*theta)   cos(m*theta) |
```

### RoPE Attention Score (relative position property)
```
q_m * k_n^T = x_m * W_q * R_{theta, n-m} * W_k^T * x_n^T
```

### Linear Bias (ALiBi)
```
softmax( <q_m, k_n> / sqrt(d_k) + mu * (n - m) )
```

### T5 Bias
```
bias(m, n) = beta_{bucket(m-n)}
```

### Layer Normalization
```
LN(x) = gamma * x_hat + beta
x_hat = (x - mu) / sqrt(sigma^2 + epsilon)
mu = (1/d) * sum(x_i)
sigma^2 = (1/d) * sum((x_i - mu)^2)
```

### RMSNorm
```
RMSNorm(x) = gamma * x / RMS(x)
RMS(x) = sqrt( (1/d) * sum(x_i^2) )
```

### Post-Norm vs Pre-Norm
```
Post-Norm: Output = LayerNorm(x + SubLayer(x))
Pre-Norm:  Output = x + SubLayer(LayerNorm(x))
```

### KL Divergence (Distillation)
```
KL(y_hat_T || y_hat_S) = sum_i y_hat_T^(i) * log(y_hat_T^(i) / y_hat_S^(i))
```

---

## Key Concepts Glossary

| Term | Definition |
|------|-----------|
| **MHA** | Multi-Head Attention -- each head has independent Q, K, V (G = h) |
| **MQA** | Multi-Query Attention -- all heads share one K, V pair (G = 1) |
| **GQA** | Grouped-Query Attention -- heads grouped, each group shares K, V (1 < G < h) |
| **RoPE** | Rotary Position Embeddings -- encodes position by rotating Q, K vectors |
| **ALiBi** | Attention with Linear Biases -- adds linear position-dependent bias to attention scores |
| **SWA** | Sliding Window Attention -- each token attends only to a local window |
| **LN** | Layer Normalization -- normalizes across features per token |
| **RMSNorm** | Root Mean Square Normalization -- simplified LN without mean subtraction |
| **Pre-Norm** | LayerNorm applied before sublayer (modern default) |
| **Post-Norm** | LayerNorm applied after sublayer + residual (original Transformer) |
| **MLM** | Masked Language Model -- BERT pretraining objective, predict masked tokens |
| **NSP** | Next Sentence Prediction -- BERT pretraining objective, predict if B follows A |
| **WordPiece** | Subword tokenization used by BERT |
| **[CLS]** | Special classification token prepended to input in BERT |
| **[SEP]** | Separator token between sentences in BERT |
| **[MASK]** | Masking token used during MLM pretraining |
| **Distillation** | Training a smaller student model to mimic a larger teacher model's soft outputs |

---

## Video Content

> **Note:** YouTube transcript for this lecture was not available for automated extraction at time of note creation. Watch the full lecture at: [https://www.youtube.com/watch?v=yT84Y5zCnaA](https://www.youtube.com/watch?v=yT84Y5zCnaA)

---

*Notes compiled from 109 lecture slides. Source: CME 295, Stanford University.*
