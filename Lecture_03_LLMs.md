# CME 295: Large Language Models — Lecture 3: LLMs

**Date:** October 10, 2025
**Course:** Stanford CME 295 — Large Language Models
**Video:** [YouTube — Lecture 3](https://www.youtube.com/watch?v=Q5baLehv5So)
**Slides:** [PDF](https://cme295.stanford.edu/slides/fall25-cme295-lecture3.pdf)

**Topics Covered:**
- Definition and architecture of Large Language Models (LLMs)
- Mixture of Experts (MoE)
- Context length
- Sampling strategies (temperature, top-k, top-p / nucleus sampling)
- Prompting techniques (zero-shot, few-shot, system prompts)
- Chain of thought (CoT) reasoning

---

## 1. What Is a Large Language Model?

### Definition
A **Large Language Model (LLM)** is a neural network trained on massive text corpora to model the probability distribution over sequences of tokens. At its core, an LLM learns:

$$P(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

That is, the conditional probability of the next token given all preceding tokens. This is called **autoregressive language modeling** (or causal language modeling).

### Key Properties
- **"Large"** refers to both the number of parameters (billions to trillions) and the scale of training data (trillions of tokens)
- LLMs are **foundation models** — pretrained on broad data and adapted for many downstream tasks
- They perform **in-context learning**: they can learn to perform new tasks from examples provided in the prompt, without updating weights

### What Makes an LLM "Large"?
| Model | Parameters | Training Tokens |
|-------|-----------|----------------|
| GPT-2 (2019) | 1.5B | ~40B |
| GPT-3 (2020) | 175B | 300B |
| LLaMA 2 (2023) | 7B–70B | 2T |
| LLaMA 3 (2024) | 8B–405B | 15T+ |
| GPT-4 (2023) | Estimated >1T (MoE) | Unknown |
| Mixtral (2024) | 8x7B (MoE) | Unknown |

---

## 2. Architecture of LLMs

### The Transformer (Recap from Lecture 2)
LLMs are built on the **Transformer architecture** (Vaswani et al., 2017, "Attention Is All You Need"). Most modern LLMs use the **decoder-only** variant.

### Decoder-Only Transformer Architecture
The standard decoder-only transformer consists of a stack of identical layers, each containing:

1. **Masked Multi-Head Self-Attention**
   - Prevents attending to future tokens (causal masking)
   - Attention formula:

   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

   where Q = queries, K = keys, V = values, d_k = dimension of keys

2. **Feed-Forward Network (FFN)**
   - Two linear transformations with a nonlinearity:

   $$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$$

   - Modern LLMs often use **SwiGLU** activation instead of GELU/ReLU:

   $$\text{SwiGLU}(x) = (\text{Swish}(W_1 x)) \otimes (W_3 x)$$

3. **Layer Normalization**
   - Modern LLMs use **RMSNorm** (Root Mean Square Layer Normalization) and **pre-norm** (applying norm before attention/FFN rather than after)

   $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \cdot \gamma$$

4. **Residual Connections**
   - Output of each sublayer: x + Sublayer(Norm(x))

### Positional Encoding in Modern LLMs
- Original Transformer used sinusoidal positional encodings
- Modern LLMs use **Rotary Position Embeddings (RoPE)**:
  - Encode position by rotating the query and key vectors
  - Enables relative position encoding
  - Better extrapolation to longer sequences than absolute positional encodings

$$\text{RoPE}(x_m, m) = x_m e^{im\theta}$$

where m is the position and theta is a frequency parameter.

### Tokenization
- LLMs operate on **tokens**, not characters or words
- Common tokenizers: **Byte Pair Encoding (BPE)**, SentencePiece, WordPiece
- Typical vocabulary size: 32K–128K tokens
- BPE algorithm: iteratively merges the most frequent pair of adjacent tokens

### Full LLM Pipeline
```
Input Text
    |
    v
Tokenizer  -->  Token IDs  -->  Token Embeddings + Positional Encoding
    |
    v
[Decoder Layer 1] --> [Decoder Layer 2] --> ... --> [Decoder Layer N]
    |
    v
Linear Layer (vocabulary projection)
    |
    v
Softmax  -->  Probability distribution over vocabulary
    |
    v
Sampling / Argmax  -->  Next Token
```

---

## 3. Mixture of Experts (MoE)

### Motivation
- Scaling model parameters without proportionally scaling compute at inference time
- Key insight: not all parameters need to be active for every input token

### How MoE Works
In an MoE layer, the standard FFN is replaced with multiple "expert" FFN sub-networks and a **gating/router network**:

$$y = \sum_{i=1}^{N} G(x)_i \cdot E_i(x)$$

where:
- N = total number of experts
- E_i(x) = output of expert i
- G(x)_i = gating weight for expert i (from the router)

### Router / Gating Network
The router is a small network that decides which experts to activate:

$$G(x) = \text{TopK}(\text{softmax}(W_g \cdot x))$$

- Typically **top-2** experts are selected per token (sparse activation)
- Only those 2 experts' FFNs are actually computed

### Key MoE Properties
- **Sparse activation**: Only a fraction of parameters are active per forward pass
- **Total parameters vs. active parameters**: e.g., Mixtral 8x7B has ~47B total parameters but only ~13B active per token
- **Efficiency**: Near-constant compute cost regardless of total expert count (for fixed top-k)

### Load Balancing
- Problem: router may learn to always route to same experts ("expert collapse")
- Solution: **auxiliary load balancing loss** that encourages uniform expert utilization

$$\mathcal{L}_{\text{balance}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

where f_i = fraction of tokens routed to expert i, P_i = average routing probability for expert i

### Notable MoE Models
| Model | Experts | Active | Total Params |
|-------|---------|--------|-------------|
| Mixtral 8x7B | 8 | 2 | ~47B |
| Mixtral 8x22B | 8 | 2 | ~141B |
| GPT-4 (rumored) | ~16 | 2 | >1T |
| DeepSeek-V2 | 160 | 6 | 236B |
| DBRX | 16 | 4 | 132B |

---

## 4. Context Length

### Definition
**Context length** (or context window) is the maximum number of tokens the model can process in a single forward pass. It determines how much text the model can "see" at once.

### Why Context Length Matters
- Limits how much information (documents, conversation history, examples) can fit in a prompt
- Longer context enables: longer document analysis, multi-turn conversations, retrieval-augmented generation with more retrieved passages, complex few-shot prompting

### Context Length of Notable Models
| Model | Context Length |
|-------|--------------|
| GPT-2 | 1,024 tokens |
| GPT-3 | 2,048 tokens |
| GPT-3.5-turbo | 4,096 / 16,384 tokens |
| GPT-4 | 8,192 / 32,768 / 128K tokens |
| Claude 2 | 100K tokens |
| Claude 3 | 200K tokens |
| Gemini 1.5 Pro | 1M / 2M tokens |
| LLaMA 2 | 4,096 tokens |
| LLaMA 3 | 8,192–128K tokens |

### Challenges with Long Context
1. **Quadratic attention cost**: Self-attention is O(n^2) in sequence length
   - For n tokens: compute and memory scale as n^2
2. **"Lost in the middle"** phenomenon: Models often attend better to information at the beginning and end of the context, performing worse on information in the middle
3. **Positional encoding extrapolation**: Models trained on shorter sequences may not generalize to longer ones without modifications

### Techniques to Extend Context Length
- **RoPE scaling** (NTK-aware interpolation): Scale the frequency basis of RoPE to extrapolate to longer sequences
- **ALiBi (Attention with Linear Biases)**: Add a linear bias based on distance, no learned positional embeddings
- **Sliding window attention**: Attend only to a local window (used in Mistral)
- **Sparse attention patterns**: Attend to a subset of positions (e.g., Longformer)
- **Ring attention**: Distribute attention computation across devices for very long sequences

---

## 5. Sampling Strategies

### The Core Problem
After the model computes a probability distribution over the vocabulary for the next token, how do we select the actual token?

$$P(x_t \mid x_{<t}) = \text{softmax}(z_t / T)$$

where z_t is the logit vector and T is the temperature.

### Greedy Decoding (Argmax)
- Select the token with the highest probability: x_t = argmax P(x_t | x_{<t})
- **Deterministic**: always produces the same output
- Problem: can be repetitive, lacks diversity, may miss globally optimal sequences

### Temperature Scaling

$$P(x_t \mid x_{<t}) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- **T = 1.0**: Standard softmax (no modification)
- **T < 1.0**: Sharper distribution, more confident/deterministic, more "focused"
- **T > 1.0**: Flatter distribution, more random/diverse, more "creative"
- **T -> 0**: Approaches greedy decoding (argmax)
- **T -> infinity**: Approaches uniform random sampling

**Intuition:** Temperature controls the "confidence" of the model. Low temperature amplifies differences between probabilities; high temperature smooths them out.

### Top-k Sampling
1. Sort tokens by probability
2. Keep only the top k tokens
3. Renormalize probabilities over those k tokens
4. Sample from the renormalized distribution

**Problem:** Fixed k is not adaptive. For some distributions, top-5 covers 95% of probability mass; for others, it covers only 30%.

### Top-p (Nucleus) Sampling
(Holtzman et al., 2020)

1. Sort tokens by probability in descending order
2. Find the smallest set of tokens whose cumulative probability >= p
3. Renormalize probabilities over that set
4. Sample from the renormalized distribution

$$\sum_{x \in V_p} P(x \mid x_{<t}) \geq p$$

where V_p is the smallest vocabulary subset meeting the threshold.

- **Adaptive**: The number of tokens considered varies based on the distribution
- **p = 0.9** (common default): Consider the smallest set of tokens whose probabilities sum to 0.9
- **p = 1.0**: No filtering (equivalent to pure sampling with temperature)

### Combining Strategies
In practice, these strategies are often combined:
- Apply temperature first (modify logits)
- Then apply top-k filtering
- Then apply top-p filtering
- Then sample from remaining distribution

### Repetition Penalty
- Reduce the probability of tokens that have already appeared in the generated text
- Prevents degenerate repetitive loops

### Beam Search
- Maintain k "beams" (candidate sequences) at each step
- Expand each beam with top candidates, keep overall top k
- More thorough than greedy but more expensive
- Common in machine translation; less common in open-ended generation

### Common Parameter Combinations
| Use Case | Temperature | top-p | top-k |
|----------|------------|-------|-------|
| Factual Q&A | 0.0–0.3 | 0.1–0.5 | 1–10 |
| Creative writing | 0.7–1.0 | 0.9–0.95 | 40–100 |
| Code generation | 0.0–0.2 | 0.1–0.3 | 1–5 |
| Brainstorming | 0.8–1.2 | 0.95 | 50–100 |
| Summarization | 0.0–0.3 | 0.5–0.8 | 10–40 |

---

## 6. Prompting Techniques

### What Is a Prompt?
The **prompt** is the input text provided to the LLM. Prompt engineering is the practice of designing prompts to elicit desired behavior from the model.

### Types of Prompts

#### Zero-Shot Prompting
Provide only the task instruction, no examples:

```
Classify the sentiment of this review as positive or negative.

Review: "This movie was absolutely wonderful, I loved every minute."
Sentiment:
```

#### One-Shot Prompting
Provide a single example:

```
Classify the sentiment of the review.

Review: "Terrible service, would not recommend."
Sentiment: Negative

Review: "This movie was absolutely wonderful, I loved every minute."
Sentiment:
```

#### Few-Shot Prompting
Provide multiple examples (typically 2-10):

```
Classify the sentiment of the review.

Review: "Terrible service, would not recommend."
Sentiment: Negative

Review: "Great product, works as advertised!"
Sentiment: Positive

Review: "It was okay, nothing special."
Sentiment: Neutral

Review: "This movie was absolutely wonderful, I loved every minute."
Sentiment:
```

**Key insight from GPT-3 paper (Brown et al., 2020):** Few-shot performance scales with model size. Larger models are better few-shot learners.

### System Prompts
A **system prompt** sets the behavior, persona, and constraints for the model:

```
System: You are a helpful medical assistant. Always recommend consulting
a doctor for serious symptoms. Never provide drug dosage information.

User: I have a headache, what should I take?
```

System prompts are used to:
- Define role/persona
- Set safety guardrails
- Specify output format
- Provide persistent instructions

### Prompt Engineering Best Practices
1. **Be specific and explicit** about what you want
2. **Provide context** — the model can only work with information in the prompt
3. **Use delimiters** to separate different parts of the prompt (e.g., ```, """, XML tags)
4. **Specify the output format** (JSON, bullet points, table, etc.)
5. **Break complex tasks into steps** (see Chain of Thought below)
6. **Provide examples** when the task is ambiguous
7. **Iterate and refine** — prompt engineering is empirical

### Structured Output Prompting
Request specific output formats:

```
Extract the following information from the text and return as JSON:
- name
- age
- occupation

Text: "Dr. Sarah Chen, 42, is a neurosurgeon at Stanford Medical Center."

Output:
```

---

## 7. Chain of Thought (CoT) Reasoning

### The Problem
LLMs can struggle with tasks requiring multi-step reasoning (arithmetic, logic, word problems) when asked to give a direct answer.

### Chain of Thought Prompting
(Wei et al., 2022)

**Key idea:** Prompt the model to show its step-by-step reasoning before giving a final answer.

#### Standard (No CoT):
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?
A: 11
```

#### With Chain of Thought:
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. He bought 2 cans of 3 balls each,
so 2 x 3 = 6 new balls. 5 + 6 = 11. The answer is 11.
```

### Zero-Shot CoT
(Kojima et al., 2022)

Simply append **"Let's think step by step"** to the prompt:

```
Q: If a train travels at 60 mph for 2.5 hours, how far does it go?
A: Let's think step by step.
```

This simple instruction significantly improves performance on reasoning tasks without needing few-shot examples.

### Few-Shot CoT
Provide examples that demonstrate the reasoning chain:

```
Q: There are 15 trees in the grove. Workers plant trees today, doubling the number.
How many trees are there now?
A: There are 15 trees originally. Doubling means 15 x 2 = 30. The answer is 30.

Q: If there are 3 cars in the parking lot and 2 more arrive, how many are there?
A: There are originally 3 cars. 2 more arrive, so 3 + 2 = 5. The answer is 5.

Q: [Your actual question]
A:
```

### Self-Consistency
(Wang et al., 2023)

An extension of CoT:
1. Sample multiple reasoning chains (using temperature > 0)
2. Extract the final answer from each chain
3. Take the **majority vote** as the final answer

This improves robustness by marginalizing over different reasoning paths.

### Why Does CoT Work?
- Breaks complex problems into simpler sub-problems
- Makes intermediate computations explicit (the model can "use" its own generated tokens as working memory)
- Reduces the effective reasoning "depth" required at each step
- The model's next-token prediction benefits from the tokens it has already generated in the chain

### Tree of Thoughts (ToT)
(Yao et al., 2023)

An extension of CoT:
- Instead of a single chain, explore a **tree** of reasoning paths
- Use search strategies (BFS, DFS) to navigate the tree
- Allow the model to evaluate and backtrack from unpromising paths
- More expensive but better for complex planning/reasoning problems

### ReAct (Reasoning + Acting)
(Yao et al., 2023)

Combine reasoning with action:
```
Thought: I need to find the population of France.
Action: Search("population of France 2024")
Observation: The population of France is approximately 68 million.
Thought: Now I can answer the question.
Answer: France has approximately 68 million people.
```

---

## 8. Key Concepts and Definitions — Summary

| Term | Definition |
|------|-----------|
| **LLM** | Large Language Model — a neural network with billions+ parameters trained on massive text to model P(next token given context) |
| **Autoregressive** | Generating tokens one at a time, each conditioned on all previous tokens |
| **Decoder-only** | Transformer variant using only masked self-attention (causal), used by GPT, LLaMA, etc. |
| **MoE (Mixture of Experts)** | Architecture with multiple expert sub-networks and a router; only a subset of experts are active per token |
| **Context length** | Maximum number of tokens processable in a single forward pass |
| **Temperature** | Scalar T that controls the sharpness of the output probability distribution |
| **Top-k sampling** | Sampling restricted to the k most probable tokens |
| **Nucleus / Top-p sampling** | Sampling restricted to the smallest set of tokens whose cumulative probability exceeds p |
| **Zero-shot** | Prompting with task instruction only, no examples |
| **Few-shot** | Prompting with task instruction plus a few demonstration examples |
| **Chain of Thought (CoT)** | Prompting the model to produce intermediate reasoning steps before the answer |
| **Self-consistency** | Sampling multiple CoT paths and taking majority vote on the answer |
| **RoPE** | Rotary Position Embeddings — encodes position via rotation of query/key vectors |
| **BPE** | Byte Pair Encoding — subword tokenization algorithm |
| **RMSNorm** | Root Mean Square Normalization — a simpler, faster alternative to LayerNorm |
| **SwiGLU** | Activation function combining Swish and Gated Linear Unit, common in modern LLMs |
| **Load balancing loss** | Auxiliary loss in MoE to prevent expert collapse |
| **Lost in the middle** | Phenomenon where models attend less to information in the middle of long contexts |
| **Beam search** | Decoding strategy maintaining k best candidates at each generation step |

---

## 9. Formulas Reference

### Autoregressive Language Modeling
$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})$$

### Self-Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### Multi-Head Attention
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### Temperature-Scaled Softmax
$$P_T(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

### MoE Gating
$$y = \sum_{i=1}^{N} G(x)_i \cdot E_i(x) \quad \text{where } G(x) = \text{TopK}(\text{softmax}(W_g x))$$

### MoE Load Balancing Loss
$$\mathcal{L}_{\text{balance}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

### RMSNorm
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \cdot \gamma$$

### Nucleus Sampling Threshold
$$V_p = \arg\min_{V' \subseteq V} \left| V' \right| \text{ s.t. } \sum_{x \in V'} P(x) \geq p$$

---

## 10. Video / Transcript Notes

**YouTube Link:** https://www.youtube.com/watch?v=Q5baLehv5So

> Note: Transcript content was not available for automated extraction at time of file creation. To supplement these notes:
> 1. Watch the lecture video and add any additional insights below
> 2. Check if YouTube auto-generated captions / transcript are available
> 3. The Stanford course page may have additional materials at https://cme295.stanford.edu/

### Placeholder for Transcript Notes
<!-- Add notes from watching the video here -->

---

## 11. Connections to Other Lectures

- **Lecture 2 (Transformers):** This lecture builds on the transformer architecture covered in Lecture 2, extending it to the full LLM setting with decoder-only design, modern normalization, and positional encoding choices.
- **Upcoming — Training:** Future lectures will cover how these models are actually trained (pretraining, fine-tuning, RLHF), which is distinct from the inference-time concerns (sampling, prompting) covered here.
- **Upcoming — RAG / Tool Use:** The prompting techniques here (especially ReAct) connect to retrieval-augmented generation and tool use in later lectures.

---

*File created for CME 295 Lecture 3 content ingestion. Slide PDF and video transcript should be reviewed to fill any gaps.*
