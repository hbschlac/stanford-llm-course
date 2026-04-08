# CME 295: Large Language Models -- Lecture 1: Transformer

**Date:** September 26, 2025
**Course:** Stanford CME 295 -- Large Language Models
**Video:** [YouTube -- Lecture 1](https://www.youtube.com/watch?v=Ub3GoFaUcds)
**Slides:** [PDF](https://cme295.stanford.edu/slides/fall25-cme295-lecture1.pdf)
**Instructors:** Afshine Amidi & Shervine Amidi

**Topics Covered:**
- NLP landscape and common tasks
- Tokenization (word-level, character-level, subword / BPE)
- Word embeddings (one-hot encoding, Word2Vec, learned embeddings)
- Recurrent Neural Networks (RNNs)
- LSTMs and GRUs
- Sequence-to-sequence models
- Attention mechanism (Query, Key, Value)
- The Transformer architecture ("Attention Is All You Need")
- Evaluation metrics (accuracy, precision, recall, F1, BLEU, ROUGE, perplexity)

---

## 1. NLP Landscape

### What is NLP?

**Natural Language Processing (NLP)** is the subfield of AI concerned with enabling computers to understand, interpret, and generate human language. This lecture surveys the key tasks and then builds up the architecture -- the Transformer -- that now dominates all of them.

### Common NLP Tasks

| Task | Description | Example |
|------|-------------|---------|
| **Text Classification** | Assign a label to a piece of text | Spam detection, topic categorization |
| **Sentiment Analysis** | Determine the emotional tone of text | "This movie was amazing" -> Positive |
| **Named Entity Recognition (NER)** | Identify and classify entities (people, places, organizations) in text | "Barack Obama visited Paris" -> [Person, Location] |
| **Machine Translation** | Convert text from one language to another | English -> French |
| **Question Answering** | Given a question (and optionally a context), produce an answer | "What is the capital of France?" -> "Paris" |
| **Text Generation** | Produce new text given a prompt or context | Autocomplete, story generation |

### Three Broad Model Categories

The slides present three categories of how models handle input and output:

1. **Classification** -- many-to-one: the model reads an entire input sequence and produces a single label (e.g., sentiment analysis)
2. **"Multi"-classification / Tagging** -- many-to-many (same length): the model produces one label per input token (e.g., NER, part-of-speech tagging)
3. **Generation** -- many-to-many (different length): the model reads an input sequence and produces an output sequence of potentially different length (e.g., machine translation)

---

## 2. Tokenization

### Why Tokenize?

Models do not operate on raw text strings. Text must first be broken into discrete units called **tokens** before any numerical processing can happen. The choice of tokenization strategy has a significant impact on vocabulary size, model capacity, and the ability to handle rare or unseen words.

### Tokenization Strategies

#### Word-Level Tokenization

Split on whitespace and punctuation.

```
"A cute teddy bear is reading." -> ["A", "cute", "teddy", "bear", "is", "reading", "."]
```

**Pros:** Intuitive; each token is a recognizable word.
**Cons:** Very large vocabulary (hundreds of thousands of words in English alone); cannot handle misspellings or rare words; out-of-vocabulary (OOV) problem.

#### Character-Level Tokenization

Each character is a separate token.

```
"cute" -> ["c", "u", "t", "e"]
```

**Pros:** Tiny vocabulary (roughly 26 letters + digits + punctuation); no OOV problem.
**Cons:** Sequences become very long; individual characters carry little semantic meaning; harder for models to learn long-range dependencies.

#### Subword Tokenization (BPE -- Byte Pair Encoding)

The dominant modern approach. Starts with individual characters and iteratively merges the most frequent adjacent pairs into new tokens.

```
"tokenization" -> ["token", "ization"]
"unhappiness"  -> ["un", "happi", "ness"]
```

**Pros:** Balances vocabulary size and sequence length; handles rare and unseen words gracefully by decomposing them into known subwords.
**Cons:** Token boundaries do not always align with linguistic morphemes.

### Vocabulary Size Tradeoffs

| Smaller Vocabulary | Larger Vocabulary |
|---|---|
| Longer sequences (more tokens per sentence) | Shorter sequences (fewer tokens per sentence) |
| Each token is simpler / less informative | Each token carries more meaning |
| Better generalization to rare words | May struggle with rare / unseen words (OOV) |
| Typical: 32K--128K tokens for modern LLMs | Word-level could be 100K+ |

### Special Tokens

The lecture's end-to-end example introduces two important special tokens:

- **`[BOS]`** -- Beginning of Sequence: prepended to the start of the output during generation
- **`[EOS]`** -- End of Sequence: appended to the end; tells the model when to stop generating

---

## 3. Word Embeddings

### The Problem: How to Represent Words as Numbers

Neural networks need numerical inputs. The question is: how do you turn words into vectors of numbers in a way that captures their meaning?

### 3.1 One-Hot Encoding (Naive Approach)

Each word is represented as a vector of length V (the vocabulary size), with a 1 in the position corresponding to that word and 0s everywhere else.

**Example** (vocabulary: {soft, teddy bear, book}):
```
soft       = (1, 0, 0)
teddy bear = (0, 1, 0)
book       = (0, 0, 1)
```

**Why it fails:** Every pair of words is equally "distant" from every other pair. The dot product between any two different one-hot vectors is always 0:

$$\langle \text{teddy bear}, \text{book} \rangle = 0$$
$$\langle \text{teddy bear}, \text{soft} \rangle = 0$$

This means the representation encodes **no semantic similarity** whatsoever. "Soft" is just as unrelated to "teddy bear" as "book" is -- which is obviously wrong.

### 3.2 Learned Embeddings

Instead of sparse one-hot vectors, we learn a dense, low-dimensional vector for each word. Words that appear in similar contexts end up with similar vectors.

**Example** (same vocabulary, with learned 3-dimensional embeddings):
```
soft       = (0.95, 0.32, 0.01)
teddy bear = (0.89, 0.45, 0.12)
book       = (0.10, 0.85, 0.70)
```

Now the dot products reflect meaning:
$$\langle \text{teddy bear}, \text{book} \rangle \approx 0$$
$$\langle \text{teddy bear}, \text{soft} \rangle \approx 1$$

The slide shows this visually: in the one-hot space, the word arrows are all perpendicular (orthogonal). In the learned embedding space, "soft" and "teddy bear" point in similar directions, while "book" and "Persian poetry" cluster together in a different region.

### 3.3 Word2Vec

**Word2Vec** (Mikolov et al., 2013) is a neural network that learns embeddings via a **proxy task** over billions of words of text. The key insight: you do not need labels -- the structure of language itself provides the training signal.

#### Proxy Tasks

Word2Vec trains with one of two proxy tasks:

- **CBOW (Continuous Bag of Words):** Given the surrounding context words, predict the middle word.
  ```
  Context: [A, ___, teddy, bear, is, reading]  ->  Predict: "cute"
  ```

- **Skip-gram:** Given one word, predict the surrounding context words.
  ```
  Input: "teddy bear"  ->  Predict: [A, cute, is, reading]
  ```

#### Architecture

A simple three-layer neural network:

| Layer | Size | Description |
|-------|------|-------------|
| **Input** | V (vocabulary size) | One-hot encoded input word |
| **Hidden** | d (embedding dimension, e.g. 2--300) | This layer IS the embedding |
| **Output** | V (vocabulary size) | Predicted word probabilities (softmax) |

#### Step-by-Step Example (Predicting Next Word)

The slides walk through predicting each word in "A cute teddy bear is reading" one at a time:

1. **Input "A"** as one-hot vector `[1,0,0,0,0,0]` -> hidden layer produces embedding `[0.2, 0.9]` -> output layer produces probability distribution `[0.2, 0.4, 0.1, 0.1, 0.1, 0.1]` -> highest probability at position 2 ("cute")
2. **Input "cute"** as one-hot vector `[0,1,0,0,0,0]` -> hidden layer produces embedding `[0.8, 0.4]` -> output layer produces `[0.2, 0.2, 0.2, 0.1, 0.2, 0.1]` -> predicts "teddy bear"
3. Continue for each word in the sequence...

The hidden layer weights -- after training on billions of sentences -- become the embedding matrix. Words that frequently appear in similar contexts develop similar hidden-layer representations.

#### Resulting Embedding Space

After training, semantically related words cluster together in the embedding space. The slides show "soft" and "teddy bear" near each other, while "Persian poetry" and "art" cluster in a separate region.

### Similarity Metrics

Once words are embedded as vectors, we can measure similarity:

- **Dot product:** $\langle w_1, w_2 \rangle = \sum_i w_{1,i} \cdot w_{2,i}$ -- measures alignment; higher = more similar
- **Cosine similarity:** $\frac{\langle w_1, w_2 \rangle}{\|w_1\| \cdot \|w_2\|}$ -- normalized version, ranges from -1 to 1

---

## 4. Recurrent Neural Networks (RNNs)

### Overview

RNNs were first introduced in the 1980s. They are a class of neural networks where connections form a **temporal sequence** -- the network processes one token at a time and passes information forward through a hidden state.

### General Form

At each time step t, the RNN:
1. Takes the current input $x^{\langle t \rangle}$
2. Combines it with the previous hidden state $a^{\langle t-1 \rangle}$
3. Produces a new hidden state $a^{\langle t \rangle}$ and (optionally) an output $y^{\langle t \rangle}$

**Hidden state formula:**

$$a^{\langle t \rangle} = f\left(W_a \cdot a^{\langle t-1 \rangle} + W_x \cdot x^{\langle t \rangle} + b\right)$$

where:
- $W_a$ = weight matrix applied to previous hidden state
- $W_x$ = weight matrix applied to current input
- $b$ = bias term
- $f$ = activation function (typically tanh or ReLU)

### Step-by-Step Example

Processing "A cute teddy bear is reading":

```
Step 1: Input "A"          -> RNN cell -> hidden state a^(1) -> output: "cute"
Step 2: Input "cute"       -> RNN cell (receives a^(1)) -> a^(2) -> output: "teddy bear"
Step 3: Input "teddy bear" -> RNN cell (receives a^(2)) -> a^(3) -> output: "is"
Step 4: Input "is"         -> RNN cell (receives a^(3)) -> a^(4) -> output: "reading"
```

Each step passes the hidden state forward, carrying information about all words seen so far. The initial hidden state $a^{\langle 0 \rangle}$ is typically initialized to zeros.

### RNN Architectures for Different Tasks

| Architecture | Input -> Output | Task | Example |
|---|---|---|---|
| **Many-to-One** | Sequence -> single label | Classification / Sentiment | "Great movie!" -> Positive |
| **Many-to-Many (same length)** | Sequence -> sequence (same length) | Tagging / NER | Each word gets a tag |
| **Many-to-Many (different length)** | Sequence -> sequence (different length) | Translation / Generation | Source sentence -> Target sentence |

### The Vanishing Gradient Problem

RNNs suffer from a critical limitation: during training via backpropagation through time, gradients must flow backwards through every time step. When sequences are long, gradients get multiplied by the weight matrix repeatedly, causing them to either:

- **Vanish** (shrink toward zero) -- the network cannot learn long-range dependencies
- **Explode** (grow uncontrollably) -- training becomes unstable

In practice, basic RNNs struggle to connect information that is more than about 10--20 steps apart. This motivates the development of gated architectures.

---

## 5. LSTMs and GRUs

### 5.1 Long Short-Term Memory (LSTM)

Introduced in "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997). LSTMs solve the vanishing gradient problem by adding a **cell state** $c^{\langle t \rangle}$ that runs alongside the hidden state, plus a system of **gates** that control information flow.

#### LSTM Gates

| Gate | Symbol | Purpose | Activation |
|------|--------|---------|-----------|
| **Forget gate** | $\Gamma_f$ | Decides what to **erase** from the cell state | Sigmoid (0 = forget, 1 = keep) |
| **Update gate** | $\Gamma_u$ | Decides what **new information** to store in the cell state | Sigmoid |
| **Relevance gate** | $\Gamma_r$ | Determines how much the **previous memory** matters for computing the candidate state | Sigmoid |
| **Output gate** | $\Gamma_o$ | Controls what part of the cell state to **output** as the hidden state | Sigmoid |

#### How It Works (Intuition)

Think of the LSTM as a selective note-taker:

1. **Forget gate** reads the old notes and decides which parts are no longer relevant
2. **Update gate** decides what new information from the current input is worth writing down
3. A **candidate value** $\tilde{c}^{\langle t \rangle}$ is computed from the current input and previous hidden state
4. The **cell state** is updated: old information (filtered by forget gate) + new information (filtered by update gate)
5. **Output gate** decides what part of the updated cell state to actually use as the output

The cell state acts as a conveyor belt that can carry information across many time steps with minimal modification, solving the vanishing gradient problem.

### 5.2 Gated Recurrent Unit (GRU)

The GRU is a **simplified version** of the LSTM that combines the forget and update gates into a single gate and merges the cell state and hidden state. It has fewer parameters and is faster to train, while achieving comparable performance on many tasks.

### Summary: Word2Vec vs. RNNs

| Method | Pros | Cons |
|--------|------|------|
| **Word2Vec** (CBOW, Skip-gram) | Very simple yet powerful; intuitive embeddings | Word order does not count; embeddings not context-aware |
| **Recurrent Neural Networks** (RNN, LSTM) | Word order matters; state-of-the-art results (pre-Transformer) | Vanishing gradient problem; slow computations (sequential, not parallelizable) |

---

## 6. Sequence-to-Sequence Models

### Encoder-Decoder Architecture

For tasks where input and output sequences have different lengths (e.g., machine translation), the **sequence-to-sequence (seq2seq)** architecture uses two RNNs:

1. **Encoder:** Reads the entire input sequence and compresses it into a fixed-length context vector (the final hidden state)
2. **Decoder:** Takes the context vector and generates the output sequence one token at a time

**Example -- English to French translation:**
```
Encoder input:  [A, cute, teddy bear, is, reading]
                     |
              context vector (final hidden state)
                     |
Decoder output: [Un, ours en peluche, mignon, lit]
```

### The Bottleneck Problem

The entire input sequence must be compressed into a single fixed-length vector. For long sentences, this creates an information bottleneck -- the decoder loses access to details from early in the input. This limitation directly motivates the attention mechanism.

---

## 7. Attention Mechanism

### History

- Introduced in 2014 by Bahdanau et al. ("Neural Machine Translation by Jointly Learning to Align and Translate")
- Motivated by the failure of seq2seq models to handle **long-term dependencies** -- the decoder could not "remember" what the input sentence was saying
- Key idea: instead of relying on a single compressed vector, let the decoder **look back** at all encoder hidden states and focus on the most relevant ones at each decoding step

### The Query, Key, Value Framework

The attention mechanism is built on three concepts:

| Component | Symbol | Plain English |
|-----------|--------|---------------|
| **Query (Q)** | $q$ | "What am I looking for?" -- the current word asking a question |
| **Key (K)** | $k$ | "What does each word offer?" -- every word advertising what it knows |
| **Value (V)** | $v$ | "What is the actual content?" -- the real information each word carries |

#### Step-by-Step (Using "teddy bear" as Query)

The slides build up the QKV concept visually using the sentence "a cute teddy bear is reading .":

1. **"teddy bear"** generates a query vector $q_{\text{teddy bear}}$: "Who is related to me?"
2. Every word generates a key vector: $k_a^T$, $k_{\text{cute}}^T$, $k_{\text{teddy bear}}^T$, $k_{\text{is}}^T$, $k_{\text{reading}}^T$, $k_.^T$
3. The query is compared against all keys via dot product to get attention scores
4. Every word also generates a value vector: $v_a$, $v_{\text{cute}}$, $v_{\text{teddy bear}}$, $v_{\text{is}}$, $v_{\text{reading}}$, $v_.$
5. The attention scores determine how much of each value to include in the output

### The Attention Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Breaking this down:

| Step | Operation | Purpose |
|------|-----------|---------|
| 1 | $QK^T$ | Compute raw attention scores (dot product of each query with every key) |
| 2 | $\div \sqrt{d_k}$ | **Scale** the scores to prevent them from getting too large (which would make softmax produce near-one-hot outputs) |
| 3 | $\text{softmax}(\cdot)$ | Convert scores into a **probability distribution** that sums to 1 |
| 4 | $\times V$ | Compute a **weighted average** of the value vectors |

**What is softmax?** A function that takes any set of numbers and converts them into probabilities between 0 and 1 that all add up to 1. It amplifies large values and suppresses small ones.

**Why divide by $\sqrt{d_k}$?** When the dimension $d_k$ is large, dot products tend to grow in magnitude. Large inputs to softmax produce very peaked distributions (close to one-hot), which kills gradients. Dividing by $\sqrt{d_k}$ keeps the values in a well-behaved range.

### Self-Attention

When Q, K, and V all come from the **same** sequence, it is called **self-attention**. Each word in a sentence attends to every other word in the same sentence, producing context-aware representations.

This is the key innovation that enables the Transformer: unlike RNNs, self-attention can connect **any two positions** in the sequence in a single step, regardless of distance.

### Scaled Dot-Product Attention (Diagram)

From the original paper, the computation flow is:

```
Q, K, V (inputs)
    |
MatMul (Q * K^T)
    |
Scale (/ sqrt(d_k))
    |
Mask (optional -- used in decoder to prevent looking at future tokens)
    |
SoftMax
    |
MatMul (* V)
    |
Output (attention-weighted values)
```

### Multi-Head Attention

Instead of computing a single attention function, **multi-head attention** runs h attention heads in parallel, each with different learned projection matrices. This allows the model to jointly attend to information from different representation subspaces at different positions.

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O

where head_i = Attention(Q * W_Q^i, K * W_K^i, V * W_V^i)
```

**Benefits:**
- Enables the model to capture **different types of relationships** simultaneously (e.g., one head might focus on syntactic relationships, another on semantic ones)
- Analogous to using multiple filters in a convolutional layer in computer vision

---

## 8. The Transformer Architecture

### Overview

- Introduced in the 2017 paper **"Attention Is All You Need"** (Vaswani et al.)
- Relies entirely on the **self-attention** mechanism -- no recurrence, no convolutions
- Achieved **state-of-the-art results** on machine translation tasks
- The foundational architecture behind all modern LLMs (GPT, BERT, LLaMA, Claude, etc.)

### High-Level Structure

The Transformer follows an **encoder-decoder** architecture:

```
Input Text -> [Encoder] -> encoded representation -> [Decoder] -> Output Text
```

Both the encoder and decoder are stacks of identical layers, each containing attention and feed-forward sublayers.

### 8.1 Input Embedding

**What happens:**
- Text is **tokenized** into a sequence of token IDs
- Each token ID is mapped to a **learned embedding vector** of dimension `d_model`

**Parameters:**
- V: vocabulary size
- `d_model`: embedding dimensions (e.g., 512 in the original paper)

### 8.2 Positional Encoding

**The problem:** Self-attention is permutation-invariant -- it has no built-in notion of word order. "The cat sat on the mat" and "mat the on sat cat the" would produce the same attention scores.

**The solution:** Add **positional encoding** vectors to the input embeddings so the model can distinguish positions.

**Two approaches:**
- **Hardcoded (sinusoidal):** The original Transformer uses sine and cosine functions of different frequencies:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

where pos = position in the sequence and i = dimension index.

- **Learned:** Train the positional encoding vectors as parameters (used in many modern models)

**Goal:** Let the model understand the **relative position** of tokens in the input. The sinusoidal formulation has a useful property: the dot product between positional encodings at nearby positions is high, and it decreases for distant positions.

### 8.3 Encoder

Each encoder layer contains:

1. **Multi-Head Self-Attention (MHA):** Every token attends to every other token in the input
2. **Feed-Forward Neural Network (FFNN):** Applied independently to each position
3. **Layer Normalization:** Normalizes activations for stable training
4. **Residual Connections:** Output of each sublayer is $x + \text{Sublayer}(x)$, allowing gradients to flow directly

**Parameters:**
- N: number of layers stacked (e.g., 6 in the original paper)
- h: number of attention heads (e.g., 8)
- `d_FF`: feed-forward hidden dimension (e.g., 2048)
- `d_key`, `d_value`: dimensions for keys and values
- `d_model`: embedding dimension

### 8.4 Decoder

The decoder is similar to the encoder but with two key differences:

1. **Masked Multi-Head Self-Attention:** Tokens can only attend to previous positions (not future ones), enforcing the autoregressive property
2. **Encoder-Decoder Attention:** Queries come from the decoder, but keys and values come from the encoder output -- this is how the decoder "looks at" the input

Each decoder layer contains:
- Masked Multi-Head Self-Attention
- Encoder-Decoder Multi-Head Attention
- Feed-Forward Neural Network
- Layer Normalization + Residual Connections

**Output "shifted right":** During training, the decoder input is the target sequence shifted right by one position, starting with `[BOS]`. This way, at each position, the model predicts the next token.

### 8.5 Output Layer

At the top of the decoder:
1. **Linear projection:** Maps the decoder output to a vector of size V (vocabulary)
2. **Softmax:** Converts to a probability distribution over the vocabulary
3. The token with the highest probability is selected as the prediction

This is essentially a **classification problem** where the "class" is the next word in the vocabulary.

### 8.6 Computational Tricks

#### Multi-Head Attention (Recap)

Running multiple self-attention layers in parallel captures different attention features simultaneously.

#### Label Smoothing

A regularization technique from a 2015 vision paper:
- Problem: models can become **overconfident**, assigning near-100% probability to a single class
- Solution: replace hard labels with smoothed labels

$$q(k|x) = \delta_{k,y} \quad \longrightarrow \quad q'(k|x) = (1 - \epsilon)\delta_{k,y} + \epsilon \cdot u(k)$$

where $\epsilon$ is a small smoothing parameter and $u(k)$ is a uniform distribution.

**Benefits:** Prevents overfitting; improves accuracy and BLEU score.

### 8.7 Original Transformer Parameters (Vaswani et al., 2017)

| Parameter | Value |
|-----------|-------|
| Encoder layers (N) | 6 |
| Decoder layers (N) | 6 |
| Model dimension (d_model) | 512 |
| Feed-forward dimension (d_FF) | 2048 |
| Attention heads (h) | 8 |
| Key/Value dimension (d_k = d_v) | 64 |

---

## 9. End-to-End Example

The slides conclude with a detailed walkthrough of translating "A cute teddy bear is reading." through the full Transformer pipeline:

### Step 1: Tokenization
```
"A cute teddy bear is reading." -> [A, cute, teddy bear, is, reading, .]
```

### Step 2: Add Special Tokens
```
[BOS, A, cute, teddy bear, is, reading, ., EOS]
```

### Step 3: Embedding
Each token is mapped to a learned embedding vector of dimension d_model.

### Step 4: Positional Encoding
Position information is added to each embedding, producing **position-aware embeddings**.

### Step 5: Create Position-Aware Embeddings Matrix
All position-aware embedding vectors are stacked into a matrix (one row per token).

### Step 6: Encoder Processing

Inside the encoder:

1. The position-aware embeddings matrix is multiplied by three learned weight matrices ($W_Q$, $W_K$, $W_V$) to produce Q, K, and V matrices
2. Self-attention is computed: $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$
3. The QK^T matrix contains all pairwise attention scores:

```
       [BOS]  A    cute  teddy bear  is  reading  .  [EOS]
[BOS]  <q,k>  <q,k> ...
A      <q,k>  <q,k> ...
cute   <q,k>  ...        ...
...
```

Each entry $\langle q_i, k_j \rangle$ measures how much token $i$ should attend to token $j$.

4. Multiplying by V produces a **weighted average of values** -- each token's representation now incorporates information from all other tokens, weighted by relevance
5. For multi-head attention: this is repeated h times with different weight matrices, then concatenated and projected through $W_O$
6. The result passes through a **feed-forward network**
7. **Residual connections** and **layer normalization** are applied after each sublayer

### Step 7: Decoder Processing

The decoder generates the output (e.g., French translation) one token at a time, using:
- Masked self-attention on the output tokens generated so far
- Cross-attention to the encoder output (keys and values from encoder, queries from decoder)
- Feed-forward network + normalization

### Step 8: Output

The final decoder hidden states are projected to vocabulary size via a linear layer, then softmax produces probabilities over the vocabulary. The highest-probability token is selected at each position.

---

## 10. Evaluation Metrics

### For Classification Tasks

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **Accuracy** | $\frac{\text{correct predictions}}{\text{total predictions}}$ | Overall correctness |
| **Precision** | $\frac{TP}{TP + FP}$ | Of everything predicted positive, how many actually were? (Avoids false alarms) |
| **Recall** | $\frac{TP}{TP + FN}$ | Of everything actually positive, how many did the model catch? (Avoids missing things) |
| **F1 Score** | $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean of precision and recall -- a single balanced metric |

Where: TP = True Positives, FP = False Positives, FN = False Negatives.

### For Generation Tasks

| Metric | What It Measures |
|--------|------------------|
| **BLEU** (Bilingual Evaluation Understudy) | Quality of machine-translated text. Measures n-gram precision: how many n-grams in the output appear in the reference. Similar in spirit to precision. |
| **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) | Quality of generated text (often used for summarization). Measures n-gram recall: how many n-grams in the reference appear in the output. Similar in spirit to recall. |
| **Perplexity** | How "surprised" the model is by a sequence. Formally: $\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^N \log P(x_i)\right)$. **Lower is better** -- the model found the text natural and predictable. |

---

## The Big Picture

This entire lecture tells one story -- a progression from simple to sophisticated:

```
One-Hot Encoding           Words are just IDs. No meaning.
       |
       v
Word2Vec Embeddings        Words have meaning based on context.
       |
       v
RNNs / LSTMs               Words have order and memory.
       |
       v
Attention Mechanism        Words can look at ALL other words at once.
       |
       v
Transformer                Combine attention + position + scale = state of the art.
```

Everything builds toward the central formula:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### Key Papers Referenced

| Paper | Authors | Year | Contribution |
|-------|---------|------|-------------|
| "Efficient Estimation of Word Representations in Vector Space" | Mikolov et al. | 2013 | Word2Vec |
| "Long Short-Term Memory" | Hochreiter & Schmidhuber | 1997 | LSTM architecture |
| "Neural Machine Translation by Jointly Learning to Align and Translate" | Bahdanau et al. | 2014 | Attention mechanism |
| "Attention Is All You Need" | Vaswani et al. | 2017 | The Transformer |

---

*Source: CME 295 -- Transformers & Large Language Models, Stanford University. Instructors: Afshine Amidi & Shervine Amidi.*
*Figures adapted from "VIP cheatsheets for Stanford's CS 230" and "Super Study Guide: Transformers & Large Language Models" by Amidi.*
