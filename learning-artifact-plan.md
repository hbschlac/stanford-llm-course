# Stanford LLM Course — Interactive Learning Artifact Plan

## Vision
A visual, interactive web app that teaches Stanford's CME 295 (Transformers & LLMs) course content — organized by concept, not by lecture — with visual simulations as the core interaction model. Built for "a smart 12-year-old who wants to learn this but is like 'wtf' after watching Stanford videos."

## Design Decisions

| Decision | Choice |
|----------|--------|
| **Format** | Clicking through — visual, game-like (NOT a chatbot/tutor) |
| **Audience** | Non-technical learner who wants both deep intuition AND practical fluency |
| **Organization** | Concept-by-concept (NOT lecture-by-lecture) |
| **Primary interaction** | Visual simulations (attention heatmaps, token flow, transformer layers) |
| **Platform** | Next.js/React web app deployed to Vercel |

## 9-Concept Learning Path

### Dependency Graph
```
CONCEPT 1: Text as Numbers
         ↓
CONCEPT 2: Attention Mechanism
         ↓
CONCEPT 3: Transformer Architecture
         ↓
CONCEPT 4: Large Language Models
    ↙        ↓         ↘
Concept 5  Concept 6  Concept 7
(Learning) (Alignment) (Reasoning)
    ↘        ↓         ↙
         ↓
CONCEPT 8: Agentic Behavior
         ↓
CONCEPT 9: Evaluation & Trends
```

### Concept 1: Text as Numbers — How Computers Read Words
**Core idea:** Computers don't understand words, only numbers. This is the foundation for everything.

**Subtopics:**
- Tokenization (word-level, character-level, subword/BPE)
- Word embeddings (One-Hot Encoding → Word2Vec → Learned Embeddings)
- Embedding space (semantic similarity, coordinates on a map)

**Source lectures:** L1 (core), L2 (positional embeddings), L4 (BPE training)

**Visualization ideas:** 2D/3D scatter plot of word embeddings; interactive tokenizer that breaks sentences into tokens

---

### Concept 2: Attention — How Models Focus on Relevant Words
**Core idea:** The breakthrough mechanism. Models ask "which words matter for understanding this word?"

**Subtopics:**
- The attention problem: "What does 'it' refer to in this sentence?"
- Query, Key, Value framework
- Self-attention and multi-head attention
- Scaled dot-product attention (simplified)

**Source lectures:** L1 (core), L2 (refined versions)

**Visualization ideas:** Attention heatmap — type a sentence, see which words attend to which

---

### Concept 3: The Transformer Architecture — Putting It All Together
**Core idea:** How tokens flow through the model — what each layer does and why.

**Subtopics:**
- Encoder-Decoder vs Decoder-only
- Masked self-attention (can only look backward)
- Feed-forward networks, residual connections, layer normalization
- Positional encoding

**Source lectures:** L1 (original architecture), L2 (refinements), L3 (decoder-only focus)

**Visualization ideas:** Animated flow diagram of tokens moving through transformer layers

---

### Concept 4: Large Language Models — Scaling Things Up
**Core idea:** What makes an LLM "large"? Why does scale matter?

**Subtopics:**
- Autoregressive language modeling (predict next token)
- Parameters (billions) + training data (trillions of tokens)
- Mixture of Experts (MoE)
- Token sampling strategies (temperature, top-k, nucleus)
- In-context learning

**Source lectures:** L3 (definitions, MoE, sampling), L2 (architectural refinements)

**Visualization ideas:** Parameter/data scale comparison chart; interactive temperature slider showing sampling behavior

---

### Concept 5: How LLMs Learn — From Data to Model
**Core idea:** How do we go from raw text on the internet to a working ChatGPT?

**Subtopics:**
- Pretraining: next-token prediction
- Training data sources and preprocessing
- Hardware (GPUs, distributed training)
- Flash Attention, mixed precision
- Supervised fine-tuning (SFT) and LoRA

**Source lectures:** L4

**Visualization ideas:** Training loop animation (data → tokens → loss → weight updates)

---

### Concept 6: Alignment — Making Models Helpful and Safe
**Core idea:** How do we make models do what we want and not do harmful things?

**Subtopics:**
- The alignment problem
- Human preference data and reward models
- RLHF (Reinforcement Learning from Human Feedback)
- DPO (Direct Preference Optimization)
- Constitutional AI

**Source lectures:** L5

**Visualization ideas:** RLHF pipeline diagram; interactive A/B comparison showing preference ranking

---

### Concept 7: Reasoning & Planning — Beyond Pattern Matching
**Core idea:** How can models solve hard problems that require actual thinking?

**Subtopics:**
- System 1 vs System 2 thinking
- Chain-of-Thought (CoT)
- Reasoning models (o1, R1) and "thinking tokens"
- Test-time compute scaling
- Tree-of-thought planning

**Source lectures:** L3 (CoT), L6 (reasoning models), L7 (tool use + reasoning)

**Visualization ideas:** Chain-of-thought example trace; tree-of-thought branching diagram

---

### Concept 8: Agentic Behavior — Tools, Memory, and Planning
**Core idea:** How can LLMs interact with the world and accomplish complex multi-step goals?

**Subtopics:**
- RAG (Retrieval-Augmented Generation)
- Vector databases and embedding search
- Function calling / tool use
- Memory systems
- Multi-agent systems

**Source lectures:** L7 (core), L6 (planning)

**Visualization ideas:** Agent loop animation with tool calls; RAG retrieval flow diagram

---

### Concept 9: Evaluation & Current Trends — Measuring and Extending
**Core idea:** How do we know if models are good? What's next?

**Subtopics:**
- Traditional metrics (BLEU, ROUGE, perplexity)
- Benchmarks (MMLU, GSM8K, HumanEval)
- LLM-as-a-Judge
- Multimodal models, long context, efficiency
- Scaling laws, open vs closed source

**Source lectures:** L8 (evaluation), L9 (trends)

**Visualization ideas:** Benchmark leaderboard; scaling law curves

---

## Cross-Lecture Coverage Matrix

| Concept | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 |
|---------|----|----|----|----|----|----|----|----|-----|
| Text as Numbers | ★★★ | ★ | · | ★ | · | · | · | · | · |
| Attention | ★★★ | ★★ | ★ | · | · | ★ | ★ | · | ★ |
| Transformer | ★★★ | ★★ | ★ | · | · | · | · | · | ★ |
| LLMs | · | ★ | ★★★ | · | · | · | · | · | · |
| Training | · | · | · | ★★★ | ★ | · | · | · | ★ |
| Alignment | · | · | · | · | ★★★ | ★ | · | · | · |
| Reasoning | · | · | ★ | · | ★ | ★★★ | ★★ | · | · |
| Agents | · | · | · | · | · | · | ★★★ | · | · |
| Eval & Trends | ★ | · | · | · | · | · | · | ★★★ | ★ |

## Content Strategy

- **Concepts 1-4 (Foundations):** "Explain like you're 12" — heavy use of analogies and visuals
- **Concepts 5-6 (How it works):** "Here's the machinery" — interactive diagrams breaking complex pipelines into steps
- **Concepts 7-9 (Applications):** Case studies — "How does Claude solve math problems?", "How does ChatGPT use tools?"

## Source Content Available
All in `/Users/hannahschlacter/Learning-Stanford LLM course/`:
- 9 lecture note MDs (`Lecture_01` through `Lecture_09`) — 17-37KB each
- 9 YouTube transcripts in `transcripts/` — ~2100-2260 lines each
- Hannah's Google Doc notes (`Hannahs_Google_Doc_Notes.md`)
- Course overview index (`Course_Overview.md`)

## Next Steps
1. Initialize Next.js project
2. Design concept navigation UI (visual map with dependency arrows)
3. Build Concept 1 as the prototype (tokenization + embedding space simulations)
4. Deploy to Vercel
5. Iterate on remaining concepts
