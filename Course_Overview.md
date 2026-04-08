# Stanford CME 295: Transformers & Large Language Models

**Instructors:** Afshine Amidi & Shervine Amidi
**University:** Stanford University
**Course Website:** [cme295.stanford.edu](https://cme295.stanford.edu)
**YouTube Playlist:** [Full Course](https://www.youtube.com/playlist?list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy)
**Companion Textbook:** [Super Study Guide: Transformers & LLMs](https://superstudy.guide) (~600 illustrations, 250 pages)
**GitHub Cheatsheets:** [afshinea/stanford-cme-295-transformers-large-language-models](https://github.com/afshinea/stanford-cme-295-transformers-large-language-models)

---

## Course Overview

This course provides a comprehensive introduction to Transformers and Large Language Models (LLMs), from the foundational architecture to current trends. It covers how these models work, how they're trained and aligned, how they reason, and how they're evaluated.

---

## Lectures

| # | Topic | Notes | Slides | Video | Transcript |
|---|-------|-------|--------|-------|------------|
| 1 | **Transformer** | [Notes](Lecture_01_Transformer.md) | [PDF](https://cme295.stanford.edu/slides/fall25-cme295-lecture1.pdf) | [YouTube](https://www.youtube.com/watch?v=Ub3GoFaUcds) | [Transcript](transcripts/Lecture_01_transcript.txt) |
| 2 | **Transformer-Based Models** | [Notes](Lecture_02_Transformer_Models.md) | [PDF](https://cme295.stanford.edu/slides/fall25-cme295-lecture2.pdf) | [YouTube](https://www.youtube.com/watch?v=yT84Y5zCnaA) | [Transcript](transcripts/Lecture_02_transcript.txt) |
| 3 | **Large Language Models** | [Notes](Lecture_03_LLMs.md) | [PDF](https://cme295.stanford.edu/slides/fall25-cme295-lecture3.pdf) | [YouTube](https://www.youtube.com/watch?v=Q5baLehv5So) | [Transcript](transcripts/Lecture_03_transcript.txt) |
| 4 | **LLM Training** | [Notes](Lecture_04_LLM_Training.md) | [PDF](https://cme295.stanford.edu/slides/fall25-cme295-lecture4.pdf) | [YouTube](https://www.youtube.com/watch?v=VlA_jt_3Qc4) | [Transcript](transcripts/Lecture_04_transcript.txt) |
| 5 | **LLM Tuning** | [Notes](Lecture_05_LLM_Tuning.md) | [PDF](https://cme295.stanford.edu/slides/fall25-cme295-lecture5.pdf) | [YouTube](https://www.youtube.com/watch?v=PmW_TMQ3l0I) | [Transcript](transcripts/Lecture_05_transcript.txt) |
| 6 | **LLM Reasoning** | [Notes](Lecture_06_LLM_Reasoning.md) | [PDF](https://cme295.stanford.edu/slides/fall25-cme295-lecture6.pdf) | [YouTube](https://www.youtube.com/watch?v=k5Fh-UgTuCo) | [Transcript](transcripts/Lecture_06_transcript.txt) |
| 7 | **Agentic LLMs** | [Notes](Lecture_07_Agentic_LLMs.md) | [PDF](https://cme295.stanford.edu/slides/fall25-cme295-lecture7.pdf) | [YouTube](https://www.youtube.com/watch?v=h-7S6HNq0Vg) | [Transcript](transcripts/Lecture_07_transcript.txt) |
| 8 | **LLM Evaluation** | [Notes](Lecture_08_LLM_Evaluation.md) | [PDF](https://cme295.stanford.edu/slides/fall25-cme295-lecture8.pdf) | [YouTube](https://www.youtube.com/watch?v=8fNP4N46RRo) | [Transcript](transcripts/Lecture_08_transcript.txt) |
| 9 | **Current Trends** | [Notes](Lecture_09_Current_Trends.md) | [PDF](https://cme295.stanford.edu/slides/fall25-cme295-lecture9.pdf) | [YouTube](https://www.youtube.com/watch?v=Q86qzJ1K1Ss) | [Transcript](transcripts/Lecture_09_transcript.txt) |

---

## Learning Path

The lectures build on each other in this order:

```
Lecture 1: Transformer          -- The foundation: NLP basics, attention, the Transformer architecture
    |
Lecture 2: Transformer Models   -- Variants built on Transformers: BERT, GPT, T5, Vision Transformers
    |
Lecture 3: LLMs                 -- What makes a model "large"; MoE, sampling, prompting, chain-of-thought
    |
Lecture 4: LLM Training         -- How LLMs are pretrained; hardware, distributed training, fine-tuning, LoRA
    |
Lecture 5: LLM Tuning           -- Alignment: RLHF, PPO, DPO; making models helpful and safe
    |
Lecture 6: LLM Reasoning        -- Chain-of-thought, reasoning models (o1, R1), RL for reasoning, GRPO
    |
Lecture 7: Agentic LLMs         -- Tool use, function calling, RAG, multi-agent systems, planning
    |
Lecture 8: LLM Evaluation       -- Benchmarks, human evaluation, automated metrics, Chatbot Arena
    |
Lecture 9: Current Trends       -- Multimodal models, long context, efficiency, open vs closed source
```

---

## Key Concepts by Lecture

### Lecture 1: Transformer
Tokenization (BPE), word embeddings (Word2Vec), RNNs/LSTMs, attention mechanism, self-attention, multi-head attention, positional encoding, encoder-decoder architecture

### Lecture 2: Transformer-Based Models
BERT (masked LM, bidirectional), GPT (autoregressive, decoder-only), T5 (text-to-text), Vision Transformer (ViT), transfer learning, pretraining vs fine-tuning

### Lecture 3: Large Language Models
Foundation models, Mixture of Experts (MoE), context length, temperature/top-k/top-p sampling, zero-shot/few-shot prompting, system prompts, chain-of-thought

### Lecture 4: LLM Training
Next-token prediction, training data (Common Crawl, Wikipedia), hardware (GPUs, H100), distributed training (data/tensor/pipeline parallelism), ZeRO, mixed precision (BF16), LoRA, QLoRA

### Lecture 5: LLM Tuning
Alignment problem, RLHF pipeline, reward models, Bradley-Terry model, PPO (clipping, KL penalty), DPO (direct preference optimization), Constitutional AI

### Lecture 6: LLM Reasoning
System 1 vs System 2, chain-of-thought, self-consistency, reasoning models (o1, R1), test-time compute scaling, RL for reasoning, GRPO, process vs outcome rewards

### Lecture 7: Agentic LLMs
Tool use, function calling, ReAct framework, RAG (retrieval-augmented generation), vector databases, multi-agent systems, planning (tree-of-thought), memory

### Lecture 8: LLM Evaluation
Perplexity, BLEU/ROUGE, benchmarks (MMLU, GSM8K, HumanEval), human evaluation, Chatbot Arena (ELO), contamination, evaluation pitfalls

### Lecture 9: Current Trends
Multimodal models (vision-language), long context windows, efficiency (quantization, distillation, speculative decoding), open vs closed source, scaling laws, safety

---

## Additional Resources

- **Hannah's Google Doc Notes:** Personal lecture notes with "explain like you're 12" study guides
- **Existing Math Breakdown:** `/Users/hannahschlacter/Documents/Claude/Projects/Stanford LLM Course/Lecture1_Math_Breakdown.md`
- **Course Cheatsheets:** Available on the GitHub repo above
