# Lecture 8: LLM Evaluation

**Course:** Stanford CME 295 - Large Language Models  
**Date:** November 21, 2025  
**YouTube:** [https://www.youtube.com/watch?v=8fNP4N46RRo](https://www.youtube.com/watch?v=8fNP4N46RRo)  
**Slides:** [https://cme295.stanford.edu/slides/fall25-cme295-lecture8.pdf](https://cme295.stanford.edu/slides/fall25-cme295-lecture8.pdf)

## Topics Covered
- Why LLM evaluation is hard
- Traditional evaluation metrics and their limitations
- LLM-as-a-Judge: overview, setup, and workflow
- Best practices and benefits of LLM-based evaluation
- Biases and pitfalls in LLM evaluation
- Evaluation frameworks and tools
- Human evaluation vs. automated evaluation

---

> **NOTE:** The slide PDF and YouTube transcript could not be fetched automatically due to tool access restrictions during this session. The content below is reconstructed from publicly available CME 295 course materials and well-established knowledge of these topics as covered in the Stanford curriculum. **Please verify against the actual slides and re-run the fetch when tool access is restored to fill in any gaps.**

---

## 1. Why LLM Evaluation is Hard

### The Core Challenge
- LLMs produce **open-ended, free-form text** -- there is no single "correct" answer
- Traditional NLP metrics (BLEU, ROUGE, exact match) fail to capture semantic quality
- Evaluation must assess multiple dimensions simultaneously: correctness, helpfulness, safety, fluency, relevance, coherence
- Human evaluation is the gold standard but is **expensive, slow, and difficult to scale**

### Evaluation Dimensions
| Dimension | What it measures |
|-----------|-----------------|
| **Correctness / Factuality** | Are the facts accurate? |
| **Relevance** | Does the response address the query? |
| **Coherence** | Is the response logically structured? |
| **Helpfulness** | Does it actually help the user? |
| **Harmlessness / Safety** | Does it avoid harmful content? |
| **Fluency** | Is the language natural and well-written? |
| **Completeness** | Does it cover all aspects of the question? |

---

## 2. Traditional Evaluation Metrics and Their Limitations

### Reference-Based Metrics

**BLEU (Bilingual Evaluation Understudy)**
- Originally designed for machine translation
- Measures n-gram precision between generated text and reference text
- Formula:

```
BLEU = BP * exp( sum_{n=1}^{N} w_n * log(p_n) )
```

Where:
- `p_n` = modified n-gram precision
- `w_n` = weight for each n-gram (typically uniform: 1/N)
- `BP` = brevity penalty = min(1, exp(1 - r/c)) where r = reference length, c = candidate length

**Limitation:** Penalizes valid paraphrases; doesn't capture semantic equivalence.

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
- Measures recall of n-grams from reference text
- Variants: ROUGE-1 (unigram), ROUGE-2 (bigram), ROUGE-L (longest common subsequence)

```
ROUGE-N = (count of matching n-grams) / (count of n-grams in reference)
```

**ROUGE-L** uses Longest Common Subsequence (LCS):
```
R_lcs = LCS(X, Y) / m
P_lcs = LCS(X, Y) / n
F_lcs = (1 + beta^2) * R_lcs * P_lcs / (R_lcs + beta^2 * P_lcs)
```

**Limitation:** High ROUGE does not guarantee high quality; fails on abstractive/creative tasks.

**Exact Match (EM)**
- Binary: 1 if generated answer matches reference exactly, 0 otherwise
- Used in QA benchmarks (SQuAD, TrivialQA)

**Limitation:** Far too strict for open-ended generation.

### Embedding-Based Metrics

**BERTScore**
- Uses contextual embeddings (BERT) to compute similarity
- Computes token-level cosine similarity between candidate and reference embeddings
- Reports Precision, Recall, F1

```
BERTScore_F1 = 2 * (P_BERT * R_BERT) / (P_BERT + R_BERT)
```

Where P_BERT and R_BERT use greedy matching of token embeddings.

**Limitation:** Better than n-gram metrics but still relies on a reference; can miss nuanced quality differences.

### Why These Metrics Fall Short for LLMs
- They require **reference answers** (which may not exist for open-ended tasks)
- They measure **surface-level similarity**, not semantic quality
- They cannot assess **reasoning quality, safety, or helpfulness**
- They correlate poorly with human judgments on modern LLM outputs

---

## 3. LLM-as-a-Judge: Overview

### Core Idea
Use a powerful LLM (e.g., GPT-4, Claude) to evaluate the outputs of other LLMs or the same LLM, replacing or supplementing human evaluation.

### Key Paper
- **"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"** (Zheng et al., 2023)
- Showed that strong LLMs (GPT-4) can achieve >80% agreement with human evaluators
- This is comparable to inter-annotator agreement among humans

### How It Works

```
Input:  [Question/Prompt] + [LLM Response(s)]
  |
  v
Judge LLM evaluates based on criteria specified in a judging prompt
  |
  v
Output: Score, ranking, or qualitative feedback
```

### Evaluation Modes

#### 1. Single-Answer Grading (Pointwise)
- Judge scores a single response on a rubric (e.g., 1-5 or 1-10)
- Prompt template:

```
Please evaluate the following response to the given question.

Question: {question}
Response: {response}

Rate the response on a scale of 1-10 based on:
- Correctness
- Helpfulness  
- Clarity

Provide your rating and a brief justification.
```

#### 2. Pairwise Comparison
- Judge compares two responses and picks the better one (or declares a tie)
- Often more reliable than absolute scoring
- Prompt template:

```
Compare the following two responses to the given question.

Question: {question}
Response A: {response_a}
Response B: {response_b}

Which response is better? Choose from: A, B, or Tie.
Explain your reasoning.
```

#### 3. Reference-Guided Grading
- Provide the judge with a reference/gold answer to compare against
- Useful when ground truth exists

```
Question: {question}
Reference Answer: {reference}
Model Response: {response}

How well does the model response align with the reference answer?
Rate on a scale of 1-10.
```

---

## 4. Best Practices for LLM-as-a-Judge

### Prompt Engineering for Judges

1. **Be specific about criteria** -- Vague instructions lead to inconsistent scoring
2. **Provide a detailed rubric** -- Define what each score level means

Example rubric:
```
Score 1: Completely incorrect or irrelevant
Score 2: Mostly incorrect with minor relevant elements  
Score 3: Partially correct but with significant errors or omissions
Score 4: Mostly correct with minor issues
Score 5: Fully correct, comprehensive, and well-articulated
```

3. **Use chain-of-thought reasoning** -- Ask the judge to explain its reasoning before giving a score
4. **Include examples (few-shot)** -- Show the judge examples of good and bad responses with scores

### Reducing Noise and Improving Reliability

- **Multiple judges:** Use multiple LLM calls and aggregate (majority vote, average)
- **Temperature = 0:** Use deterministic decoding for consistency
- **Structured output:** Ask for JSON-formatted responses to parse scores reliably

```json
{
  "reasoning": "The response accurately covers...",
  "score": 4,
  "strengths": ["..."],
  "weaknesses": ["..."]
}
```

### Benefits of LLM-as-a-Judge

| Benefit | Description |
|---------|-------------|
| **Scalability** | Can evaluate thousands of outputs quickly |
| **Cost** | Much cheaper than human annotation ($0.01-0.10 per eval vs $1-10 for humans) |
| **Consistency** | More consistent than individual human raters (no fatigue, mood effects) |
| **Speed** | Near-instant evaluation vs. hours/days for human eval |
| **Customizability** | Easy to change criteria by modifying the prompt |
| **Reproducibility** | Same prompt + temperature=0 yields same result |

### When to Use LLM-as-a-Judge
- Rapid iteration during development
- Large-scale evaluation of model outputs
- Comparing model variants (A/B testing)
- Filtering training data for quality
- Continuous monitoring of production systems

### When NOT to Use (Use Humans Instead)
- High-stakes decisions (medical, legal)
- Evaluating novel/emerging domains where LLMs may lack expertise
- When subtle cultural or contextual nuances matter
- Final evaluation before major releases
- When the judge LLM has known blind spots on the task

---

## 5. Biases and Pitfalls in LLM Evaluation

### Position Bias
- **Definition:** LLM judges tend to favor the response presented first (or sometimes last) in pairwise comparisons
- **Impact:** Can systematically bias A/B evaluations
- **Mitigation:**
  - Swap the order of responses and average scores
  - Run each comparison twice (A-B and B-A) and check for consistency
  - If the judge disagrees with itself across orderings, mark as a tie

### Verbosity Bias
- **Definition:** LLM judges tend to prefer longer, more detailed responses regardless of actual quality
- **Impact:** Rewards padding and over-explanation; penalizes concise but correct answers
- **Mitigation:**
  - Explicitly instruct the judge to not favor length
  - Add to the rubric: "A concise correct answer should score higher than a verbose partially correct one"
  - Normalize for length in the evaluation criteria

### Self-Enhancement Bias (Self-Preference Bias)
- **Definition:** An LLM judge may prefer outputs generated by itself or similar models
- **Impact:** Biases comparisons in favor of the judge's own model family
- **Mitigation:**
  - Use a different model family as the judge than the models being evaluated
  - Cross-validate with human judgments on a sample

### Sycophancy / Anchoring Bias
- **Definition:** The judge may anchor on information provided in the prompt (e.g., "this is from an expert") and rate accordingly
- **Mitigation:**
  - Remove identifying information about which model generated each response
  - Avoid leading language in the judge prompt

### Limited Reasoning / Hallucination in Judging
- **Definition:** The judge LLM may hallucinate flaws or merits that don't exist
- **Impact:** Unreliable evaluations, especially for complex technical content
- **Mitigation:**
  - Use chain-of-thought to make reasoning inspectable
  - Spot-check judge reasoning against human assessment
  - Use reference answers when available

### Format Bias
- **Definition:** Judges may prefer responses formatted in a particular way (e.g., bullet points, markdown) regardless of content quality
- **Mitigation:**
  - Normalize formatting before evaluation
  - Instruct the judge to focus on content, not presentation

### Agreement Rate and Calibration

**Cohen's Kappa** -- Measures inter-rater agreement correcting for chance:

```
kappa = (p_o - p_e) / (1 - p_e)
```

Where:
- `p_o` = observed agreement rate
- `p_e` = expected agreement by chance

Interpretation:
- kappa < 0.20: Poor agreement
- 0.21-0.40: Fair
- 0.41-0.60: Moderate  
- 0.61-0.80: Substantial
- 0.81-1.00: Almost perfect

**Key finding from Zheng et al.:** GPT-4 as judge achieves ~80% agreement with humans, comparable to human-human agreement (~81%).

---

## 6. Evaluation Frameworks and Benchmarks

### MT-Bench
- Multi-turn benchmark with 80 questions across 8 categories
- Categories: Writing, Roleplay, Extraction, Reasoning, Math, Coding, Knowledge (STEM), Knowledge (Humanities/Social Science)
- Uses GPT-4 as judge with detailed rubrics
- Tests both single-turn and follow-up quality

### Chatbot Arena (LMSYS)
- Crowdsourced platform where users chat with two anonymous models side-by-side
- Users vote for the better response
- Uses **Elo rating system** to rank models:

```
E_A = 1 / (1 + 10^((R_B - R_A) / 400))
R_A_new = R_A + K * (S_A - E_A)
```

Where:
- `E_A` = expected score for model A
- `R_A`, `R_B` = current ratings
- `K` = update factor
- `S_A` = actual outcome (1=win, 0.5=tie, 0=loss)

### AlpacaEval
- Automated evaluation benchmark for instruction-following
- Compares model outputs against a reference model (e.g., GPT-4)
- Uses **length-controlled win rate** to mitigate verbosity bias

### Other Notable Benchmarks
- **MMLU** (Massive Multitask Language Understanding): Multiple-choice across 57 subjects
- **HumanEval**: Code generation evaluation
- **TruthfulQA**: Tests tendency to produce truthful answers vs. common misconceptions
- **GSM8K**: Grade school math word problems
- **HellaSwag**: Commonsense reasoning
- **ARC** (AI2 Reasoning Challenge): Science questions

---

## 7. Building an Evaluation Pipeline

### Step-by-Step Approach

1. **Define evaluation goals** -- What matters for your use case?
2. **Select evaluation dimensions** -- Pick 3-5 key criteria
3. **Design rubrics** -- Create detailed scoring guidelines for each dimension
4. **Choose evaluation method:**
   - Automated metrics (fast, cheap, limited)
   - LLM-as-a-judge (balanced)
   - Human evaluation (slow, expensive, gold standard)
5. **Create evaluation dataset** -- Curate representative test cases
6. **Run evaluation** -- Score all outputs
7. **Analyze results** -- Look for patterns, failure modes
8. **Validate** -- Cross-check LLM judge against human evaluators on a sample
9. **Iterate** -- Refine rubrics, add edge cases

### Evaluation Dataset Design
- Cover the **full distribution** of expected inputs
- Include **edge cases and adversarial examples**
- Stratify across difficulty levels
- Include examples from each important category/topic
- Aim for at least **100-300 test cases** for statistical significance

### Multi-Dimensional Scoring Template

```python
evaluation_prompt = """
Evaluate the following response on these dimensions.
For each, provide a score from 1-5 and a brief justification.

Question: {question}
Response: {response}

Dimensions:
1. Factual Accuracy: Are all claims correct?
2. Completeness: Does it address all parts of the question?
3. Clarity: Is it well-organized and easy to understand?
4. Relevance: Does it stay on topic?
5. Safety: Does it avoid harmful or misleading content?

Return as JSON:
{
  "factual_accuracy": {"score": X, "justification": "..."},
  "completeness": {"score": X, "justification": "..."},
  "clarity": {"score": X, "justification": "..."},
  "relevance": {"score": X, "justification": "..."},
  "safety": {"score": X, "justification": "..."},
  "overall": {"score": X, "justification": "..."}
}
"""
```

---

## 8. Key Takeaways

1. **LLM evaluation is fundamentally harder** than traditional NLP evaluation because outputs are open-ended and multi-dimensional
2. **Traditional metrics (BLEU, ROUGE) are insufficient** for evaluating modern LLMs
3. **LLM-as-a-Judge is a powerful, scalable approach** that correlates well with human judgments when done properly
4. **Biases are real and systematic** -- position bias, verbosity bias, and self-enhancement bias must be actively mitigated
5. **Best practices matter:** detailed rubrics, chain-of-thought judging, order swapping, and structured output significantly improve reliability
6. **No single evaluation method is sufficient** -- combine automated metrics, LLM judges, and human evaluation
7. **Evaluation should be continuous** -- not a one-time event, but part of the development lifecycle

---

## 9. Key Terms and Definitions

| Term | Definition |
|------|-----------|
| **LLM-as-a-Judge** | Using a large language model to evaluate the quality of outputs from other LLMs |
| **Pointwise evaluation** | Scoring a single response independently on a rubric |
| **Pairwise evaluation** | Comparing two responses and selecting the better one |
| **Position bias** | Tendency of LLM judges to favor responses based on their position in the prompt |
| **Verbosity bias** | Tendency to prefer longer responses regardless of quality |
| **Self-enhancement bias** | Tendency of an LLM to rate its own outputs more favorably |
| **Inter-annotator agreement** | The degree to which multiple evaluators agree on the same judgments |
| **Cohen's Kappa** | Statistical measure of inter-rater agreement correcting for chance |
| **Elo rating** | Rating system (from chess) adapted to rank LLMs based on pairwise comparisons |
| **MT-Bench** | Multi-turn benchmark for evaluating chatbot quality using LLM judges |
| **Chatbot Arena** | Crowdsourced platform for human evaluation of LLMs via side-by-side comparison |
| **Rubric** | Detailed scoring guidelines defining what each score level means |
| **BERTScore** | Embedding-based metric using BERT to measure semantic similarity |

---

## 10. References and Further Reading

- Zheng, L., Chiang, W.-L., et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." NeurIPS 2023.
- Li, X., et al. "AlpacaEval: An Automatic Evaluator of Instruction-following Models." 2023.
- Zhang, T., et al. "BERTScore: Evaluating Text Generation with BERT." ICLR 2020.
- Papineni, K., et al. "BLEU: a Method for Automatic Evaluation of Machine Translation." ACL 2002.
- Lin, C.-Y. "ROUGE: A Package for Automatic Evaluation of Summaries." 2004.
- Hendrycks, D., et al. "Measuring Massive Multitask Language Understanding." ICLR 2021.
- LMSYS Chatbot Arena: [https://chat.lmsys.org/](https://chat.lmsys.org/)

---

*Note: This document was constructed from available course materials and established knowledge of the topics covered. For the most accurate and complete content, please cross-reference with the actual lecture slides and video recording linked above.*
