# CME 295: Large Language Models — Lecture 6: LLM Reasoning

**Course:** Stanford CME 295 — Large Language Models
**Video:** [YouTube — Lecture 6](https://www.youtube.com/watch?v=k5Fh-UgTuCo)
**Slides:** [PDF](https://cme295.stanford.edu/slides/fall25-cme295-lecture6.pdf)

**Topics Covered:**
- What is reasoning? System 1 vs System 2 thinking
- Chain-of-Thought (CoT) revisited: zero-shot, few-shot, self-consistency
- Reasoning models (OpenAI o1/o3, DeepSeek R1) and thinking tokens
- Test-time compute scaling: search strategies and verifiers
- Reinforcement learning for reasoning: ORM vs PRM
- GRPO (Group Relative Policy Optimization): DeepSeek's approach
- The "increasing output length" phenomenon and fixes (DAPO, Dr. GRPO)
- Applications: DeepSeek R1-Zero, R1 training recipes, distillation, benchmarks

---

## 1. What Is Reasoning?

### Defining Reasoning in the Context of LLMs

Reasoning is the process of drawing conclusions, making inferences, or solving problems by combining known information through logical steps. For LLMs, reasoning refers to the model's ability to break a complex problem into intermediate steps and arrive at a correct conclusion -- rather than just pattern-matching to produce an answer in one shot.

### System 1 vs System 2 Thinking (Kahneman)

Daniel Kahneman's framework from *Thinking, Fast and Slow* provides a useful lens for understanding LLM reasoning:

| | System 1 | System 2 |
|---|---|---|
| **Speed** | Fast, automatic | Slow, deliberate |
| **Effort** | Low effort, intuitive | High effort, effortful |
| **Examples** | Recognizing a face, reading a word | Solving a math proof, planning a trip |
| **LLM analog** | Standard next-token prediction | Chain-of-thought reasoning |

**Standard LLMs operate like System 1:** they produce outputs quickly in a single forward pass per token, relying on learned patterns. This works well for simple factual recall and fluent text generation, but struggles with multi-step logic, math, and planning.

**Reasoning models aim for System 2:** by generating explicit intermediate reasoning steps (thinking tokens), the model can "slow down" and work through problems methodically. This trades more computation at inference time for better accuracy on hard problems.

### Why Reasoning Is Hard for LLMs

- **Fixed computation per token:** In a standard transformer, each output token requires roughly the same amount of computation (one forward pass through the network). A simple factual answer and a complex logical derivation get the same "thinking budget" per token.
- **No scratchpad by default:** Without chain-of-thought prompting, the model must compute its answer entirely within its hidden states -- it has no external working memory.
- **Training signal is about prediction, not correctness:** LLMs are trained to predict the next token in text, not to verify the logical validity of their reasoning. They can produce plausible-sounding but logically incorrect chains.
- **Compositionality gap:** LLMs may know individual facts but fail to compose them correctly across multiple reasoning steps (e.g., "Alice is taller than Bob, Bob is taller than Carol -- who is shortest?").

---

## 2. Chain-of-Thought (CoT) Revisited

This section recaps material from Lecture 3, extended with context for reasoning models.

### What Is Chain-of-Thought Prompting?

Chain-of-thought (CoT) prompting encourages the model to produce intermediate reasoning steps before giving a final answer. By writing out its "work," the model uses its own output tokens as a form of working memory, effectively giving itself more computation to solve the problem.

### Zero-Shot CoT

The simplest form: append **"Let's think step by step"** to the prompt.

**Example:**
```
Q: If a store has 4 shelves and each shelf holds 8 boxes, and each box
   contains 6 items, how many items are there in total?

A: Let's think step by step.
   - 4 shelves x 8 boxes per shelf = 32 boxes
   - 32 boxes x 6 items per box = 192 items
   The answer is 192.
```

This simple instruction was shown to significantly improve performance on reasoning benchmarks (Kojima et al., 2022, "Large Language Models are Zero-Shot Reasoners").

### Few-Shot CoT

Provide worked examples in the prompt that demonstrate the step-by-step reasoning format. The model then follows the demonstrated pattern for the new question.

**Key insight:** The quality and format of the demonstration examples matters -- the model imitates the reasoning style shown, not just the answer format.

### Self-Consistency (Wang et al., 2023)

Self-consistency improves CoT by sampling multiple reasoning paths and taking a majority vote on the final answer:

1. **Sample** N different chain-of-thought reasoning paths (using temperature > 0)
2. **Extract** the final answer from each path
3. **Take majority vote** across the N answers

**Why it works:** Different reasoning paths may make different errors, but the correct answer tends to appear more frequently. This is a simple form of test-time compute scaling -- spending more inference compute to improve accuracy.

**Analogy:** Like asking a classroom of students to each solve a problem independently, then going with the most common answer.

---

## 3. Reasoning Models

### What Are Reasoning Models?

Reasoning models are LLMs that produce extended **chains of thought (CoT)** before arriving at a final answer. Rather than immediately outputting an answer, these models "think step by step" -- generating intermediate reasoning tokens that break down complex problems. Crucially, these models are **trained** to reason (via RL), not merely prompted to do so.

### Key Examples of Reasoning Models

| Model | Organization | Notes |
|-------|-------------|-------|
| o1, o1-mini, o1-pro | OpenAI | Pioneered the reasoning model paradigm |
| o3, o3-mini, o4-mini | OpenAI | Later iterations with improved capabilities |
| DeepSeek R1 | DeepSeek | Open-weight; detailed training recipe published |
| Claude 3.5 Sonnet (extended thinking) | Anthropic | Extended thinking mode for reasoning tasks |
| Gemini 2.0 Flash Thinking | Google | Thinking-enabled variant |
| QwQ | Alibaba | Open-weight reasoning model |
| Grok 3 (with thinking) | xAI | Thinking mode for complex tasks |

### "Thinking Tokens" / Internal Reasoning Traces

Reasoning models generate a special block of **thinking tokens** before producing the final answer. These tokens represent the model's internal deliberation -- exploring approaches, checking intermediate results, and sometimes backtracking.

**Structure of a reasoning model output:**
```
<think>
Let me break this problem down...
First, I need to find...
Wait, that approach won't work because...
Let me try a different method...
[extended reasoning]
</think>
<answer>
The answer is 42.
</answer>
```

These thinking tokens are typically **hidden from the user** in production (e.g., OpenAI o1 does not show its thinking), but DeepSeek R1 makes them visible.

### Two Types of Scaling Laws

The lecture distinguishes between two paradigms for improving LLM performance:

#### Pre-training Scaling (Traditional)
- **Scaling law:** More compute at training time leads to better performance
- Governed by Chinchilla-style scaling laws (Hoffmann et al., 2022)
- Focus: increase data, model size, training FLOPs

#### Test-Time / Inference Scaling (Reasoning Models)
- **Scaling law:** More compute at inference time leads to better performance
- The model "thinks longer" by generating more reasoning tokens
- This is the paradigm reasoning models exploit

### How Reasoning Models Improve with More Thinking

**Observation:** Increasing the number of reasoning tokens at test time improves accuracy.

- DeepSeek-R1-Zero shows increasing accuracy on AIME benchmarks as training progresses
- Response length keeps increasing with RL training steps (from ~1000 tokens at step 0 to ~9000+ tokens at step 8000)
- Both pass@1 and consensus@16 metrics improve steadily

### "Aha Moment" in Reasoning

During RL training, models spontaneously develop behaviors like:
- **Self-reflection:** "Wait, let me reconsider..."
- **Backtracking:** Recognizing errors and trying alternative approaches
- **Verification:** Checking intermediate results before proceeding

This emergent behavior was notably observed in DeepSeek-R1-Zero without any supervised fine-tuning on reasoning traces -- the model discovered these strategies purely through RL.

---

## 4. Test-Time Compute Scaling

### The Core Idea

Instead of making a model smarter by training longer (more FLOPs at training time), we can make it smarter by **thinking longer at inference time** (more FLOPs at test time). This is the fundamental insight behind reasoning models.

### Search Strategies at Inference Time

Several strategies allow models to use more compute at test time:

#### Best-of-N Sampling
- Generate N candidate responses
- Score each response (using a verifier or reward model)
- Return the highest-scoring response
- Simple but effective; linear scaling of compute with N

#### Self-Consistency (Majority Voting)
- Generate N reasoning chains
- Extract the final answer from each
- Return the most common answer (majority vote)
- No verifier needed -- relies on statistical robustness

#### Beam Search
- Maintain a "beam" of the top-K partial solutions at each step
- Expand each beam, score the expansions, keep the top-K
- More structured exploration than independent sampling

#### Tree Search (e.g., Monte Carlo Tree Search)
- Build a tree of reasoning steps
- Use a value function to evaluate partial reasoning paths
- Explore promising branches more deeply
- Similar to how AlphaGo searches game trees, but applied to reasoning

### Verifiers

Verifiers are models that evaluate the quality of a generated response. They are critical for test-time compute scaling because they determine which of the multiple sampled responses to select.

**Two types of verifiers** (see Section 5 for details):
- **Outcome-based:** Checks only the final answer
- **Process-based:** Checks each reasoning step individually

### When to Scale Training vs Inference Compute

The optimal strategy depends on the problem:
- **Easy problems:** Standard inference is sufficient; extra test-time compute is wasted
- **Medium problems:** Test-time scaling (self-consistency, best-of-N) helps significantly
- **Very hard problems:** May need both better base models AND test-time scaling
- **Diminishing returns:** Each additional sample yields less marginal improvement

---

## 5. Reinforcement Learning for Reasoning

### Why RL for Reasoning?

Traditional supervised fine-tuning (SFT) requires explicit reasoning demonstrations -- a human has to write out the correct chain-of-thought for each training example. RL allows models to:
- **Discover reasoning strategies on their own** without human demonstrations
- **Optimize directly for correct final answers** (the reward)
- **Develop emergent reasoning behaviors** not present in any training data (the "aha moment")

### RL Training Setup for Reasoning

The basic RL loop for reasoning models:

1. **Question (q)** is sampled from a dataset
2. **Policy model** generates multiple candidate outputs (o_1, o_2, ..., o_G) -- a "group"
3. **Reward model** (or rule-based verifier) scores each output (r_1, r_2, ..., r_G)
4. **Advantage** is computed relative to the group
5. **Policy is updated** to increase probability of high-advantage outputs

### Reward Design

For reasoning tasks, rewards are typically:

**R1-Zero reward:**
- **Formatting reward:** Did the model use the correct `<think>...</think>` and `<answer>...</answer>` format?
- **Accuracy reward:** Is the final answer correct?

**R1 reward (reasoning RL stage):**
- Formatting + accuracy + **language consistency** (prevents language mixing in the chain-of-thought)

**R1 reward (general RL stage):**
- For reasoning data: Formatting + accuracy
- For general data: Helpfulness + harmlessness

### Outcome-Based Reward Models (ORM)

An ORM evaluates only the **final answer** of a reasoning chain:

- **Input:** Question + full reasoning chain + final answer
- **Output:** Score (correct / incorrect, or a scalar reward)
- **Advantage:** Simple to implement; clear signal (right or wrong)
- **Limitation:** Does not distinguish a correct answer reached by flawed reasoning from one reached by sound reasoning; does not provide feedback on intermediate steps

### Process-Based Reward Models (PRM)

A PRM evaluates **each step** of the reasoning chain:

- **Input:** Question + reasoning chain up to step k
- **Output:** Score for step k (is this step valid?)
- **Advantage:** Provides fine-grained feedback; can catch errors mid-chain; encourages sound reasoning process
- **Limitation:** Much harder to train (requires step-level annotations); computationally expensive to run on every step

### ORM vs PRM Summary

| Feature | ORM | PRM |
|---------|-----|-----|
| **What it evaluates** | Final answer only | Each reasoning step |
| **Training data** | (question, answer) pairs with correctness labels | Step-level correctness annotations |
| **Feedback granularity** | Coarse (whole-chain) | Fine (per-step) |
| **What it rewards** | Getting the right answer | Getting each step right |
| **Risk** | May reward lucky guesses with flawed reasoning | Expensive to annotate and train |

**Key insight:** The DeepSeek R1 training pipeline uses primarily **outcome-based** rewards (formatting + accuracy), yet still achieves strong reasoning through the RL process. The model learns good reasoning processes as an emergent behavior of optimizing for correct outcomes.

---

## 6. GRPO (Group Relative Policy Optimization)

### Definition

**GRPO = Group Relative Policy Optimization**

Introduced in: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", Shao et al., 2024.

GRPO is the RL algorithm used by DeepSeek to train its reasoning models. It is a variant of policy gradient methods designed to be more memory-efficient than PPO by eliminating the need for a separate value (critic) model.

### GRPO Objective (High Level)

The loss function has two components:

$$\mathscr{L}(\theta) = \text{Maximize advantages} + \text{Don't deviate too much from old/base model}$$

### Key Insight: Group-Based Advantage Estimation

$$\text{Advantage} \approx \text{Reward} - \text{Avg(reward of group)}$$

More precisely, the advantage for output i is:

$$A_i = \frac{r_i - \text{mean}(\{r_1, r_2, \cdots, r_G\})}{\text{std}(\{r_1, r_2, \cdots, r_G\})}$$

**This is the big difference compared to PPO!** Instead of using a learned value function to estimate advantages, GRPO uses the group statistics (mean and standard deviation of rewards within a sampled group of outputs) as the baseline.

**Plain-language explanation:** For each question, the model generates G different answers. Each answer gets a reward. The advantage of any one answer is simply "how much better was this answer compared to the average of the group?" -- normalized by the spread. Answers above average get positive advantage (encouraged); answers below average get negative advantage (discouraged).

### GRPO Full Objective

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}[q \sim P(Q), \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{old}}(O|q)]$$

$$\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min\left[\frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})} \hat{A}_{i,t},\ \text{clip}\left(\frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})}, 1-\varepsilon, 1+\varepsilon\right) \hat{A}_{i,t}\right] - \beta \mathbb{D}_{KL}[\pi_\theta || \pi_{ref}] \right\}$$

Where:
- **q** = question sampled from dataset P(Q)
- **o_i** = the i-th output sampled from the old policy
- **G** = group size (number of sampled outputs per question)
- **pi_theta / pi_theta_old** = current / old policy (the probability ratio measures how much the policy has changed)
- **A_hat_{i,t}** = estimated advantage for token t of output i
- **epsilon** = clipping parameter (prevents too-large policy updates)
- **beta** = KL penalty coefficient (controls how much the policy can drift from the reference)
- **pi_ref** = reference model (frozen copy of the base model)

### GRPO Pipeline (Diagram)

```
q --> [Policy Model] --> o_1, o_2, ..., o_G --> [Reward Model] --> r_1, r_2, ..., r_G --> [Group Computation] --> A_1, A_2, ..., A_G
         ^                                          |
         |                                    [Reference Model]
         |                                          |
         +------------------------------------------+
                          KL divergence
```

The pipeline works as follows:
1. A question q is fed to the **Policy Model**
2. The policy generates **G outputs** (a group) for that question
3. Each output is scored by the **Reward Model** to get rewards r_1 through r_G
4. The **Group Computation** normalizes rewards to get advantages A_1 through A_G
5. A **Reference Model** (frozen) computes KL divergence to regularize updates
6. The Policy Model weights are updated to increase probability of high-advantage outputs

### Comparison: GRPO vs PPO

#### PPO Pipeline

```
q --> [Policy Model] --> o --> [Reference Model] --KL--> (+) --> r
                          |-> [Reward Model] -------->
                          |-> [Value Model] ----------> v ---> GAE --> A
```

PPO requires a **Value Model** (also called a critic), which is typically the same size as the policy model. This model learns to predict the expected reward, and is used together with actual rewards to compute advantages via Generalized Advantage Estimation (GAE).

#### Similarities
- Both use the **probability ratio** (pi_theta / pi_theta_old) for the policy gradient update
- Both use **clipping** to prevent too-large policy updates: clip(ratio, 1-epsilon, 1+epsilon)

#### Differences

| Feature | GRPO | PPO |
|---------|------|-----|
| **Advantage estimation** | Group-based: normalized (reward - mean) / std across G samples | GAE using a learned value model |
| **Value model** | Not needed | Required (additional model to train) |
| **KL penalty** | Explicit beta * D_KL[pi_theta \|\| pi_ref] term in objective | KL folded into the reward signal |
| **Number of outputs per question** | G outputs (a group) | Typically 1 output |
| **Frozen models** | Reference model + Reward model | Reference model + Reward model |
| **Trained models** | Policy model only | Policy model + Value model |

**Key advantage of GRPO:** No need to train a separate value model, which is computationally expensive (often the same size as the policy model). This makes GRPO significantly more memory-efficient -- a critical concern when the policy model is already hundreds of billions of parameters.

### PPO Full Objective (for Comparison)

$$\mathcal{J}_{PPO}(\theta) = \mathbb{E}[q \sim P(Q), o \sim \pi_{\theta_{old}}(O|q)] \frac{1}{|o|} \sum_{t=1}^{|o|} \min\left[\frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\theta_{old}}(o_t|q, o_{<t})} A_t,\ \text{clip}\left(\frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\theta_{old}}(o_t|q, o_{<t})}, 1-\varepsilon, 1+\varepsilon\right) A_t\right]$$

**Similarities with GRPO (highlighted):**
- The **ratio** pi_theta / pi_theta_old appears in both
- The **clipping** mechanism clip(ratio, 1-epsilon, 1+epsilon) appears in both

**Differences (highlighted):**
- GRPO has an **explicit KL penalty** term; PPO folds KL into the reward
- GRPO uses **group-based advantage** A_hat_{i,t}; PPO uses GAE-based advantage A_t

---

## 7. The "Increasing Output Length" Phenomenon

### Observation

Response length keeps increasing with RL training. In the DeepSeek-R1-Zero experiments, the average length per response grows from ~1000 tokens to ~9000+ tokens over 8000 training steps. While accuracy also improves, the length increase is disproportionate -- the model becomes increasingly verbose.

### Why This Happens: Length Bias in the Objective

The GRPO objective normalizes by `1/|o_i|` (dividing by the length of each output). This creates an asymmetric incentive:

| | A > 0 (good output) | A < 0 (bad output) |
|---|---|---|
| **Short output** | Strong upward push (large gradient per token) | Strong downward push -- **Bad incentive!** |
| **Long output** | Weak upward push (small gradient per token) | Weak downward push (small gradient per token) |

**The problem explained simply:** When a short response gets a negative advantage (it was wrong), the `1/|o_i|` factor makes the penalty per token very large. The model learns to avoid short wrong answers by making its answers longer -- even when brevity would be fine. Meanwhile, long wrong answers receive only a small penalty per token. This systematically pushes the model toward longer outputs.

### Mitigating the Increasing Length Phenomenon

**Problem:** The `1/|o_i|` normalization in the GRPO objective biases toward longer outputs.

**Remedy:** Equalize token-level contributions by changing the normalization.

#### DAPO (Decoupled Advantage Policy Optimization)
- Reference: "DAPO: An Open-Source LLM Reinforcement Learning System at Scale", Yu et al., 2025
- Replaces per-output normalization with **global normalization across all tokens in the group:**

$$\text{GRPO: } \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \quad \longrightarrow \quad \text{DAPO: } \frac{1}{\sum_{i=1}^{G} |o_i|} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|}$$

Every token across all outputs in the group contributes equally to the gradient.

#### Dr. GRPO
- Reference: "Understanding R1-Zero-Like Training: A Critical Perspective", Liu et al., 2025
- Uses a different normalization scheme that removes `1/|o_i|` entirely:

$$\text{Dr. GRPO: } \frac{1}{G} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|}$$

This sums over tokens directly without dividing by output length.

#### Empirical Results

- **Token Efficiency:** Dr. GRPO achieves higher reward for a given output length compared to standard GRPO (better reward-per-token)
- **Output Length (Correct):** Both Dr. GRPO and GRPO converge to similar lengths for correct outputs (~400 tokens)
- **Output Length (Incorrect):** Dr. GRPO produces shorter incorrect outputs -- the model learns to "fail fast" rather than generating long wrong answers, which is more compute-efficient

### Exploration of Other Adjustments

#### Bias Linked to Level of Difficulty

The advantage estimation can be biased by question difficulty. The standard advantage:

$$\hat{A}_{i,t} = \frac{R(\mathbf{q}, \mathbf{o}_i) - \text{mean}(\{R(\mathbf{q}, \mathbf{o}_1), \ldots, R(\mathbf{q}, \mathbf{o}_G)\})}{\text{std}(\{R(\mathbf{q}, \mathbf{o}_1), \ldots, R(\mathbf{q}, \mathbf{o}_G)\})}$$

Questions where all G outputs are correct (easy questions) or all wrong (very hard questions) have zero variance in the group, leading to degenerate advantages (division by zero or undefined gradients). These questions contribute nothing to learning.

#### Encouraging Diversity (Asymmetric Clipping)

Standard clipping uses symmetric bounds:

$$\text{clip}(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon)$$

DAPO introduces **asymmetric clipping** to encourage exploration:

$$\text{clip}(r_{i,t}(\theta), 1-\varepsilon_{low}, 1+\varepsilon_{high})$$

Where epsilon_high > epsilon_low. This allows the policy to more easily **increase** probabilities of good outputs (larger upper bound) than **decrease** probabilities of bad ones (smaller lower bound), encouraging exploration of diverse reasoning strategies.

---

## 8. Applications: DeepSeek Training Recipes and Results

### DeepSeek Model Family Overview

```
Base model (V3-Base) --> "Traditional" model (V3)    [SFT + RLHF]
                     \
                      --> Reasoning models:
                            R1-Zero   [proof of concept -- RL only]
                            R1        [full pipeline -- SFT + RL + SFT + RL]
```

### DeepSeek R1-Zero Training Recipe

R1-Zero is a **proof of concept** showing that RL alone (without any supervised reasoning data) can produce a reasoning model.

**Step 1:** Pretrain model with "traditional" techniques: **V3-Base**
- Architecture: DeepSeekMoE (Mixture of Experts)
- ~671B total parameters, ~37B active per token
- Uses Multi-Head Latent Attention (MLA) for efficient KV-cache
- Router selects top-K experts per token from routed + shared expert pools

**Step 2:** GRPO with reasoning data: **R1-Zero**
- Uses a structured prompt template that instructs the model to use think/answer tags:
```
A conversation between User and Assistant. The user asks a question, and the
Assistant solves it. The assistant first thinks about the reasoning process
in the mind and then provides the user with the answer. The reasoning
process and answer are enclosed within <think> </think> and <answer>
</answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>.
User: <this placeholder is replaced by a reasoning query>
Assistant:
```
- Reward = Formatting + Accuracy (rule-based, no learned reward model needed for math)

**R1-Zero Results (AIME benchmark):**
- AIME accuracy increases from ~0.2 to ~0.7+ (pass@1) over 8000 RL steps
- Consensus@16 reaches ~0.85, competitive with OpenAI o1-0912

**R1-Zero Benefits vs Challenges:**

| Benefits | Challenges |
|----------|------------|
| Develops reasoning abilities without any SFT on reasoning data | Chains of reasoning have formatting and readability issues |
| Emergent self-reflection and backtracking behaviors | Language mixing (switches between languages mid-chain) |
| Proves RL alone can induce reasoning | Not ready for production use |

### DeepSeek R1 Training Recipe (Full Pipeline -- 5 Steps)

R1 is the **full pipeline** that builds on R1-Zero's insights with additional SFT stages for quality and readability.

**Step 1:** Pretrain model with "traditional" techniques: **V3-Base**
- Same MoE architecture (~671B total, ~37B active)

**Step 2:** "Small-scale" SFT with reasoning data
- Data source: Long CoTs generated with R1-Zero and **rewritten by humans** for readability
- Purpose: Initialize the model with clean reasoning format before RL
- Fixes the readability and formatting issues seen in R1-Zero

**Step 3:** GRPO with reasoning data
- ~Same RL process as R1-Zero
- Reward = Formatting + accuracy + **language consistency** (new reward signal that prevents language mixing in the chain-of-thought)

**Step 4:** "Large-scale" SFT with reasoning AND non-reasoning data
- **~600k pairs** of reasoning data (maths, coding, logic)
  - Created via **rejection sampling** of "R1 so far" responses, filtered by rules + V3 judge
- **~200k pairs** of "general" data (question answering, summarization, etc.)
  - Mostly **reuses V3 SFT data**
- Purpose: Maintain general capabilities while adding reasoning

**Step 5:** GRPO with reasoning and non-reasoning data: **R1**
- Reasoning data: Maths, coding, logic -- Reward = Formatting + accuracy
- General data: Mostly reuses V3 RL data -- Reward = helpfulness + harmlessness
- This final RL stage polishes both reasoning and general capabilities

### DeepSeek R1 Benchmark Results

Benchmark comparison across leading models (selected highlights, best results in bold):

| Category | Benchmark | Claude 3.5 Sonnet | GPT-4o | DeepSeek V3 | o1-mini | o1-1217 | **DeepSeek R1** |
|----------|-----------|-------------------|--------|-------------|---------|---------|-------------|
| English | MMLU | 88.3 | 87.2 | 88.5 | 85.2 | **91.8** | 90.8 |
| English | MMLU-Redux | 88.9 | 88.0 | 89.1 | 86.7 | - | **92.9** |
| English | MMLU-Pro | 78.0 | 72.6 | 75.9 | 80.3 | - | **84.0** |
| English | DROP | 88.3 | 83.7 | 91.6 | 83.9 | 90.2 | **92.2** |
| English | IF-Eval | **86.5** | 84.3 | 86.1 | 84.8 | - | 83.3 |
| English | GPQA Diamond | 65.0 | 49.9 | 59.1 | 60.0 | **75.7** | 71.5 |
| English | SimpleQA | 28.4 | 38.2 | 24.9 | 7.0 | **47.0** | 30.1 |
| English | FRAMES | 72.5 | 80.5 | 73.3 | 76.9 | - | **82.5** |
| English | AlpacaEval2.0 | 52.0 | 51.1 | 70.0 | 57.8 | - | **87.6** |
| English | ArenaHard | 85.2 | 80.4 | 85.5 | 92.0 | - | **92.3** |
| Code | LiveCodeBench | 38.9 | 32.9 | 36.2 | 53.8 | 63.4 | **65.9** |
| Code | Codeforces (Percentile) | 20.3 | 23.6 | 58.7 | 93.4 | **96.6** | 96.3 |
| Code | Codeforces (Rating) | 717 | 759 | 1134 | 1820 | **2061** | 2029 |
| Code | SWE Verified | **50.8** | 38.8 | 42.0 | 41.6 | 48.9 | 49.2 |
| Math | AIME 2024 | 16.0 | 9.3 | 39.2 | 63.6 | 79.2 | **79.8** |
| Math | MATH-500 | 78.3 | 74.6 | 90.2 | 90.0 | 96.4 | **97.3** |
| Math | CNMO 2024 | 13.1 | 10.8 | 43.2 | 67.6 | - | **78.8** |
| Chinese | CLUEWSC | 85.4 | 87.9 | 90.9 | 89.9 | - | **92.8** |
| Chinese | C-Eval | 76.7 | 76.0 | 86.5 | 68.9 | - | **91.8** |

**Key observations:**
- R1 achieves best-in-class results on many benchmarks, particularly in math, code, and reasoning
- R1 is competitive with or exceeds OpenAI o1-1217 on most tasks
- Particularly strong gains in AIME 2024 (79.8), MATH-500 (97.3), and LiveCodeBench (65.9)
- Some weaknesses remain in factual knowledge tasks like SimpleQA (30.1 vs o1's 47.0)

### Distillation: Transferring Reasoning to Smaller Models

#### Two Types of Distillation

**Traditional distillation (from Lecture 2):**
- Teacher (T) and Student (S) process the same input x
- Goal: Match the teacher's next-token probability distribution
- Student learns to mimic the full output distribution

**Distillation for reasoning models (used by DeepSeek):**
- Teacher: R1 generates complete reasoning traces (full responses including thinking tokens)
- Student: R1-Distill learns to reproduce **entire responses** via SFT
- Goal: SFT-learn the reasoning traces themselves, not token distributions
- Uses targeted inputs specifically designed for reasoning

**Key difference:** Traditional distillation matches distributions; reasoning distillation matches full responses (the student is fine-tuned on the teacher's complete outputs as training examples).

#### Distillation Results

| Model | AIME 2024 (pass@1) | AIME 2024 (cons@64) | MATH-500 | GPQA Diamond | LiveCodeBench | CodeForces |
|-------|----|----|----|----|----|----|
| GPT-4o-0513 | 9.3 | 13.4 | 74.6 | 49.9 | 32.9 | 759 |
| Claude-3.5-Sonnet-1022 | 16.0 | 26.7 | 78.3 | 65.0 | 38.9 | 717 |
| OpenAI o1-mini | 63.6 | 80.0 | 90.0 | 60.0 | 53.8 | 1820 |
| QwQ-32B-Preview | 50.0 | 60.0 | 90.6 | 54.5 | 41.9 | 1316 |
| DeepSeek-R1-Distill-Qwen-1.5B | 28.9 | 52.7 | 83.9 | 33.8 | 16.9 | 954 |
| DeepSeek-R1-Distill-Qwen-7B | 55.5 | 83.3 | 92.8 | 49.1 | 37.6 | 1189 |
| DeepSeek-R1-Distill-Qwen-14B | 69.7 | 80.0 | 93.9 | 59.1 | 53.1 | 1481 |
| **DeepSeek-R1-Distill-Qwen-32B** | **72.6** | **83.3** | **94.3** | **62.1** | **57.2** | **1691** |
| DeepSeek-R1-Distill-Llama-8B | 50.4 | 80.0 | 89.1 | 49.0 | 39.6 | 1205 |
| **DeepSeek-R1-Distill-Llama-70B** | **70.0** | **86.7** | **94.5** | **65.2** | **57.5** | **1633** |

**Remarkable findings:**
- Even a 1.5B parameter distilled model (R1-Distill-Qwen-1.5B) achieves 28.9% on AIME 2024, far exceeding GPT-4o (9.3%) at a fraction of the size
- The 32B distilled model matches or exceeds o1-mini on several benchmarks

#### Distillation vs RL from Scratch

Comparing at the 32B parameter count:

| Model | AIME pass@1 | AIME cons@64 | MATH-500 | GPQA Diamond | LiveCodeBench |
|-------|----|----|----|----|-----|
| QwQ-32B-Preview | 50.0 | 60.0 | 90.6 | 54.5 | 41.9 |
| DeepSeek-R1-Zero-Qwen-32B | 47.0 | 60.0 | 91.6 | 55.0 | 40.2 |
| **DeepSeek-R1-Distill-Qwen-32B** | **72.6** | **83.3** | **94.3** | **62.1** | **57.2** |

**Key takeaway:** Distillation from a large reasoning model (R1) massively outperforms training a smaller model from scratch with RL (R1-Zero). The distilled 32B model gains +25.6 points on AIME pass@1 over the RL-from-scratch version. This suggests that for practical deployment, distillation is a far more compute-efficient path to smaller reasoning models than training them with RL independently.

---

## Key Concepts Glossary

| Term | Definition |
|------|-----------|
| **Reasoning** | The ability to draw conclusions through logical, multi-step inference -- corresponding to Kahneman's "System 2" thinking |
| **Chain of Thought (CoT)** | Extended reasoning traces produced by the model before the final answer, serving as a form of working memory |
| **Zero-Shot CoT** | Adding "Let's think step by step" to a prompt to elicit reasoning without examples |
| **Self-Consistency** | Sampling multiple reasoning paths and taking a majority vote on the final answer |
| **Thinking Tokens** | The internal reasoning tokens generated by reasoning models (e.g., inside `<think>` tags) before producing a final answer |
| **Test-Time Compute** | Computational resources used during inference; reasoning models trade more inference compute for better answers |
| **GRPO** | Group Relative Policy Optimization -- RL algorithm that computes advantages relative to a group of sampled outputs, eliminating the need for a value model |
| **PPO** | Proximal Policy Optimization -- standard RL algorithm that uses a learned value model and GAE for advantage estimation |
| **GAE** | Generalized Advantage Estimation -- method used in PPO to estimate advantages using a value function |
| **ORM** | Outcome-based Reward Model -- evaluates only the final answer for correctness |
| **PRM** | Process-based Reward Model -- evaluates each reasoning step individually |
| **KL Divergence** | Kullback-Leibler divergence -- measures how much the updated policy has diverged from the reference model; used as a regularizer |
| **Clipping** | Constraining the policy ratio to [1-epsilon, 1+epsilon] to prevent destructively large policy updates |
| **Advantage** | How much better (or worse) a particular output is compared to the baseline (group average in GRPO, value function in PPO) |
| **Rejection Sampling** | Generating many candidate outputs and keeping only those that meet quality criteria |
| **DAPO** | Decoupled Advantage Policy Optimization -- variant that fixes length bias via global token normalization and uses asymmetric clipping |
| **Dr. GRPO** | Variant of GRPO that removes per-output length normalization to fix the increasing length phenomenon |
| **MoE** | Mixture of Experts -- architecture where only a subset of parameters are activated per token (V3-Base uses ~37B of 671B) |
| **R1-Zero** | DeepSeek's proof-of-concept reasoning model trained with RL only (no SFT on reasoning data) |
| **R1** | DeepSeek's full reasoning model trained with the complete 5-step pipeline |
| **Distillation** | Training a smaller model to reproduce the outputs (reasoning traces) of a larger model |
| **Best-of-N** | Generating N responses and selecting the best one using a verifier |

---

## Key Formulas Summary

### GRPO Advantage

$$A_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}$$

### GRPO Objective

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min\left[\frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})} \hat{A}_{i,t},\ \text{clip}\left(\frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})}, 1-\varepsilon, 1+\varepsilon\right) \hat{A}_{i,t}\right] - \beta \mathbb{D}_{KL}[\pi_\theta || \pi_{ref}] \right\}\right]$$

### PPO Objective

$$\mathcal{J}_{PPO}(\theta) = \mathbb{E}\left[\frac{1}{|o|} \sum_{t=1}^{|o|} \min\left[\frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\theta_{old}}(o_t|q, o_{<t})} A_t,\ \text{clip}\left(\frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\theta_{old}}(o_t|q, o_{<t})}, 1-\varepsilon, 1+\varepsilon\right) A_t\right]\right]$$

### DAPO Normalization Fix

$$\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \quad \longrightarrow \quad \frac{1}{\sum_{i=1}^{G}|o_i|} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|}$$

### Dr. GRPO Normalization Fix

$$\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \quad \longrightarrow \quad \frac{1}{G} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|}$$

### Asymmetric Clipping (DAPO)

$$\text{clip}(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon) \quad \longrightarrow \quad \text{clip}(r_{i,t}(\theta), 1-\varepsilon_{low}, 1+\varepsilon_{high})$$

---

## References

- "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", Shao et al., 2024.
- "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", DeepSeek-AI, 2025.
- "DeepSeek-V3 Technical Report", DeepSeek-AI, 2024.
- "DAPO: An Open-Source LLM Reinforcement Learning System at Scale", Yu et al., 2025.
- "Understanding R1-Zero-Like Training: A Critical Perspective", Liu et al., 2025.
- "Super Study Guide: Transformers & Large Language Models", Amidi et al., 2024.
- "Large Language Models are Zero-Shot Reasoners", Kojima et al., 2022.
- "Self-Consistency Improves Chain of Thought Reasoning in Language Models", Wang et al., 2023.
- "Thinking, Fast and Slow", Kahneman, 2011.

---

*Source: Stanford CME 295, Fall 2025, Lecture 6 slides and video.*
