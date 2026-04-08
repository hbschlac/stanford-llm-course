# CME 295: Large Language Models -- Lecture 5: LLM Tuning

**Course:** Stanford CME 295 -- Large Language Models
**Instructors:** Afshine Amidi & Shervine Amidi
**Video:** [YouTube -- Lecture 5](https://www.youtube.com/watch?v=PmW_TMQ3l0I)
**Slides:** [PDF](https://cme295.stanford.edu/slides/fall25-cme295-lecture5.pdf)

**Topics Covered:**
- Preference tuning and the alignment problem
- Data collection for human preferences (Bradley-Terry model, ELO scoring)
- RLHF (Reinforcement Learning from Human Feedback) -- the InstructGPT pipeline
- Reward model training
- PPO (Proximal Policy Optimization) -- clipping, KL penalty, advantages
- Best-of-N (BoN) sampling as an RL workaround
- DPO (Direct Preference Optimization) -- derivation, implicit reward, comparison to PPO
- Behavior of preference-tuned models

---

## 1. Recap: The LLM Training Pipeline

The lecture begins by situating preference tuning within the full LLM training pipeline, building on previous lectures:

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
  [Preference tuning]    -->  Model aligns with (human) preferences    <-- TODAY'S FOCUS
```

**Key insight:** Even after supervised finetuning (SFT), a model can still misbehave. Preference tuning is the step that teaches the model *which* responses are better or worse, not just *how* to respond.

---

## 2. Preference Tuning: The Alignment Problem

### Why do we need preference tuning?

**Context.** After SFT, a model may still produce unhelpful, harmful, or inappropriate responses. We need to inject *negative signals* -- the model needs to learn not just what to say, but what NOT to say.

**Example from the slides:**

| Prompt | Bad Response (from SFT model) |
|--------|-------------------------------|
| "Suggest a new activity I could do with my teddy bear." | "I'd suggest you do not spend much time with your teddy bear at all." |

This response is grammatically correct and follows instructions, but it is unhelpful and dismissive. SFT alone cannot fix this because SFT only trains on *positive* examples.

### The preference pair approach

**Idea.** Collect preference pairs and train the model on them. For a given prompt, present two responses and label which one is better:

- **Preferred (winner):** "Of course! Teddy bears not only make awesome companions for a delightful sleep, but can also be great buddies for fun activities. How about you both watch a movie together?"
- **Rejected (loser):** "I'd suggest you do not spend much time with your teddy bear at all."

### Why preference tuning instead of more SFT?

- **Easier to compare than to generate** -- it is simpler for a human to say "A is better than B" than to write a perfect response from scratch
- Comparison judgments are more consistent across annotators
- Captures subtle quality differences that are hard to encode in explicit rules

---

## 3. Data Collection for Preference Tuning

The lecture covers how to collect the human preference data that drives all alignment methods.

### The InstructGPT pipeline (Ouyang et al., 2022)

The foundational paper "Training language models to follow instructions with human feedback" established the three-step pipeline:

1. **Supervised Finetuning (SFT)** -- train on demonstration data
2. **Reward Model (RM) Training** -- train a model to predict human preferences
3. **Reinforcement Learning (RL)** -- optimize the LLM using the reward model

### Collecting comparison data

For each prompt, human labelers are shown multiple model outputs and asked to rank them. The key details:

- Labelers compare pairs of responses and indicate which is preferred
- This produces a dataset of (prompt, winner, loser) triples
- More scalable than having humans write ideal responses from scratch

### Bradley-Terry Model

The Bradley-Terry model provides the mathematical framework for turning pairwise comparisons into scores. Given two responses y_1 and y_2 to a prompt x:

```
P(y_1 > y_2 | x) = exp(r(x, y_1)) / [exp(r(x, y_1)) + exp(r(x, y_2))]
```

This can be rewritten using the sigmoid function:

```
P(y_1 > y_2 | x) = sigma(r(x, y_1) - r(x, y_2))
```

Where:
- `r(x, y)` is the reward (or score) assigned to response y given prompt x
- `sigma` is the sigmoid function: sigma(z) = 1 / (1 + exp(-z))

**Plain-language explanation:** The Bradley-Terry model says the probability that response A beats response B depends on the *difference* in their scores. If A's score is much higher than B's, it is very likely to be preferred. If they are close, it is a coin flip.

### ELO Scoring

ELO scoring (borrowed from chess rankings) provides another way to rank responses:
- Each response starts with a base rating
- After each comparison, the winner gains points and the loser loses points
- The amount of points exchanged depends on how "surprising" the outcome was
- This produces a global ranking of response quality

---

## 4. RLHF: Reinforcement Learning from Human Feedback

### Overview

RLHF is the two-step process that uses human preference data to improve an LLM:

**Step 1: Reward Modeling** -- Train a separate model to predict which responses humans prefer
**Step 2: Reinforcement Learning** -- Use that reward model to guide the LLM toward better responses

### Step 1: Reward Model Training

**Data:**
- O(10,000) observations (relatively small dataset)
- Label = human rating (this is where the "HF" from RLHF comes from)

**Model:**
- Pretrained LLM with a **classification head** (instead of next token prediction)
- Can use encoder-only architectures: BERT and similar models via [CLS] projection

**Reference:** "RewardBench: Evaluating Reward Models for Language Modeling", Lambert et al., 2024.

The reward model takes a (prompt, response) pair as input and outputs a single scalar score representing how "good" that response is according to human preferences.

### Step 2: Reinforcement Learning

**Idea.** Change the weights of the LLM to penalize bad answers and promote good answers via **Reinforcement Learning** using the **Reward Model**.

The RL loop works as follows:

```
[Prompt] --> [LLM (Trained)] --> [Response] --> [RM (Frozen)] --> [Score: thumbs up/down]
                ^                                                         |
                |_________________________via RL__________________________|
```

Key details of this diagram:
- The **LLM is being trained** (weights are updated)
- The **Reward Model is frozen** (weights are fixed)
- The RL signal flows back from the reward model's score to update the LLM

**Data:**
- O(100,000) observations (much larger than reward model training data)
- Label = score given by the reward model (not human labels -- those are too expensive at this scale)

**Model.** Initialized at the SFT model (not from scratch).

**Training.** Change weights of policy (LLM) via objective function:

```
L(theta) = [Maximize rewards] + [Don't deviate too much from base model]
```

The "don't deviate too much" term exists to avoid **"reward hacking"** and **training instability** -- without it, the model could find degenerate ways to get high reward scores without actually producing good responses.

---

## 5. PPO: Proximal Policy Optimization

PPO is the specific RL algorithm most commonly used in the RLHF pipeline. It was introduced by Schulman et al., 2017.

### The PPO-RLHF Objective

The full objective function:

```
L(theta) = -[r(x, y_hat) - lambda * KL(pi_theta(y_hat | x) || pi_ref(y_hat | x))]
```

Where:
- `r(x, y_hat)` = reward from the reward model ("maximize rewards")
- `KL(pi_theta || pi_ref)` = KL divergence between current policy and reference model ("don't deviate too much from base model")
- `lambda` = hyperparameter controlling the strength of the KL penalty
- `pi_theta` = current policy (the LLM being trained)
- `pi_ref` = reference policy (the base/SFT model, frozen)

### KL Divergence Explained

KL divergence measures how different two probability distributions are:

```
KL(P || Q) = sum_i p_i * log(p_i / q_i)
```

**Plain-language explanation:** KL divergence answers the question "how different is distribution P from distribution Q?" If the two distributions are identical, KL = 0. The more different they are, the larger the KL value. In the RLHF context, it measures how far the trained model has drifted from the original SFT model.

The negative sign in the objective means we are *minimizing* the loss, which means:
- Maximizing the reward (good)
- Minimizing the KL divergence from the base model (staying close)

### PPO Actually Computes Advantages (Not Just Rewards)

The lecture clarifies that PPO does not directly use raw rewards. Instead:

```
Advantage = Reward - Baseline
```

The **baseline** comes from a **Value function**:
- Operates at the token level
- Predicts "what would be the reward if we follow the policy from here"
- Trained jointly with the policy
- Label = reward

The advantage computation uses the **GAE (Generalized Advantage Estimation)** method.

**Reference:** "High-Dimensional Continuous Control Using Generalized Advantage Estimation", Schulman et al., 2015.

### Variation 1: PPO-Clip

**Idea.** Clip the ratio between new and old policy to prevent large updates.

```
L^CLIP(theta) = E_t[min(r_t(theta) * A_hat_t, clip(r_t(theta), 1 - epsilon, 1 + epsilon) * A_hat_t)]
```

Where the probability ratio is:

```
r_t(theta) = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t)
```

**Important terminology notes from the slides:**
- This is an **objective function** (to be maximized), NOT a loss (despite being called "L")
- The `r` here refers to the **ratio** between policies, not to rewards -- this is a common source of confusion

**How clipping works (two cases):**

| Case | Behavior |
|------|----------|
| **A > 0** (good action) | The objective increases with r, but is clipped at r = 1 + epsilon. This prevents the policy from moving too far toward a good action. |
| **A < 0** (bad action) | The objective decreases with r, but is clipped at r = 1 - epsilon. This prevents the policy from moving too far away from a bad action. |

The clipping creates a "trust region" -- the policy can only change by a bounded amount per update step. Typical values: epsilon = 0.1 or 0.2.

### Variation 2: PPO-KL Penalty

**Idea.** Instead of clipping, penalize the difference in policy distributions directly.

```
L^KLPEN(theta) = E_t[pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t) * A_hat_t - beta * KL[pi_theta_old(. | s_t), pi_theta(. | s_t)]]
```

**Important terminology distinction:**
- `theta_old` = model from the **previous RL iteration** (changes each step)
- `theta_ref` = the **base model** (fixed throughout training)

**Modern practice:** Nowadays, the KL divergence is computed with respect to **ref** (the base model), not old (the previous iteration). This keeps the model anchored to the original SFT model throughout training.

### Four Models Required in PPO-RLHF

PPO-based RLHF requires maintaining **four separate models** simultaneously:

| Model | Role | Status |
|-------|------|--------|
| **Policy** (pi_theta) | The LLM being trained | Updated |
| **Value function** | Predicts expected reward | Updated |
| **Reward Model** | Scores responses | Frozen |
| **Base/Reference Model** | Anchor for KL penalty | Frozen |

This is computationally expensive and one of the main motivations for simpler alternatives like DPO.

---

## 6. Challenges with RL-Based Approaches

The lecture identifies several practical challenges with RLHF:

1. **Requires training a reward model** -- this makes it a 2-stage process (train RM, then do RL), adding complexity
2. **Many hyperparameters to tune** -- lambda/beta for KL penalty, learning rates, clipping epsilon, etc.
3. **Training instability** -- RL training is notoriously unstable and can diverge
4. **Metric to monitor training** -- it is hard to know if training is going well (reward can increase while quality decreases due to reward hacking)
5. **Need diversity in completions** -- the RL exploration requires generating diverse responses
6. **Not abundantly clear why preference tuning absolutely needs RL** -- this motivates DPO

### Alternatives to PPO

The lecture mentions several PPO variants and alternatives:
- **REINFORCE** -- a simpler policy gradient method
- **GRPO** -- Group Relative Policy Optimization

**Reference:** "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", Shao et al., 2024.

---

## 7. Best-of-N (BoN) Sampling

Before introducing DPO, the lecture presents **Best-of-N** as a simpler workaround that avoids RL entirely.

### How BoN Works

**Idea.** Skip the RL step and leverage the reward model scores at inference time instead.

**Strategy:**
1. Given a prompt, generate **several outputs** with the SFT model
2. **Rank** each output using the score given by the reward model
3. **Take the best one** (highest reward score)

### BoN Example

```
Prompt: "Suggest a new activity I could do with my teddy bear."

Output 1: "Of course! Teddy bears not only make awesome       RM --> 0.8  --> Rank #1 (SELECTED)
companions for a delightful sleep, but can also be great
buddies for fun activities. How about you both watch a
movie together?"

Output 2: "I'd suggest you do not spend much time with        RM --> -2   --> Rank #3
your teddy bear at all."

Output 3: "Take your teddy bear on a picnic in your           RM --> 0.2  --> Rank #2
backyard."
```

The highest-scored response (0.8) is selected and returned to the user.

### BoN Limitations

- **Costly at inference time** -- need to generate N responses for every single prompt
- Does not actually improve the model's weights -- it is a filtering strategy, not a training method
- Requires running the reward model at inference time

---

## 8. DPO: Direct Preference Optimization

DPO is the major alternative to RLHF covered in this lecture. It was introduced by Rafailov et al., 2023 in the paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model."

### Motivation for DPO

Three problems motivate moving beyond RL:

1. **Limitations using RL** -- the PPO-KL Penalty objective has 4 separate components (numbered 1-4 in the slides), each requiring a separate model and careful tuning
2. **Best-of-N is costly at inference time** -- it does not improve the model itself
3. **Why don't we train in a supervised fashion?** -- the key question DPO answers

### The DPO Loss Function

DPO rewrites the loss function in a **supervised** way, eliminating the need for a separate reward model:

```
L_DPO(pi_theta; pi_ref) = -E_{(x, y_w, y_l) ~ D} [log sigma(beta * log(pi_theta(y_w | x) / pi_ref(y_w | x)) - beta * log(pi_theta(y_l | x) / pi_ref(y_l | x)))]
```

Where:
- `x` = the prompt
- `y_w` = the winning (preferred) response
- `y_l` = the losing (rejected) response
- `pi_theta` = the policy being trained
- `pi_ref` = the reference (base/SFT) model
- `beta` = temperature parameter controlling deviation from reference
- `sigma` = sigmoid function
- `D` = dataset of preference pairs

### Key Properties of DPO

1. **No need to train a separate reward model** -- there is no `r(x, y)` term in the loss
2. **Operates directly on preference data** -- uses (prompt, winner, loser) triples directly
3. **Similar to Bradley-Terry formulation** -- but with a special kind of "implicit reward"

### The Implicit Reward in DPO

DPO defines an implicit reward:

```
r_theta(x, y) = beta * pi_theta(y | x) / pi_ref(y | x)
```

This means the DPO loss can be rewritten as:

```
L_DPO(pi_theta; pi_ref) = -E_{(x, y_w, y_l) ~ D} [log sigma(r_theta(x, y_w) - r_theta(x, y_l))]
```

**Plain-language explanation:** The implicit reward measures how much more likely the current model is to produce a response compared to the reference model. If the model strongly prefers the winning response over the losing one (relative to the reference), the loss is low. The model learns to increase the probability of preferred responses and decrease the probability of rejected ones.

### Where Does the DPO Formulation Come From?

The lecture walks through the 5-step derivation:

**Step 1: Start from PPO objective**

```
max_{pi_theta} E_{x ~ D, y ~ pi_theta(y|x)} [r_phi(x, y)] - beta * D_KL[pi_theta(y | x) || pi_ref(y | x)]
```

**Step 2: Derive optimal policy**

The closed-form solution for the optimal policy is:

```
pi*(y | x) = (1 / Z(x)) * pi_ref(y | x) * exp((1/beta) * r*(x, y))
```

Where Z(x) is a normalization constant (partition function).

**Step 3: Identify a "reward" term**

Rearranging Step 2 to solve for the reward:

```
r*(x, y) = beta * log(pi*(y | x) / pi_ref(y | x)) + beta * log Z(x)
```

**Step 4: Write Bradley-Terry formulation for this "reward"**

Plugging the reward from Step 3 into the Bradley-Terry preference model:

```
p*(y_w > y_l | x) = 1 / (1 + exp(beta * log(pi*(y_l | x) / pi_ref(y_l | x)) - beta * log(pi*(y_w | x) / pi_ref(y_w | x))))
```

Note: the Z(x) terms cancel out in the difference, which is crucial.

**Step 5: "Infer" DPO loss function**

Taking the negative log-likelihood of the Bradley-Terry model from Step 4 gives us the final DPO loss:

```
L_DPO(pi_theta; pi_ref) = -E_{(x, y_w, y_l) ~ D} [log sigma(beta * log(pi_theta(y_w | x) / pi_ref(y_w | x)) - beta * log(pi_theta(y_l | x) / pi_ref(y_l | x)))]
```

---

## 9. PPO-Based RLHF vs. DPO: Comparison

### Ease of Implementation

| Aspect | RLHF (PPO) | DPO |
|--------|-----------|-----|
| **Training approach** | Multi-stage training (RM then RL) | Supervised learning (single stage) |
| **Extra models needed** | Reward model, value model, base model (3 extra) | Base model is the only extra model needed (1 extra) |
| **Hyperparameters** | Many (epsilon, beta, learning rates, etc.) | Fewer (mainly beta) |
| **Training stability** | RL training can be unstable | Standard supervised training -- more stable |
| **Implementation complexity** | High -- need RL infrastructure | Lower -- standard training loop |

### Performance

**No common absolute consensus.** Performance varies from task to task and is sensitive to implementation.

**Reference:** "Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study", Xu et al., 2024.

---

## 10. Behavior of Preference-Tuned Models

The lecture concludes by showing how preference tuning changes model behavior:

### Instruction-tuned only (SFT):

```
Prompt: "Can I put my teddy bear in the washer?"
Response: "No, it might get damaged. Try hand washing instead."
```

This is factually correct but somewhat cold and minimal.

### Instruction-tuned + Preference-tuned (SFT + RLHF/DPO):

```
Prompt: "Can I put my teddy bear in the washer?"
Response: "It's better not to. Your teddy could get hurt! A gentle hand wash is safer."
```

The preference-tuned model produces a warmer, more empathetic response while conveying the same information. This illustrates how preference tuning adjusts *tone, helpfulness, and style* beyond just factual accuracy.

---

## 11. Summary: The Full Alignment Pipeline

Putting it all together, the modern LLM alignment pipeline looks like:

```
[Pretrained LLM]
       |
       v
[Supervised Finetuning (SFT)]  -- Train on (instruction, response) pairs
       |
       v
[Preference Tuning]            -- Choose one of:
   |                              - RLHF (PPO): Train reward model, then RL
   |                              - DPO: Direct supervised optimization on preference pairs
   |                              - BoN: Inference-time filtering (no training)
   v
[Aligned LLM]                  -- Model that is helpful, harmless, and honest
```

### Method Comparison Table

| Method | Requires RM? | Training Type | # Models | Complexity | When to Use |
|--------|-------------|---------------|----------|------------|-------------|
| **SFT only** | No | Supervised | 1 | Low | Basic instruction following |
| **RLHF (PPO)** | Yes | RL | 4 | High | Maximum control, large-scale alignment |
| **DPO** | No | Supervised | 2 | Medium | Simpler alignment with good results |
| **BoN** | Yes | None (inference) | 2 | Low (training) / High (inference) | Quick improvement without retraining |

---

## 12. Key Formulas Reference

### Bradley-Terry Preference Model
```
P(y_1 > y_2 | x) = sigma(r(x, y_1) - r(x, y_2))
```

### PPO-RLHF Objective
```
L(theta) = -[r(x, y_hat) - lambda * KL(pi_theta(y_hat | x) || pi_ref(y_hat | x))]
```

### KL Divergence
```
KL(P || Q) = sum_i p_i * log(p_i / q_i)
```

### PPO-Clip Objective
```
L^CLIP(theta) = E_t[min(r_t(theta) * A_hat_t, clip(r_t(theta), 1 - epsilon, 1 + epsilon) * A_hat_t)]
    where r_t(theta) = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t)
```

### PPO-KL Penalty Objective
```
L^KLPEN(theta) = E_t[(pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t)) * A_hat_t - beta * KL[pi_theta_ref(. | s_t), pi_theta(. | s_t)]]
```

### DPO Loss Function
```
L_DPO(pi_theta; pi_ref) = -E_{(x, y_w, y_l) ~ D} [log sigma(beta * log(pi_theta(y_w | x) / pi_ref(y_w | x)) - beta * log(pi_theta(y_l | x) / pi_ref(y_l | x)))]
```

### DPO Implicit Reward
```
r_theta(x, y) = beta * (pi_theta(y | x) / pi_ref(y | x))
```

### DPO Optimal Policy (from derivation)
```
pi*(y | x) = (1 / Z(x)) * pi_ref(y | x) * exp((1 / beta) * r*(x, y))
```

---

## 13. Key Papers Referenced

| Paper | Authors | Year | Topic |
|-------|---------|------|-------|
| "Training language models to follow instructions with human feedback" | Ouyang et al. | 2022 | InstructGPT / RLHF pipeline |
| "Proximal Policy Optimization Algorithms" | Schulman et al. | 2017 | PPO algorithm |
| "High-Dimensional Continuous Control Using Generalized Advantage Estimation" | Schulman et al. | 2015 | GAE method for advantages |
| "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" | Rafailov et al. | 2023 | DPO |
| "RewardBench: Evaluating Reward Models for Language Modeling" | Lambert et al. | 2024 | Reward model evaluation |
| "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" | Shao et al. | 2024 | GRPO and RL alternatives |
| "Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study" | Xu et al. | 2024 | PPO vs DPO comparison |
| "Super Study Guide: Transformers and Large Language Models" | Amidi et al. | 2024 | KL divergence figures |
