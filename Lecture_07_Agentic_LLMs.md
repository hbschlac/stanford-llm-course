# Lecture 7: Agentic LLMs

**Course:** Stanford CME 295 -- Large Language Models  
**Date:** November 14, 2025  
**YouTube:** [https://www.youtube.com/watch?v=h-7S6HNq0Vg](https://www.youtube.com/watch?v=h-7S6HNq0Vg)  
**Slides:** [https://cme295.stanford.edu/slides/fall25-cme295-lecture7.pdf](https://cme295.stanford.edu/slides/fall25-cme295-lecture7.pdf)

**Topics covered:**
- Retrieval-Augmented Generation (RAG)
- Advanced RAG techniques
- Function calling / tool use
- LLM agents
- ReAct framework

> **Note:** This file was assembled from the lecture topic outline and domain knowledge. Slide PDF and YouTube transcript were not directly accessible at time of creation. Update with exact slide content when possible.

---

## 1. Retrieval-Augmented Generation (RAG)

### 1.1 Motivation

LLMs have several well-known limitations that RAG addresses:

- **Knowledge cutoff:** Models are trained on data up to a fixed date and cannot access newer information.
- **Hallucination:** Models may generate plausible-sounding but factually incorrect content.
- **Domain specificity:** General-purpose models lack deep knowledge of proprietary or niche domains.
- **Context window limits:** Even with large context windows, stuffing all relevant documents is infeasible.

RAG bridges these gaps by **retrieving relevant external documents at inference time** and conditioning the model's generation on that retrieved context.

### 1.2 Core RAG Pipeline

```
User Query --> Retriever --> Top-k Documents --> [Query + Documents] --> LLM --> Response
```

**Three stages:**

1. **Indexing:** Documents are chunked, embedded, and stored in a vector database.
2. **Retrieval:** Given a user query, retrieve the top-k most relevant chunks.
3. **Generation:** The LLM generates a response conditioned on both the query and retrieved context.

### 1.3 Retrieval Methods

#### Dense Retrieval

- Documents and queries are encoded into dense vector representations using an embedding model (e.g., OpenAI `text-embedding-3-small`, Sentence-BERT, E5, GTE).
- Similarity is computed via cosine similarity or dot product:

$$\text{sim}(q, d) = \frac{q \cdot d}{\|q\| \|d\|}$$

- Approximate Nearest Neighbor (ANN) search with libraries like FAISS, Annoy, or HNSW for efficient retrieval at scale.

#### Sparse Retrieval

- Traditional keyword-based methods: **BM25**, TF-IDF.
- BM25 scoring formula:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

Where:
- `f(t, d)` = term frequency of term `t` in document `d`
- `|d|` = document length
- `avgdl` = average document length
- `k_1`, `b` = tunable parameters (typically k_1 = 1.2, b = 0.75)
- `IDF(t)` = inverse document frequency

#### Hybrid Retrieval

- Combines dense and sparse retrieval scores (e.g., via Reciprocal Rank Fusion or linear combination).
- Often outperforms either method alone.

### 1.4 Chunking Strategies

| Strategy | Description | Trade-offs |
|---|---|---|
| Fixed-size chunks | Split by token/character count with optional overlap | Simple but may break semantic units |
| Sentence-based | Split on sentence boundaries | Preserves grammar but may lose context |
| Paragraph-based | Split on paragraph boundaries | Good semantic coherence |
| Recursive/hierarchical | Split by headings, then paragraphs, then sentences | Best semantic structure but more complex |
| Semantic chunking | Use embeddings to detect topic shifts | Most context-aware but expensive |

**Chunk size considerations:**
- Too small: loses context, increases retrieval noise.
- Too large: dilutes relevance signal, wastes context window.
- Typical range: 256--1024 tokens with 10--20% overlap.

### 1.5 Embedding Models

Key properties of good embedding models for RAG:
- **Semantic similarity alignment:** Similar meaning maps to nearby vectors.
- **Instruction-tuned:** Some models (e.g., E5-instruct) accept task-specific prefixes.
- **Dimensionality vs. performance trade-off:** Common dimensions: 384, 768, 1024, 1536.

---

## 2. Advanced RAG Techniques

### 2.1 Query Transformation

Improve retrieval by transforming the user query before searching:

- **Query rewriting:** Use an LLM to rephrase the query for better retrieval.
- **HyDE (Hypothetical Document Embeddings):** Generate a hypothetical answer, then use its embedding to retrieve real documents. The intuition: a generated answer is closer in embedding space to relevant documents than the original question.
- **Multi-query generation:** Generate multiple reformulations of the query, retrieve for each, then merge results.
- **Step-back prompting:** Ask a more general version of the question to retrieve broader context.

### 2.2 Re-ranking

After initial retrieval, apply a more powerful model to re-score and re-order results:

- **Cross-encoder re-rankers:** A model (e.g., Cohere Rerank, BGE-reranker) that takes `(query, document)` pairs and produces a relevance score. More accurate than bi-encoder similarity but too expensive for full corpus search.
- **Pipeline:** Retrieve top-100 with bi-encoder, re-rank to top-10 with cross-encoder.
- **LLM-as-judge re-ranking:** Prompt an LLM to score or rank retrieved passages.

### 2.3 Context Compression and Filtering

- **Extractive compression:** Pull out only the most relevant sentences/spans from retrieved chunks.
- **Abstractive compression:** Use an LLM to summarize retrieved documents before passing to the generator.
- **Relevance filtering:** Remove retrieved documents below a similarity threshold.

### 2.4 Multi-hop / Iterative RAG

For complex questions requiring information synthesis across multiple documents:

1. Retrieve documents for initial sub-question.
2. Use retrieved info to formulate follow-up queries.
3. Retrieve additional documents.
4. Synthesize final answer from all retrieved context.

### 2.5 RAG Evaluation

| Metric | What It Measures |
|---|---|
| **Context Relevance** | Are the retrieved documents relevant to the query? |
| **Faithfulness / Groundedness** | Is the generated answer supported by the retrieved context? |
| **Answer Relevance** | Does the generated answer address the original question? |
| **Context Recall** | Does the retrieved context cover all aspects of the ground truth? |
| **Context Precision** | What fraction of retrieved documents are actually relevant? |

Frameworks: RAGAS, TruLens, DeepEval.

### 2.6 RAG vs. Fine-tuning vs. Long Context

| Approach | Best For | Limitations |
|---|---|---|
| **RAG** | Dynamic/updating knowledge, attribution, large corpora | Retrieval errors propagate; latency overhead |
| **Fine-tuning** | Changing model behavior/style, structured outputs | Expensive; doesn't add factual knowledge well |
| **Long context** | Small, fixed document sets | Cost scales with context; no corpus-level search |

These approaches are complementary, not mutually exclusive.

---

## 3. Function Calling / Tool Use

### 3.1 Concept

Function calling enables LLMs to interact with external systems by generating structured outputs that map to predefined function signatures.

**Key insight:** The LLM does not execute functions -- it generates the function name and arguments in a structured format (typically JSON). An orchestration layer executes the function and returns results to the LLM.

### 3.2 Function Calling Flow

```
User message
    |
    v
LLM decides: respond directly OR call a function
    |
    v (if function call)
LLM outputs: { "name": "function_name", "arguments": { ... } }
    |
    v
Orchestrator executes function, gets result
    |
    v
Result is appended to conversation
    |
    v
LLM generates final response incorporating function result
```

### 3.3 Function / Tool Definition Schema

Functions are defined with:
- **Name:** Identifier for the function.
- **Description:** Natural language description of what the function does (critical for the LLM to decide when to use it).
- **Parameters:** JSON Schema defining expected arguments, types, required fields.

Example (OpenAI format):
```json
{
  "name": "get_weather",
  "description": "Get the current weather for a given location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City and state, e.g. San Francisco, CA"
      },
      "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"]
      }
    },
    "required": ["location"]
  }
}
```

### 3.4 Parallel and Sequential Function Calls

- **Parallel:** The model may request multiple independent function calls in a single turn (e.g., get weather for three cities simultaneously).
- **Sequential / chained:** The output of one function call feeds into the next (e.g., search for a user, then fetch their orders).

### 3.5 Tool Use Best Practices

- Write clear, specific function descriptions.
- Keep the number of tools manageable (model performance degrades with too many tools).
- Validate function arguments before execution.
- Handle errors gracefully and return informative error messages to the LLM.
- Use `tool_choice` / `function_call` parameters to force or suggest tool usage when appropriate.

---

## 4. LLM Agents

### 4.1 Definition

An **agent** is a system that uses an LLM as a reasoning engine to autonomously:
1. **Plan** a sequence of actions to achieve a goal.
2. **Act** by invoking tools / functions.
3. **Observe** the results.
4. **Iterate** until the goal is achieved or a stopping condition is met.

Agents go beyond simple function calling by adding a **loop**: the LLM repeatedly reasons about its current state and decides the next action.

### 4.2 Agent Architecture

```
                  +------------------+
                  |   User Goal      |
                  +--------+---------+
                           |
                  +--------v---------+
              +-->|   LLM (Reason)   |---+
              |   +--------+---------+   |
              |            |             |
              |   +--------v---------+   |
              |   |   Plan / Decide  |   |
              |   +--------+---------+   |
              |            |             |
              |   +--------v---------+   |
              |   |  Execute Action  |   |
              |   |  (Tool / API)    |   |
              |   +--------+---------+   |
              |            |             |
              |   +--------v---------+   |
              +---+  Observe Result  |<--+
                  +--------+---------+
                           |
                  +--------v---------+
                  | Final Answer     |
                  +------------------+
```

### 4.3 Key Components of an Agent

| Component | Description |
|---|---|
| **LLM backbone** | The reasoning engine (e.g., GPT-4, Claude, Gemini) |
| **System prompt** | Instructions defining the agent's role, constraints, and behavior |
| **Tools** | Functions the agent can call (APIs, databases, code execution, web search) |
| **Memory** | Conversation history, scratchpad, or external memory store |
| **Planning strategy** | How the agent decomposes tasks and sequences actions |
| **Stopping criteria** | When to terminate the loop (answer found, max iterations, error) |

### 4.4 Memory in Agents

- **Short-term memory:** The conversation/context window. Limited by token count.
- **Long-term memory:** External stores (vector databases, key-value stores) that persist across sessions.
- **Working memory / scratchpad:** An explicit area for the agent to track intermediate reasoning and state.

### 4.5 Planning Strategies

- **Single-step:** Model decides one action at a time (ReAct-style).
- **Plan-then-execute:** Model generates a full plan upfront, then executes each step.
- **Adaptive planning:** Model generates a plan but can revise it based on intermediate results.
- **Tree of Thought:** Explore multiple reasoning paths in parallel, evaluate, and select the best.

---

## 5. ReAct Framework

### 5.1 Overview

**ReAct** (Reason + Act) is a prompting framework introduced by Yao et al. (2023) that interleaves reasoning traces and actions in a single generation loop.

**Key insight:** By explicitly generating reasoning ("thoughts") before each action, the model can:
- Better plan its next step.
- Track progress toward the goal.
- Handle exceptions and adapt.

### 5.2 The ReAct Loop

Each iteration consists of three components:

1. **Thought:** The model reasons about the current state, what information is needed, and what to do next.
2. **Action:** The model selects and parameterizes a tool/action.
3. **Observation:** The environment returns the result of the action.

```
Thought 1: I need to find the population of France to answer this question.
Action 1: Search["population of France 2024"]
Observation 1: The population of France in 2024 is approximately 68.4 million.

Thought 2: Now I have the population. The user asked for an approximate number.
Action 2: Finish["The population of France is approximately 68.4 million."]
```

### 5.3 ReAct vs. Alternatives

| Approach | Reasoning | Acting | Strengths | Weaknesses |
|---|---|---|---|---|
| **Standard prompting** | Implicit | None | Simple | No tool access; hallucination-prone |
| **Chain-of-Thought (CoT)** | Explicit | None | Better reasoning | Still no external info; can hallucinate facts |
| **Act-only** | None | Yes | Can use tools | No explicit planning; may take wrong actions |
| **ReAct** | Explicit | Yes | Grounded reasoning + tool use | More tokens; requires good tool design |

### 5.4 ReAct Prompt Template (Simplified)

```
You are a helpful assistant. You have access to the following tools:
- Search[query]: Searches the web for information.
- Calculator[expression]: Evaluates a mathematical expression.
- Finish[answer]: Returns the final answer.

Use the following format:

Thought: <your reasoning about what to do next>
Action: <tool_name>[<input>]
Observation: <result from the tool>
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer.
Action: Finish[<final answer>]

Question: {user_question}
```

### 5.5 ReAct Implementation Considerations

- **Max iterations:** Set a limit to prevent infinite loops (commonly 5--15 steps).
- **Error handling:** If a tool returns an error, the model should reason about it and try an alternative approach.
- **Observation truncation:** Tool outputs may be very long; truncate or summarize to fit context.
- **Grounding:** The observation step forces the model to incorporate real data rather than hallucinating.

---

## 6. Putting It All Together: Agentic RAG

Agentic RAG combines RAG with agent capabilities:

1. The agent receives a user query.
2. It **reasons** about what information is needed (Thought).
3. It **retrieves** relevant documents using RAG tools (Action).
4. It **evaluates** whether the retrieved information is sufficient (Observation + Thought).
5. If not sufficient, it may:
   - Reformulate the query and retrieve again.
   - Use a different tool (web search, database query, calculator).
   - Decompose the question into sub-questions.
6. Once sufficient information is gathered, it generates the final answer.

### Agentic RAG vs. Naive RAG

| Feature | Naive RAG | Agentic RAG |
|---|---|---|
| Query handling | Single retrieval pass | Multi-step, adaptive retrieval |
| Query transformation | None | Rewriting, decomposition |
| Tool use | Retriever only | Multiple tools (search, code, APIs) |
| Self-reflection | None | Evaluates retrieval quality, retries |
| Complex queries | Struggles | Handles multi-hop reasoning |

---

## 7. Real-World Agent Frameworks

| Framework | Key Features |
|---|---|
| **LangChain / LangGraph** | Popular open-source; chains, agents, tool integrations; graph-based workflows |
| **LlamaIndex** | Data framework for RAG; document loaders, indices, query engines |
| **AutoGen (Microsoft)** | Multi-agent conversations; agents can collaborate |
| **CrewAI** | Role-based multi-agent orchestration |
| **OpenAI Assistants API** | Built-in tool use, code interpreter, file search |
| **Anthropic Claude Tool Use** | Native function calling with Claude models |

---

## 8. Key Takeaways

1. **RAG** addresses LLM knowledge limitations by grounding generation in retrieved documents.
2. **Advanced RAG** techniques (query transformation, re-ranking, iterative retrieval) significantly improve quality.
3. **Function calling** enables LLMs to interact with external systems via structured outputs.
4. **Agents** add a reasoning loop around tool use, enabling autonomous multi-step problem solving.
5. **ReAct** is the foundational framework for interleaving reasoning and action.
6. **Agentic RAG** combines the best of retrieval and agent capabilities for robust, adaptive information access.

---

## 9. Key Definitions

| Term | Definition |
|---|---|
| **RAG** | Retrieval-Augmented Generation -- augmenting LLM generation with retrieved external documents |
| **Dense retrieval** | Using learned vector embeddings and similarity search to find relevant documents |
| **Sparse retrieval** | Using term-frequency-based methods (BM25, TF-IDF) for document retrieval |
| **Hybrid retrieval** | Combining dense and sparse retrieval scores |
| **Re-ranking** | Using a cross-encoder or more powerful model to re-score initially retrieved documents |
| **HyDE** | Hypothetical Document Embeddings -- generating a hypothetical answer to improve retrieval |
| **Function calling** | LLM capability to output structured function invocations rather than free text |
| **Agent** | A system using an LLM to autonomously plan, act, observe, and iterate toward a goal |
| **ReAct** | Reason + Act framework that interleaves explicit reasoning with tool use |
| **Agentic RAG** | RAG enhanced with agent capabilities for adaptive, multi-step retrieval |
| **Chunking** | Splitting documents into smaller segments for embedding and retrieval |
| **Cross-encoder** | A model that jointly encodes query-document pairs for more accurate relevance scoring |
| **Grounding** | Ensuring generated content is supported by retrieved evidence |

---

## 10. References

- Lewis, P. et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS.
- Yao, S. et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models." ICLR.
- Gao, L. et al. (2023). "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE). ACL.
- Schick, T. et al. (2023). "Toolformer: Language Models Can Teach Themselves to Use Tools." NeurIPS.
- Wei, J. et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS.
- Robertson, S. & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." Foundations and Trends in Information Retrieval.
