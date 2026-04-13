# AutoDAN (Simplified)

## Overview
AutoDAN generates **stealthy jailbreak prompts** by treating prompt generation as an **evolutionary optimization problem** over natural language.

Instead of gradients, it uses a **hierarchical genetic algorithm (GA)** that operates on:
- words (micro level)
- sentences (macro level)

---

## Core Idea
Start from a known jailbreak prompt (e.g., DAN), then iteratively:
1. Generate variations  
2. Evaluate how well they bypass safety  
3. Keep and refine the best ones  

---

## Algorithm

### Input
- Prototype jailbreak prompt \(J_p\)
- Target harmful query \(Q\)

---

### Initialization
- Create a population by **LLM-based rewriting** of \(J_p\)
- Ensures diversity while keeping meaning

---

### Loop

Repeat until success or max iterations:

#### 1. Fitness Evaluation
- For each prompt \(J\):
  - Query model with \((J, Q)\)
  - Score based on **whether model answers vs refuses**

---

#### 2. Word-level Optimization
- Track words appearing in successful prompts
- Replace weaker words with high-scoring synonyms

---

#### 3. Selection
- Keep top-performing prompts (elitism)
- Sample others proportional to fitness

---

#### 4. Crossover
- Combine prompts by **swapping sentences**

---

#### 5. Mutation
- Use LLM to **rewrite prompts naturally**
- Maintains fluency and meaning

---

### Termination
- Stop when model no longer refuses  
- Or iteration limit is reached

---

## Key Properties

- **Semantic (not token-level)** → produces natural prompts  
- **Stealthy** → bypasses perplexity-based defenses  
- **Transferable** → works across models  
- **Hierarchical search** → avoids local minima  

---

## Pseudocode

```python
population = diversify(prototype_prompt)

while not success:
    score each prompt

    # word-level update
    update words using momentum scoring

    # selection
    keep top prompts
    sample parents

    # evolution
    crossover sentences
    mutate via LLM rewrite

return best prompt
```

---

## One-line Summary
AutoDAN evolves jailbreak prompts via **hierarchical genetic search over natural language**, optimizing for both effectiveness and stealth.

