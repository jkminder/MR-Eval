# GPTFuzz

**Paper**: [GPTFuzz: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts](https://arxiv.org/abs/2309.10253) (Yu et al., 2023)

## Overview

GPTFuzz treats jailbreaking as a **software fuzzing problem**. It maintains a tree of jailbreak prompt templates and iteratively mutates them using an LLM, guided by Monte Carlo Tree Search (MCTS) to focus effort on the most promising branches.

Unlike gradient-based methods (GCG, PEZ), GPTFuzz is fully **black-box** — it only needs the target model's text output, not its weights or logits.

## How It Works

### 1. Seed Initialization
Starts from a corpus of hand-crafted jailbreak templates (`GPTFuzzer.csv`). Each seed becomes a root node in the search tree.

### 2. Selection (MCTS + UCB)
At each iteration, MCTS selects which node to mutate next using the Upper Confidence Bound (UCB) formula:

```
score = reward + ratio * sqrt(log(parent_visits) / node_visits) - alpha * depth
```

This balances **exploitation** (high-reward templates) vs. **exploration** (less-visited branches), with a depth penalty to avoid over-extending chains.

### 3. Mutation (5 LLM-driven operators)
The selected template is mutated by an LLM (e.g., GPT-4) using one of five strategies:

| Operator | What it does |
|---|---|
| `GenerateSimilar` | Creates a stylistically similar template |
| `CrossOver` | Blends two templates into a new one |
| `Expand` | Prepends new sentences to the template |
| `Shorten` | Condenses the template while preserving meaning |
| `Rephrase` | Rewrites the template in different words |

### 4. Evaluation
Each mutated template is filled with the target harmful behavior and queried against the target model. Success is judged by a **RoBERTa-based classifier** (`hubert233/GPTFuzz`) that predicts jailbreak vs. refusal — no access to model internals required.

### 5. Tree Update
Successful mutations are added as child nodes and their reward is backpropagated up the tree, increasing future selection probability for that branch.

## Key Properties

| Property | Value |
|---|---|
| Model access | Black-box (query only) |
| Optimization | MCTS heuristic search |
| Fitness signal | RoBERTa classifier |
| Transferability | High (classifier is model-agnostic) |
| Query budget | Configurable (`max_query`, default 1000) |

## Comparison to Related Methods

| | GPTFuzz | AutoDAN | GCG |
|---|---|---|---|
| Access required | Black-box | White-box (logits) | White-box (gradients) |
| Search strategy | MCTS tree | Genetic algorithm | Greedy token search |
| Fitness signal | Classifier | CE loss on logits | CE loss on logits |
| Transferability | High | Medium | Low |
| Prompt style | Template mutations | DAN-style evolved | Gibberish suffixes |

## Files

```
gptfuzz/
├── gptfuzz.py              # HarmBench integration (TestCasesGenerator)
├── GPTFuzzer.csv           # Initial seed jailbreak templates
└── gptfuzzer/
    └── fuzzer/
        ├── core.py         # Main fuzzing loop + prompt tree
        ├── mutator.py      # 5 mutation operators
        └── selection.py    # MCTS / UCB / EXP3 selection policies
```

## Configuration

From `harmbench/configs/method_configs/GPTFuzz_config.yaml`:
- `max_query`: query budget per behavior
- `max_iteration`: iteration limit
- `energy`: mutations attempted per selected seed
- `max_jailbreak`: stop after N successes
