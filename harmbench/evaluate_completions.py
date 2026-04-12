import transformers
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
import argparse
import os
import csv
from tqdm import tqdm 
import numpy as np

from vllm import LLM, SamplingParams

import eval_utils
from eval_utils import compute_results_hashing, compute_results_classifier, compute_results_advbench
from summary_utils import build_results_summary, get_results_summary_path, write_json
# Set this to disable warning messages in the generation mode.
transformers.utils.logging.set_verbosity_error()

def parse_args():
    parser = argparse.ArgumentParser(description="Running red teaming with baseline methods.")
    parser.add_argument("--cls_path", type=str, default='cais/HarmBench-Llama-2-13b-cls',
                        help="The name or path of the classifier for computing ASR")
    parser.add_argument("--behaviors_path", type=str, default='./data/behavior_datasets/harmbench_behaviors_text_all.csv',
                        help="The path to the behaviors file")
    parser.add_argument("--completions_path", type=str,
                        help="The path to the completions to evaluate")
    parser.add_argument("--save_path", type=str,
                        help="The path for saving results")
    parser.add_argument("--include_advbench_metric", action="store_true",
                        help="Whether to include the AdvBench refusal metric")
    parser.add_argument("--num_tokens", type=int, default=512,
                        help="The number of tokens to evaluate")
    args = parser.parse_args()
    return args


def main():
    # ========== load arguments and config ========== #
    args = parse_args()
    print(args)
    
    # ========== load behaviors (for tags and context strings) ========== #
    with open(args.behaviors_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        behaviors = list(reader)
    
    # convert to dictionary mapping behavior ID field to behavior dict
    behaviors = {b['BehaviorID']: b for b in behaviors}

    # ========== initialize results ========== #
    with open(args.completions_path, 'r') as f:
        completions = json.load(f)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.cls_path)
    tokenizer.truncation_side="right"

    # Clip the 'generation' field of the completions to have a maximum of num_tokens tokens
    for behavior_id, completion_list in completions.items():
        for completion in completion_list:
            generation = completion['generation']
            tokenized_text = tokenizer.encode(generation, max_length=args.num_tokens, truncation=True)
            clipped_generation = tokenizer.decode(tokenized_text, skip_special_tokens=True)
            completion['generation'] = clipped_generation

    # ========== evaluate completions ========== #
    cls = LLM(model=args.cls_path, tensor_parallel_size=1, gpu_memory_utilization=0.95)
    cls_tokenizer = getattr(cls.llm_engine.tokenizer, "tokenizer", cls.llm_engine.tokenizer)
    cls_tokenizer.truncation_side = "left"

    cls_params = SamplingParams(temperature=0.0, max_tokens=1)

    # ── Build one big batch for the classifier instead of per-behavior calls ──
    # Collect (behavior_id, index_in_data, input_str) for all non-hash behaviors
    batch_keys = []   # (behavior_id, idx) for each input in the big batch
    batch_inputs = [] # classifier prompt strings

    results = {}
    hash_behavior_ids = set()

    for behavior_id, data in completions.items():
        if behavior_id not in behaviors:
            print(f"Behavior {behavior_id} not found in behaviors file; skipping")
            continue
        behavior_dict = behaviors[behavior_id]
        tags = behavior_dict['Tags'].split(', ')

        if 'hash_check' in tags:
            # Handle hash-check behaviors separately (no classifier needed)
            current_results = compute_results_hashing(behavior_dict, data)
            if args.include_advbench_metric:
                current_results = compute_results_advbench(current_results, data)
            results[behavior_id] = current_results
            hash_behavior_ids.add(behavior_id)
        else:
            behavior_string = behavior_dict['Behavior']
            context_string = behavior_dict.get('ContextString', '')
            for idx, d in enumerate(data):
                if 'context' in tags or 'multimodal' in tags:
                    inp = eval_utils.LLAMA2_CLS_PROMPT['prompt_contextual'].format(
                        behavior=behavior_string, generation=d['generation'], context=context_string)
                else:
                    inp = eval_utils.LLAMA2_CLS_PROMPT['prompt'].format(
                        behavior=behavior_string, generation=d['generation'])
                batch_keys.append((behavior_id, idx))
                batch_inputs.append(inp)

    # Run the classifier in one shot over all inputs
    print(f"Running classifier on {len(batch_inputs)} completions in a single batch...")
    batch_outputs = cls.generate(batch_inputs, cls_params, use_tqdm=True)

    # Reconstruct per-behavior results from the flat batch output
    labels_map = {}  # behavior_id -> list of labels (index-ordered)
    for (behavior_id, idx), output in zip(batch_keys, batch_outputs):
        pred = output.outputs[0].text
        label = int(pred.lower() == "yes") if pred.lower() in ("yes", "no") else -1
        if label == -1:
            behavior_string = behaviors[behavior_id]['Behavior']
            print(f"WARNING: Detected label=-1 with {behavior_string}, pred={pred!r}")
        labels_map.setdefault(behavior_id, {})[idx] = label

    for behavior_id, data in completions.items():
        if behavior_id in hash_behavior_ids or behavior_id not in behaviors:
            continue
        current_results = []
        for idx, d in enumerate(data):
            d['label'] = labels_map.get(behavior_id, {}).get(idx, -1)
            current_results.append(d)
        if args.include_advbench_metric:
            current_results = compute_results_advbench(current_results, data)
        results[behavior_id] = current_results

    success_rates = []

    for i, (behavior_id, data) in enumerate(results.items()):
        success_mean = np.mean([d['label'] == 1 for d in data])
        print(f"{i}. {behavior_id} === average asr: {success_mean}")
        success_rates.append(success_mean)
    
    print("Average ASR: ", np.mean(success_rates))

    # Make dirs to output_file if not yet exist
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True) if os.path.dirname(args.save_path) else None
    with open(args.save_path, 'w') as file:
        json.dump(results, file, indent=4)

    summary_path = get_results_summary_path(args.save_path)
    summary = build_results_summary(results)
    write_json(summary_path, summary)
    print(f"Saved summary to {summary_path}")

if __name__ == "__main__":
    main()
