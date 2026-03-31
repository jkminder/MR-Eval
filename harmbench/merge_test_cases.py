from baselines import get_method_class
import argparse
import json
import os
from summary_utils import build_test_case_summary, write_json

def parse_args():
    parser = argparse.ArgumentParser(description="Merge batched test cases in save_dir into test_cases.json.")
    parser.add_argument("--method_name", type=str, default='GCG',
                        help="The name of the red teaming method")
    parser.add_argument("--save_dir", type=str, default='./test_cases',
                        help="The directory used for saving test cases")
    args = parser.parse_args()

    return args

def main():
    # ========== load arguments and config ========== #
    args = parse_args()
    print(args)
    
    method_name = args.method_name

    # ========== merge test cases ========== #
    print('Merging test cases')
    method_class = get_method_class(method_name)
    method_class.merge_test_cases(args.save_dir)

    test_cases_path = os.path.join(args.save_dir, "test_cases.json")
    logs_path = os.path.join(args.save_dir, "logs.json")
    summary_path = os.path.join(args.save_dir, "summary.json")

    if os.path.exists(test_cases_path):
        with open(test_cases_path, "r", encoding="utf-8") as handle:
            test_cases = json.load(handle)

        if os.path.exists(logs_path):
            with open(logs_path, "r", encoding="utf-8") as handle:
                logs = json.load(handle)
        else:
            logs = {}

        summary = build_test_case_summary(test_cases, logs)
        write_json(summary_path, summary)
        print(f"Saved summary.json to {summary_path}")

if __name__ == "__main__":
    main()
