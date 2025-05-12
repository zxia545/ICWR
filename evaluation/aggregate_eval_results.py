import os
import glob
import json
import pandas as pd
import argparse


def aggregate_results(eval_folder, output_excel):
    result_rows = []
    files = glob.glob(os.path.join(eval_folder, '*_eval_result.json'))
    for file in files:
        dataset = os.path.basename(file).split('_')[0]
        with open(file) as f:
            data = json.load(f)
        # Count wins for each model
        win_counts = {}
        total = 0
        for item in data:
            winner = item.get('result')
            if winner is not None:
                win_counts[winner] = win_counts.get(winner, 0) + 1
                total += 1
        for model, count in win_counts.items():
            win_rate = count / total if total > 0 else 0
            result_rows.append({
                'dataset': dataset,
                'model': model,
                'win_count': count,
                'total': total,
                'win_rate': win_rate
            })
    df = pd.DataFrame(result_rows)
    df.to_excel(output_excel, index=False)
    print(f"Wrote summary to {output_excel}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate eval results into Excel.')
    parser.add_argument('--eval_folder', type=str, required=True, help='Folder containing *_eval_result.json files')
    parser.add_argument('--output_excel', type=str, required=True, help='Output Excel file path')
    args = parser.parse_args()
    aggregate_results(args.eval_folder, args.output_excel)

if __name__ == "__main__":
    main() 