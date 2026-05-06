import argparse
import pandas as pd
import numpy as np

from utils import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_scenarios", type=int, default=10)
    parser.add_argument("--seed", type=int, default=5293)
    args = parser.parse_args()

    input_path = PROJECT_ROOT / "results" / "model_outputs.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}")

    outputs = pd.read_csv(input_path).fillna("")

    scenario_ids = sorted(outputs["scenario_id"].unique())
    rng = np.random.default_rng(args.seed)
    sampled_ids = sorted(rng.choice(scenario_ids, size=min(args.n_scenarios, len(scenario_ids)), replace=False))

    sample = outputs[outputs["scenario_id"].isin(sampled_ids)].copy()
    sample = sample.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    sample["output_id"] = [f"HV{i+1:03d}" for i in range(len(sample))]

    blind = sample[[
        "output_id",
        "user_profile",
        "user_question",
        "model_output",
    ]].copy()

    for col in ["relevance", "personalization", "guideline_alignment", "safety", "comments"]:
        blind[col] = ""

    mapping = sample[[
        "output_id",
        "scenario_id",
        "system_name",
        "expected_safety_warning",
        "expected_guideline_topic",
    ]].copy()

    blind_path = PROJECT_ROOT / "results" / "human_validation_blind_template.csv"
    mapping_path = PROJECT_ROOT / "results" / "human_validation_mapping.csv"

    blind.to_csv(blind_path, index=False)
    mapping.to_csv(mapping_path, index=False)

    print(f"Saved blind human validation template to {blind_path}")
    print(f"Saved mapping file to {mapping_path}")
    print("Send ONLY the blind template to raters. Keep the mapping file private.")


if __name__ == "__main__":
    main()
