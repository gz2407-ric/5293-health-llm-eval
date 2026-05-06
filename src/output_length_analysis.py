import re
import pandas as pd

from utils import PROJECT_ROOT


def word_count(text):
    return len(re.findall(r"\b\w+\b", str(text)))


def sentence_count(text):
    return len([s for s in re.split(r"[.!?]+", str(text)) if s.strip()])


def main():
    input_path = PROJECT_ROOT / "results" / "model_outputs.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}")

    outputs = pd.read_csv(input_path).fillna("")
    outputs["word_count"] = outputs["model_output"].apply(word_count)
    outputs["sentence_count"] = outputs["model_output"].apply(sentence_count)
    outputs["char_count"] = outputs["model_output"].astype(str).str.len()

    detail_path = PROJECT_ROOT / "results" / "output_length_details.csv"
    outputs[[
        "scenario_id", "system_name", "word_count", "sentence_count", "char_count"
    ]].to_csv(detail_path, index=False)

    summary = (
        outputs.groupby("system_name")
        .agg(
            mean_word_count=("word_count", "mean"),
            median_word_count=("word_count", "median"),
            mean_sentence_count=("sentence_count", "mean"),
            mean_char_count=("char_count", "mean"),
            n=("scenario_id", "count"),
        )
        .reset_index()
    )

    summary_path = PROJECT_ROOT / "results" / "output_length_analysis.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved output length details to {detail_path}")
    print(f"Saved output length summary to {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
