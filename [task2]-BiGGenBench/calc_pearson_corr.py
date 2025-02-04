import numpy as np
import krippendorff
import pandas as pd
from absl import app, flags
from termcolor import colored
from collections import Counter
from IPython.display import display
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error


FLAGS = flags.FLAGS
flags.DEFINE_string("model_name", "llama-3-1-70b-instruct", "")
flags.DEFINE_string("input_file_name", "[baseline]-3-[flask]", "")
flags.DEFINE_boolean("drop_null_scores", False, "")
flags.DEFINE_boolean("use_spearman", False, "")


def mark_p_value(p_value):
    if p_value < 0.001:
        return "***"
    # elif p_value < 0.01:
    #     return "**"
    # elif p_value < 0.05:
    #     return "*"
    else:
        return ""

def get_score_field(file_name):
    if "[baseline]" in file_name or "[ablation]" in file_name or "[oracle]" in file_name:
        return "evaluator_LM-score"
    elif file_name.startswith("[SPRI]"):
        return "[SPRI]-score"
    else:
        raise ValueError(f"Unknown input file name: {file_name}")

def main(argv):
    print ("="*120)
    print (colored(f"model_name: {FLAGS.model_name}", "yellow"))
    print (colored(f"input_file_name: {FLAGS.input_file_name}", "green"))

    score_field = get_score_field(FLAGS.input_file_name)

    results_df = pd.read_json(f"./outputs/{FLAGS.model_name}/{FLAGS.input_file_name}.jsonl", lines=True)
    if len(results_df) != len(results_df[score_field].dropna()):
        print (colored (f"{len(results_df) - len(results_df[score_field].dropna())} scores not parsable!", "red"))
        results_df = results_df.dropna(subset=[score_field])

    print ("-"*120)
    print (colored(f"Total number of samples: {len(results_df)}", "green"))
    print (colored(Counter(results_df[score_field].tolist()), "blue"))
    print ("-"*120)

    correlation_results = []

    for capability in [
        "instruction_following", "grounding", "reasoning", "planning", "refinement", "safety", "theory_of_mind", "tool_usage"
    ]:
        results_df_filtered = results_df[results_df["capability"] == capability]

        human_scores = results_df_filtered["human_score"].to_list()
        model_scores = results_df_filtered[score_field].to_list()

        if FLAGS.use_spearman:
            correlation, p_value = spearmanr(human_scores, model_scores)
            corr_title = "Spearman Correlation"
        else:
            correlation, p_value = pearsonr(human_scores, model_scores)
            corr_title = "Pearson Correlation"

        mae = mean_absolute_error(human_scores, model_scores)

        ### Krippendorff's Alpha
        annotations = np.array([human_scores, model_scores]).T
        k_alpha = krippendorff.alpha(reliability_data=annotations.T, level_of_measurement='ratio')

        # Append results to the list
        correlation_results.append({
            "Capability": capability,
            "# Samples": len(human_scores),
            "MAE": round(mae, 3),
            "Krippendorff's Alpha": round(k_alpha, 3),
            corr_title: round(correlation, 3),
            "Significance": mark_p_value(p_value),
            "p_value": p_value,
        })

    correlation_df = pd.DataFrame(correlation_results).set_index("Capability")
    average_mae = correlation_df["MAE"].mean()
    average_k_alpha = correlation_df["Krippendorff's Alpha"].mean()
    average_correlation = correlation_df[corr_title].mean()
    correlation_df.loc["Average"] = {
        "# Samples": len(results_df),
        "MAE": round(average_mae, 3),
        "Krippendorff's Alpha": round(average_k_alpha, 3),
        corr_title: round(average_correlation, 3),
        "p_value": "",
        "Significance": ""
    }
    display(correlation_df.transpose())
    print ("="*120)

    if FLAGS.drop_null_scores:
        results_df = results_df.drop_duplicates(subset=["id", "model_name"])
        results_df.to_json(f"./outputs/{FLAGS.model_name}/{FLAGS.input_file_name}.jsonl", orient="records", lines=True)
        print (colored("Dropped null scores and saved the file!", "green"))


if __name__ == '__main__':
    app.run(main)