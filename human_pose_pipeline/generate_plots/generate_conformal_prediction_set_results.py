import csv
import os

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../results/final/motion_prediction/no_uncertainty_no_ood",
)
CSV_PATH_CONFORMAL = os.path.join(RESULTS_DIR, "coverage_stats_conformal_prediction_sets.csv")
CSV_PATH_SARA = os.path.join(RESULTS_DIR, "coverage_stats_sara.csv")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "conformal_prediction_set_results.tex")


def read_overall_stats(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        stats = {row["metric"]: float(row["value"]) for row in reader}
    return stats["overall_coverage_percent"], stats["overall_volume_m3"]


def bold(val, fmt, is_bold):
    s = f"{val:{fmt}}"
    return r"\textbf{" + s + "}" if is_bold else s


def generate_table(sara_coverage, sara_volume, conformal_coverage, conformal_volume):
    # Higher coverage is better, lower volume is better
    sara_coverage_bold = sara_coverage > conformal_coverage
    conformal_coverage_bold = conformal_coverage >= sara_coverage
    sara_volume_bold = sara_volume < conformal_volume
    conformal_volume_bold = conformal_volume <= sara_volume

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"    \centering")
    lines.append(r"    \caption{Conformal prediction set test results on H36M.}")
    lines.append(r"    \label{tab:conformal_prediction_set}")
    lines.append(r"    \begin{tabular}{lcc}")
    lines.append(r"        \toprule")
    lines.append(r"        \textbf{Method} & $\uparrow$ Coverage (\%) & $\downarrow$ Volume ($m^3$) \\")
    lines.append(r"        \midrule")
    lines.append(
        f"        SARA~\\cite{{schepp_2022_SaRATool}} & "
        f"{bold(sara_coverage, '.2f', sara_coverage_bold)} & "
        f"{bold(sara_volume, '.3f', sara_volume_bold)} \\\\"
    )
    lines.append(
        f"        Conformal prediction sets (ours) & "
        f"{bold(conformal_coverage, '.2f', conformal_coverage_bold)} & "
        f"{bold(conformal_volume, '.3f', conformal_volume_bold)} \\\\"
    )
    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines) + "\n"


def main():
    sara_coverage, sara_volume = read_overall_stats(CSV_PATH_SARA)
    conformal_coverage, conformal_volume = read_overall_stats(CSV_PATH_CONFORMAL)

    table = generate_table(sara_coverage, sara_volume, conformal_coverage, conformal_volume)

    with open(OUTPUT_PATH, "w") as f:
        f.write(table)
    print(f"Saved table to {OUTPUT_PATH}")
    print()
    print(table)


if __name__ == "__main__":
    main()
