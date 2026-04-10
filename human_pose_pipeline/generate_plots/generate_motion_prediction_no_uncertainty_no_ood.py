import csv
import os

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../results/final/motion_prediction",
)
CSV_PATH_STAGE1 = os.path.join(
    RESULTS_DIR, "stage_1_no_uncertainty_no_ood", "per_time_mpjpe_results_test.csv"
)
CSV_PATH_FINAL = os.path.join(
    RESULTS_DIR, "no_uncertainty_no_ood", "per_time_mpjpe_results_test.csv"
)
OUTPUT_PATH = os.path.join(
    RESULTS_DIR, "no_uncertainty_no_ood", "motion_prediction.tex"
)

# 25 fps: 80ms=frame2, 160ms=frame4, 320ms=frame8, 400ms=frame10
TIME_POINTS = {80: 2, 160: 4, 320: 8, 400: 10}


def read_mpjpe(csv_path):
    results = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[int(row["time_point"])] = float(row["mpjpe_mm"])
    return results


def bold(val, is_bold):
    s = f"{val:.1f}"
    return r"\textbf{" + s + "}" if is_bold else s


def generate_table(stage1, final):
    # Baseline rows: (label, cite, 80, 160, 320, 400)
    baselines = [
        ("Repeating Last-Frame", "guo_2023_BackMLP", 23.8, 44.4, 76.1, 88.2),
        ("One FC", "guo_2023_BackMLP", 14.0, 33.2, 68.0, 81.5),
        ("HisRep", "mao_2020_HistoryRepeats", 10.4, 22.6, 47.1, 58.3),
        ("ST-DGCN", "ma_2022_ProgressivelyGenerating", 10.3, 22.7, 47.4, 58.5),
        ("ST-Trans", "saadatnejad_2024_ReliableHuman", 10.4, 23.4, 48.4, 59.2),
        ("SiMLPe", "guo_2023_BackMLP", 9.6, 21.7, 46.3, 57.3),
    ]

    stage1_vals = [stage1[TIME_POINTS[ms]] for ms in (80, 160, 320, 400)]
    final_vals = [final[TIME_POINTS[ms]] for ms in (80, 160, 320, 400)]

    # Find column-wise minimums across all methods
    all_vals = [[b[2], b[3], b[4], b[5]] for b in baselines] + [stage1_vals, final_vals]
    col_mins = [min(col) for col in zip(*all_vals)]

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"    \centering")
    lines.append(r"    \caption{Motion prediction test results on H36M.}")
    lines.append(r"    \label{tab:prediction-benchmark}")
    lines.append(r"    \begin{tabular}{lcccc}")
    lines.append(r"        \toprule")
    lines.append(r"        & \multicolumn{4}{c}{$\downarrow$ \textbf{MPJPE (mm)}} \\")
    lines.append(r"        \cmidrule(lr){2-5}")
    lines.append(
        r"        \textbf{Method} & \textbf{80\,ms} & \textbf{160\,ms} & \textbf{320\,ms} & \textbf{400\,ms} \\"
    )
    lines.append(r"        \midrule")

    for i, (name, cite, v80, v160, v320, v400) in enumerate(baselines):
        vals = [v80, v160, v320, v400]
        cells = [bold(v, v <= col_mins[j]) for j, v in enumerate(vals)]
        row = f"        {name}~\\cite{{{cite}}} & {' & '.join(cells)} \\\\"
        lines.append(row)

    lines.append(r"        \midrule")
    stage1_cells = [bold(v, v <= col_mins[j]) for j, v in enumerate(stage1_vals)]
    lines.append(f"        Ours (stage 1) & {' & '.join(stage1_cells)} \\\\")
    final_cells = [bold(v, v <= col_mins[j]) for j, v in enumerate(final_vals)]
    lines.append(f"        Ours (final) & {' & '.join(final_cells)} \\\\")
    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines) + "\n"


def main():
    stage1 = read_mpjpe(CSV_PATH_STAGE1)
    final = read_mpjpe(CSV_PATH_FINAL)
    table = generate_table(stage1, final)
    with open(OUTPUT_PATH, "w") as f:
        f.write(table)
    print(f"Saved table to {OUTPUT_PATH}")
    print()
    print(table)


if __name__ == "__main__":
    main()
