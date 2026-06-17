import argparse
import csv
import os

RESULTS_BASE_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../results/final/full_pipeline",
)
OUTPUT_PATH = os.path.join(RESULTS_BASE_DIR, "full_pipeline_results.tex")
SENTENCE_OUTPUT_PATH = os.path.join(RESULTS_BASE_DIR, "full_pipeline_sentence.tex")

N_CORRECT_POSES_VALUES = [3, 5, 10, 50]
N_OURS = 3
N_NO_PIPELINE = 50


def read_stats(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        stats = {}
        for row in reader:
            metric = row["metric"]
            stats[metric] = {
                "rate": float(row["rate"]) if row["rate"] else None,
                "count": int(row["count"]) if row["count"] else None,
                "total": int(row["total"]) if row["total"] else None,
            }
    return stats


def read_mpjpe(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["metric"] == "overall_mpjpe_mm":
                return float(row["value"])
    raise ValueError(f"overall_mpjpe_mm not found in {csv_path}")


def bold(val, fmt, is_bold):
    s = f"{val:{fmt}}"
    return r"\textbf{" + s + "}" if is_bold else s


def label_for_n(n):
    if n == N_OURS:
        return r" (ours)"
    return ""


def generate_table(rows, add_motion_ood):
    # rows: list of (n, pose_buffer_good_rate, motion_ood_rate, motion_valid_rate, mpjpe)
    pose_invalid_rates = [1.0 - r[1] for r in rows]
    motion_ood_rates = [r[2] for r in rows]
    motion_rates = [r[3] for r in rows]
    mpjpes = [r[4] for r in rows if r[4] is not None]
    min_pose_invalid = min(pose_invalid_rates)
    min_motion_ood = min(motion_ood_rates)
    max_motion = max(motion_rates)
    min_mpjpe = min(mpjpes) if mpjpes else None

    n_cols = 4 + (1 if add_motion_ood else 0)
    col_spec = "l" + "c" * (n_cols - 1)

    header_cols = [
        r"$N_{\text{req}}$",
        r"$\downarrow$ $\mathcal{H}$ invalid [\%]",
    ]
    if add_motion_ood:
        header_cols.append(r"$\downarrow$ Motion OOD [\%]")
    header_cols += [
        r"$\uparrow$ Motion valid [\%]",
        r"$\downarrow$ MPJPE [mm]",
    ]

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"    \centering")
    lines.append(
        r"    \caption{Full pipeline evaluation results on H36M for varying $N_{\text{req}}$.}"
    )
    lines.append(r"    \label{tab:full_pipeline_results}")
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"        \toprule")
    lines.append("        " + " & ".join(header_cols) + r" \\")
    lines.append(r"        \midrule")
    for n, pose_rate, motion_ood_rate, motion_rate, mpjpe in rows:
        pose_invalid = 1.0 - pose_rate
        pose_str = bold(pose_invalid * 100, ".2f", pose_invalid <= min_pose_invalid)
        motion_ood_str = bold(motion_ood_rate * 100, ".2f", motion_ood_rate <= min_motion_ood)
        motion_str = bold(motion_rate * 100, ".2f", motion_rate >= max_motion)
        if mpjpe is not None:
            mpjpe_str = bold(mpjpe, ".2f", min_mpjpe is not None and mpjpe <= min_mpjpe)
        else:
            mpjpe_str = "--"
        n_label = f"{n}{label_for_n(n)}"
        row_cols = [n_label, pose_str]
        if add_motion_ood:
            row_cols.append(motion_ood_str)
        row_cols += [motion_str, mpjpe_str]
        lines.append("        " + " & ".join(row_cols) + r" \\")
    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines) + "\n"


def generate_sentence(rows):
    row_by_n = {r[0]: r for r in rows}
    if N_OURS not in row_by_n or N_NO_PIPELINE not in row_by_n:
        return None

    _, pose_ours, _, _, mpjpe_ours = row_by_n[N_OURS]
    _, pose_base, _, _, mpjpe_base = row_by_n[N_NO_PIPELINE]

    # Relative increase in invalid pose buffer rate: (invalid_ours - invalid_base) / invalid_base
    invalid_ours = 1.0 - pose_ours
    invalid_base = 1.0 - pose_base
    invalid_pct = (invalid_ours - invalid_base) / invalid_base * 100

    # Relative change in MPJPE (ours vs baseline)
    mpjpe_pct = None
    if mpjpe_ours is not None and mpjpe_base is not None:
        mpjpe_pct = (mpjpe_ours - mpjpe_base) / mpjpe_base * 100

    invalid_str = f"{abs(invalid_pct):.1f}"
    mpjpe_str = f"{abs(mpjpe_pct):.1f}" if mpjpe_pct is not None else r"0 \todo{}"

    sentence = (
        r"Our results in~\cref{tab:full_pipeline_results} show that our OOD pipeline "
        r"reduces the rate of invalid pose buffers "
        r"$\sum_{i=K_I - N_{\text{req}}+1}^{K_I} v_i < N_{\text{req}}$ "
        rf"by \SI{{{invalid_str}}}{{\percent}} "
        rf"while only increasing the average MPJPE by \SI{{{mpjpe_str}}}{{\percent}}."
    )
    return sentence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--add_motion_ood",
        action="store_true",
        help="Include the Motion OOD [%%] column in the table.",
    )
    args = parser.parse_args()

    rows = []
    for n in N_CORRECT_POSES_VALUES:
        csv_path = os.path.join(
            RESULTS_BASE_DIR,
            f"n_correct_poses_required_{n}",
            "motion_validity_stats.csv",
        )
        if not os.path.exists(csv_path):
            print(f"Skipping n={n}: {csv_path} not found.")
            continue
        stats = read_stats(csv_path)

        pose_buffer_good_rate = stats["pose_buffer_good_rate"]["rate"]

        no_motion = stats["no_motion_output_total"]
        no_motion_invalid = stats["no_motion_output_invalid_motion"]
        motion_ood_rate = no_motion_invalid["count"] / no_motion["total"]
        motion_valid_rate = 1.0 - no_motion["count"] / no_motion["total"]

        mpjpe_path = os.path.join(
            RESULTS_BASE_DIR,
            f"n_correct_poses_required_{n}",
            "motion_prediction",
            "mpjpe_results_test.csv",
        )
        mpjpe = read_mpjpe(mpjpe_path) if os.path.exists(mpjpe_path) else None

        rows.append((n, pose_buffer_good_rate, motion_ood_rate, motion_valid_rate, mpjpe))

    if not rows:
        print("No results found.")
        return

    table = generate_table(rows, args.add_motion_ood)
    with open(OUTPUT_PATH, "w") as f:
        f.write(table)
    print(f"Saved table to {OUTPUT_PATH}")
    print()
    print(table)

    sentence = generate_sentence(rows)
    if sentence:
        with open(SENTENCE_OUTPUT_PATH, "w") as f:
            f.write(sentence + "\n")
        print(f"Saved sentence to {SENTENCE_OUTPUT_PATH}")
        print()
        print(sentence)


if __name__ == "__main__":
    main()
