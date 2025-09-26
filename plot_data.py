import cloudpickle
import argparse
import matplotlib.pyplot as plt  # noqa: F401
import numpy as np  # noqa: F401
import jax.numpy as jnp
from collections import defaultdict
import os
from sklearn.metrics import roc_auc_score


from src.datasets import dataloader_from_string


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path", type=str, default="../datasets/", help="root of dataset"
)
parser.add_argument("--scores_file_path", type=str)
parser.add_argument(
    "--ID_dataset",
    type=str,
    choices=[
        "Sinusoidal",
        "UCI",
        "MNIST",
        "FMNIST",
        "SVHN",
        "CIFAR-10",
        "CIFAR-100",
        "CelebA",
        "ImageNet",
    ],
    default="MNIST",
    required=True,
)
parser.add_argument("--OOD_datasets", nargs="+", help="List of OOD datasets to score")
parser.add_argument("--test_batch_size", default=256, type=int)
parser.add_argument("--model_seed", default=420, type=int)


def auroc(scores_id, scores_ood):
    labels = np.zeros(len(scores_id) + len(scores_ood), dtype="int32")
    labels[len(scores_id) :] = 1
    scores = np.concatenate([scores_id, scores_ood])
    return roc_auc_score(labels, scores)


def plot_score_categories(all_data, save_path=None, figsize_per_image=(2, 2.5)):
    """
    Plot low, medium, and high scoring images for ID and OOD data.

    Args:
        all_data: Dictionary with structure {distribution: {'X': images, 'Y': labels, 'scores': scores}}
        save_path: Path to save the figure
        figsize_per_image: Size of each subplot
    """

    # Separate ID and OOD data
    id_data = all_data.get("ID", None)
    ood_data = {k: v for k, v in all_data.items() if k != "ID"}

    if id_data is None:
        print("No ID data found!")
        return

    if not ood_data:
        print("No OOD data found!")
        return

    # Function to get low, medium, high scoring samples
    def get_score_categories(X, Y, scores, n_samples=8):
        scores_np = np.array(scores)
        indices = np.argsort(scores_np)

        n_total = len(scores_np)

        # Get indices for low, medium, high
        low_indices = indices[:n_samples]
        high_indices = indices[-n_samples:]

        # Medium: take samples from the middle third
        mid_start = n_total // 3
        mid_end = 2 * n_total // 3
        mid_indices = indices[mid_start:mid_end]

        # Sample randomly from middle if we have enough samples
        if len(mid_indices) >= n_samples:
            medium_indices = np.random.choice(mid_indices, n_samples, replace=False)
        else:
            medium_indices = mid_indices

        return {
            "low": {
                "X": X[low_indices],
                "Y": Y[low_indices],
                "scores": scores[low_indices],
                "indices": low_indices,
            },
            "medium": {
                "X": X[medium_indices],
                "Y": Y[medium_indices],
                "scores": scores[medium_indices],
                "indices": medium_indices,
            },
            "high": {
                "X": X[high_indices],
                "Y": Y[high_indices],
                "scores": scores[high_indices],
                "indices": high_indices,
            },
        }

    # Get categories for ID data
    id_categories = get_score_categories(id_data["X"], id_data["Y"], id_data["scores"])

    # Combine all OOD data
    ood_X = np.concatenate([data["X"] for data in ood_data.values()], axis=0)
    ood_Y = np.concatenate([data["Y"] for data in ood_data.values()], axis=0)
    ood_scores = np.concatenate([data["scores"] for data in ood_data.values()], axis=0)

    ood_categories = get_score_categories(ood_X, ood_Y, ood_scores)

    # Create the plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    categories = ["low", "medium", "high"]
    data_types = ["ID", "OOD"]

    for row, (data_type, data_cats) in enumerate(
        zip(data_types, [id_categories, ood_categories])
    ):
        for col, category in enumerate(categories):
            ax = axes[row, col]

            cat_data = data_cats[category]
            X_cat = cat_data["X"]
            Y_cat = cat_data["Y"]
            scores_cat = cat_data["scores"]

            # Create a mini subplot grid for the 8 images
            n_images = min(8, len(X_cat))
            cols_mini = 4
            rows_mini = 2

            # Clear the main axis
            ax.clear()
            ax.set_xlim(0, cols_mini)
            ax.set_ylim(0, rows_mini)
            ax.set_aspect("equal")
            ax.axis("off")

            # Plot mini images
            for i in range(n_images):
                if i >= len(X_cat):
                    break

                mini_row = i // cols_mini
                mini_col = i % cols_mini

                # Get image and info
                img = np.array(X_cat[i]).squeeze()
                label = np.argmax(Y_cat[i]) if len(Y_cat[i].shape) > 0 else Y_cat[i]
                score = float(scores_cat[i])

                # Position for mini subplot
                x_pos = mini_col
                y_pos = rows_mini - 1 - mini_row

                # Create mini axes
                mini_ax = fig.add_axes(
                    [
                        ax.get_position().x0
                        + (x_pos / cols_mini) * ax.get_position().width,
                        ax.get_position().y0
                        + (y_pos / rows_mini) * ax.get_position().height,
                        ax.get_position().width / cols_mini * 0.9,
                        ax.get_position().height / rows_mini * 0.7,
                    ]
                )

                mini_ax.imshow(img, cmap="gray", aspect="equal")
                mini_ax.set_title(f"L:{label}\nS:{score:.3f}", fontsize=8, pad=2)
                mini_ax.axis("off")

            # Set main subplot title
            mean_score = np.mean(scores_cat)
            ax.set_title(
                f"{data_type} - {category.title()} Scores\n(Mean: {mean_score:.4f})",
                fontsize=12,
                pad=20,
            )

    plt.suptitle(
        "Score Categories: Low, Medium, High for ID vs OOD Data", fontsize=16, y=0.95
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Score categories plot saved to {save_path}")

    plt.show()
    return fig


def plot_class_specific_comparison(
    all_data, class_id, ood_dataset_name, save_dir="results"
):
    """
    Plot low, medium, high scoring images for a specific class comparing ID vs specific OOD dataset.

    Args:
        all_data: Dictionary with structure {distribution: {'X': images, 'Y': labels, 'scores': scores}}
        class_id: The class ID to analyze (0-9 for most datasets)
        ood_dataset_name: Name of the OOD dataset to compare against
        save_dir: Directory to save the plot
    """

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Get ID data for this class
    if "ID" not in all_data:
        print("No ID data found!")
        return None

    id_data = all_data["ID"]
    id_labels = (
        np.argmax(id_data["Y"], axis=1) if len(id_data["Y"].shape) > 1 else id_data["Y"]
    )
    id_class_mask = id_labels == class_id

    if not np.any(id_class_mask):
        print(f"No ID samples found for class {class_id}")
        return None

    id_class_data = {
        "X": id_data["X"][id_class_mask],
        "Y": id_data["Y"][id_class_mask],
        "scores": id_data["scores"][id_class_mask],
    }

    # Get OOD data for this class
    if ood_dataset_name not in all_data:
        print(f"OOD dataset {ood_dataset_name} not found!")
        return None

    ood_data = all_data[ood_dataset_name]
    ood_labels = (
        np.argmax(ood_data["Y"], axis=1)
        if len(ood_data["Y"].shape) > 1
        else ood_data["Y"]
    )
    ood_class_mask = ood_labels == class_id

    if not np.any(ood_class_mask):
        print(f"No OOD samples found for class {class_id} in {ood_dataset_name}")
        return None

    ood_class_data = {
        "X": ood_data["X"][ood_class_mask],
        "Y": ood_data["Y"][ood_class_mask],
        "scores": ood_data["scores"][ood_class_mask],
    }

    # Function to get low, medium, high scoring samples
    def get_score_categories(X, Y, scores, n_samples=8):
        if len(scores) == 0:
            return None

        scores_np = np.array(scores)
        indices = np.argsort(scores_np)
        n_total = len(scores_np)

        # Adjust n_samples if we have fewer samples
        n_samples = min(n_samples, n_total)

        if n_samples <= 0:
            return None

        # Get indices for low, medium, high
        low_indices = indices[:n_samples]
        high_indices = indices[-n_samples:]

        # Medium: take samples from the middle third
        if n_total >= 3:
            mid_start = n_total // 3
            mid_end = 2 * n_total // 3
            mid_indices = indices[mid_start:mid_end]

            if len(mid_indices) >= n_samples:
                medium_indices = np.random.choice(mid_indices, n_samples, replace=False)
            else:
                medium_indices = mid_indices
        else:
            # If we have very few samples, just use what we have
            medium_indices = indices[:n_samples]

        return {
            "low": {
                "X": X[low_indices],
                "Y": Y[low_indices],
                "scores": scores[low_indices],
            },
            "medium": {
                "X": X[medium_indices],
                "Y": Y[medium_indices],
                "scores": scores[medium_indices],
            },
            "high": {
                "X": X[high_indices],
                "Y": Y[high_indices],
                "scores": scores[high_indices],
            },
        }

    # Get categories for both datasets
    id_categories = get_score_categories(
        id_class_data["X"], id_class_data["Y"], id_class_data["scores"]
    )
    ood_categories = get_score_categories(
        ood_class_data["X"], ood_class_data["Y"], ood_class_data["scores"]
    )

    if id_categories is None or ood_categories is None:
        print(f"Insufficient data for class {class_id} comparison")
        return None

    # Create the plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    categories = ["low", "medium", "high"]
    data_types = ["ID", ood_dataset_name]

    for row, (data_type, data_cats) in enumerate(
        zip(data_types, [id_categories, ood_categories])
    ):
        for col, category in enumerate(categories):
            ax = axes[row, col]

            cat_data = data_cats[category]
            X_cat = cat_data["X"]
            Y_cat = cat_data["Y"]
            scores_cat = cat_data["scores"]

            # Create a mini subplot grid for the images
            n_images = min(8, len(X_cat))
            cols_mini = 4
            rows_mini = 2

            # Clear the main axis
            ax.clear()
            ax.set_xlim(0, cols_mini)
            ax.set_ylim(0, rows_mini)
            ax.set_aspect("equal")
            ax.axis("off")

            # Plot mini images
            for i in range(n_images):
                if i >= len(X_cat):
                    break

                mini_row = i // cols_mini
                mini_col = i % cols_mini

                # Get image and info
                img = np.array(X_cat[i]).squeeze()
                label = np.argmax(Y_cat[i]) if len(Y_cat[i].shape) > 0 else Y_cat[i]
                score = float(scores_cat[i])

                # Position for mini subplot
                x_pos = mini_col
                y_pos = rows_mini - 1 - mini_row

                # Create mini axes
                mini_ax = fig.add_axes(
                    [
                        ax.get_position().x0
                        + (x_pos / cols_mini) * ax.get_position().width,
                        ax.get_position().y0
                        + (y_pos / rows_mini) * ax.get_position().height,
                        ax.get_position().width / cols_mini * 0.9,
                        ax.get_position().height / rows_mini * 0.7,
                    ]
                )

                mini_ax.imshow(img, cmap="gray", aspect="equal")
                mini_ax.set_title(f"L:{label}\nS:{score:.3f}", fontsize=8, pad=2)
                mini_ax.axis("off")

            # Set main subplot title
            mean_score = np.mean(scores_cat)
            ax.set_title(
                f"{data_type} - {category.title()} Scores\n(Mean: {mean_score:.4f})",
                fontsize=12,
                pad=20,
            )

    # Add statistics text
    id_mean = np.mean(id_class_data["scores"])
    id_std = np.std(id_class_data["scores"])
    ood_mean = np.mean(ood_class_data["scores"])
    ood_std = np.std(ood_class_data["scores"])

    stats_text = (
        f"ID: μ={id_mean:.4f}, σ={id_std:.4f} (n={len(id_class_data['scores'])})\n"
    )
    stats_text += f"{ood_dataset_name}: μ={ood_mean:.4f}, σ={ood_std:.4f} (n={len(ood_class_data['scores'])})"

    plt.figtext(
        0.02,
        0.02,
        stats_text,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
    )

    plt.suptitle(
        f"Class {class_id}: ID vs {ood_dataset_name} Score Comparison",
        fontsize=16,
        y=0.95,
    )
    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(save_dir, f"class_{class_id}_ID_vs_{ood_dataset_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()  # Close to save memory
    print(f"Class {class_id} comparison saved to {save_path}")

    return fig


def create_all_class_comparisons(all_data, save_dir="results", num_classes=10):
    """
    Create comparison plots for all classes and all OOD datasets.

    Args:
        all_data: Dictionary with all the data
        save_dir: Directory to save all plots
        num_classes: Number of classes to analyze (default 10 for MNIST, CIFAR-10, etc.)
    """

    # Create main results directory
    os.makedirs(save_dir, exist_ok=True)

    # Get list of OOD datasets
    ood_datasets = [k for k in all_data.keys() if k != "ID"]

    if not ood_datasets:
        print("No OOD datasets found!")
        return

    print(
        f"Creating class-specific comparisons for {len(ood_datasets)} OOD datasets and {num_classes} classes..."
    )

    # Create summary statistics
    summary_stats = {}

    # Create plots for each class and each OOD dataset
    for class_id in range(num_classes):
        for ood_dataset in ood_datasets:
            try:
                fig = plot_class_specific_comparison(
                    all_data, class_id, ood_dataset, save_dir
                )
                if fig is not None:
                    # Collect statistics for summary
                    if ood_dataset not in summary_stats:
                        summary_stats[ood_dataset] = {}

                    # Get class-specific statistics
                    id_data = all_data["ID"]
                    id_labels = (
                        np.argmax(id_data["Y"], axis=1)
                        if len(id_data["Y"].shape) > 1
                        else id_data["Y"]
                    )
                    id_class_mask = id_labels == class_id

                    ood_data = all_data[ood_dataset]
                    ood_labels = (
                        np.argmax(ood_data["Y"], axis=1)
                        if len(ood_data["Y"].shape) > 1
                        else ood_data["Y"]
                    )
                    ood_class_mask = ood_labels == class_id

                    if np.any(id_class_mask) and np.any(ood_class_mask):
                        id_class_scores = id_data["scores"][id_class_mask]
                        ood_class_scores = ood_data["scores"][ood_class_mask]

                        summary_stats[ood_dataset][class_id] = {
                            "id_mean": np.mean(id_class_scores),
                            "id_std": np.std(id_class_scores),
                            "id_count": len(id_class_scores),
                            "ood_mean": np.mean(ood_class_scores),
                            "ood_std": np.std(ood_class_scores),
                            "ood_count": len(ood_class_scores),
                            "score_diff": np.mean(id_class_scores)
                            - np.mean(ood_class_scores),
                        }

            except Exception as e:
                print(f"Error creating plot for class {class_id} vs {ood_dataset}: {e}")
                continue

    # Save summary statistics
    create_summary_report(summary_stats, save_dir, num_classes)

    print(f"All class comparison plots saved in {save_dir}/")


def create_summary_report(summary_stats, save_dir, num_classes):
    """
    Create a summary report of all class comparisons.
    """
    summary_path = os.path.join(save_dir, "summary_statistics.txt")

    with open(summary_path, "w") as f:
        f.write("=== CLASS-SPECIFIC SCORE COMPARISON SUMMARY ===\n\n")

        for ood_dataset in summary_stats:
            f.write(f"OOD Dataset: {ood_dataset}\n")
            f.write("-" * 50 + "\n")

            # Overall statistics across all classes
            all_score_diffs = [
                stats["score_diff"] for stats in summary_stats[ood_dataset].values()
            ]
            if all_score_diffs:
                f.write(
                    f"Overall score difference (ID - OOD): {np.mean(all_score_diffs):.4f} ± {np.std(all_score_diffs):.4f}\n"  # noqa: E501
                )

            # Per-class breakdown
            f.write("\nPer-class breakdown:\n")
            f.write("Class | ID Mean±Std (n) | OOD Mean±Std (n) | Difference\n")
            f.write("-" * 60 + "\n")

            for class_id in range(num_classes):
                if class_id in summary_stats[ood_dataset]:
                    stats = summary_stats[ood_dataset][class_id]
                    f.write(
                        f"{class_id:5d} | {stats['id_mean']:6.3f}±{stats['id_std']:5.3f} ({stats['id_count']:4d}) | "
                        f"{stats['ood_mean']:6.3f}±{stats['ood_std']:5.3f} ({stats['ood_count']:4d}) | "
                        f"{stats['score_diff']:8.3f}\n"
                    )
                else:
                    f.write(f"{class_id:5d} | No data available\n")

            f.write("\n" + "=" * 70 + "\n\n")

    print(f"Summary report saved to {summary_path}")


def plot_score_separation_heatmap(all_data, save_dir="results", num_classes=10):
    """
    Create a heatmap showing score separation (ID - OOD) for each class and OOD dataset.
    """
    ood_datasets = [k for k in all_data.keys() if k != "ID"]

    if not ood_datasets:
        return

    # Create separation matrix
    separation_matrix = np.zeros((len(ood_datasets), num_classes))

    for i, ood_dataset in enumerate(ood_datasets):
        for class_id in range(num_classes):
            # Get class-specific data
            id_data = all_data["ID"]
            id_labels = (
                np.argmax(id_data["Y"], axis=1)
                if len(id_data["Y"].shape) > 1
                else id_data["Y"]
            )
            id_class_mask = id_labels == class_id

            ood_data = all_data[ood_dataset]
            ood_labels = (
                np.argmax(ood_data["Y"], axis=1)
                if len(ood_data["Y"].shape) > 1
                else ood_data["Y"]
            )
            ood_class_mask = ood_labels == class_id

            if np.any(id_class_mask) and np.any(ood_class_mask):
                id_class_scores = id_data["scores"][id_class_mask]
                ood_class_scores = ood_data["scores"][ood_class_mask]
                separation_matrix[i, class_id] = np.mean(id_class_scores) - np.mean(
                    ood_class_scores
                )
            else:
                separation_matrix[i, class_id] = np.nan

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, max(6, len(ood_datasets))))

    im = ax.imshow(separation_matrix, cmap="RdBu_r", aspect="auto")

    # Set ticks and labels
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels([f"Class {i}" for i in range(num_classes)])
    ax.set_yticks(range(len(ood_datasets)))
    ax.set_yticklabels(ood_datasets)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Score Difference (ID - OOD)", rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(ood_datasets)):
        for j in range(num_classes):
            if not np.isnan(separation_matrix[i, j]):
                _ = ax.text(
                    j,
                    i,
                    f"{separation_matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

    plt.title("Score Separation Heatmap: ID vs OOD by Class", fontsize=14, pad=20)
    plt.xlabel("Class")
    plt.ylabel("OOD Dataset")
    plt.tight_layout()

    # Save heatmap
    heatmap_path = os.path.join(save_dir, "score_separation_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Score separation heatmap saved to {heatmap_path}")

    return fig


def plot_score_distributions(all_data, ood_classes=None, save_path=None):
    """
    Plot histograms of score distributions for ID vs OOD data.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    # ID scores
    if "ID" in all_data:
        id_scores = np.array(all_data["ID"]["scores"])
        ax.hist(id_scores, bins=50, alpha=0.5, color="blue", label="ID")
        ax.set_title("ID Score Distribution")
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        ax.axvline(
            np.mean(id_scores),
            color="blue",
            linestyle="--",
            label=f"Mean: {np.mean(id_scores):.4f}",
        )
        ax.legend()

    # OOD scores
    if not ood_classes:
        ood_data = {k: v for k, v in all_data.items() if k != "ID"}
    else:
        ood_data = {k: v for k, v in all_data.items() if k in ood_classes}
    if ood_data:
        ood_scores = np.concatenate([data["scores"] for data in ood_data.values()])
        ax.hist(ood_scores, bins=50, alpha=0.5, color="red", label="OOD")
        ax.axvline(
            np.mean(ood_scores),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(ood_scores):.4f}",
        )
        ax.legend()

    plt.text(3, 2.5, f"AUROC score = {auroc(id_scores, ood_scores)}")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Score distributions plot saved to {save_path}")

    plt.show()
    return fig


def plot_images_with_scores(
    X,
    Y,
    batch_scores,
    max_images=None,
    figsize_per_image=(2, 2.5),
    cols=8,
    save_path=None,
    title_prefix="",
    cmap="gray",
):
    """
    Plot 28x28 images with their labels and scores as captions.

    Args:
        X: Image data of shape (batch_size, 28, 28, 1)
        Y: One-hot encoded labels of shape (batch_size, 10)
        batch_scores: Scores of shape (batch_size,)
        max_images: Maximum number of images to plot (None for all)
        figsize_per_image: Size of each subplot
        cols: Number of columns in the plot grid
        save_path: Path to save the figure (optional)
        title_prefix: Prefix for the main title
        cmap: Colormap for images
    """

    # Convert to numpy for matplotlib
    X_np = np.array(X)
    Y_np = np.array(Y)
    scores_np = np.array(batch_scores)

    # Get class labels from one-hot encoding
    class_labels = np.argmax(Y_np, axis=1)

    # Determine number of images to plot
    n_images = len(X_np) if max_images is None else min(max_images, len(X_np))

    # Calculate grid dimensions
    rows = (n_images + cols - 1) // cols

    # Create figure
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * figsize_per_image[0], rows * figsize_per_image[1])
    )

    # Handle case where we have only one row
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # Plot images
    for i in range(n_images):
        row = i // cols
        col = i % cols
        ax = axes[row, col]

        # Squeeze the channel dimension for display (28, 28, 1) -> (28, 28)
        img = X_np[i].squeeze()

        # Display image
        ax.imshow(img, cmap=cmap)
        ax.axis("off")

        # Create caption with label and score
        label = class_labels[i]
        score = scores_np[i]
        caption = f"Label: {label}\nScore: {score:.4f}"

        ax.set_title(caption, fontsize=8, pad=5)

    # Hide empty subplots
    for i in range(n_images, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis("off")

    # Add main title
    if title_prefix:
        fig.suptitle(
            f"{title_prefix} - Images with Labels and Scores", fontsize=14, y=0.98
        )
    else:
        fig.suptitle("Images with Labels and Scores", fontsize=14, y=0.98)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()

    return fig


def plot_top_bottom_scores(
    X,
    Y,
    batch_scores,
    n_top=8,
    n_bottom=8,
    figsize_per_image=(2, 2.5),
    save_path=None,
    cmap="gray",
):
    """
    Plot the top and bottom scoring images separately.

    Args:
        X: Image data of shape (batch_size, 28, 28, 1)
        Y: One-hot encoded labels of shape (batch_size, 10)
        batch_scores: Scores of shape (batch_size,)
        n_top: Number of top scoring images to show
        n_bottom: Number of bottom scoring images to show
        figsize_per_image: Size of each subplot
        save_path: Path to save the figure (optional)
        cmap: Colormap for images
    """

    # Convert to numpy
    scores_np = np.array(batch_scores)

    # Get indices for top and bottom scores
    top_indices = np.argsort(scores_np)[-n_top:][::-1]  # Highest scores first
    bottom_indices = np.argsort(scores_np)[:n_bottom]  # Lowest scores first

    # Create subplots
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot top scores
    plt.sca(ax_top)
    plt.title("Highest Scoring Images", fontsize=12, pad=20)
    plot_images_with_scores(
        X[top_indices],
        Y[top_indices],
        batch_scores[top_indices],
        cols=n_top,
        figsize_per_image=figsize_per_image,
        cmap=cmap,
    )

    # Plot bottom scores
    plt.sca(ax_bottom)
    plt.title("Lowest Scoring Images", fontsize=12, pad=20)
    plot_images_with_scores(
        X[bottom_indices],
        Y[bottom_indices],
        batch_scores[bottom_indices],
        cols=n_bottom,
        figsize_per_image=figsize_per_image,
        cmap=cmap,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()

    return fig


if __name__ == "__main__":
    args = parser.parse_args()
    scores_dict = cloudpickle.load(open(args.scores_file_path, "rb"))
    args_dict = scores_dict["args_dict"]
    eigenvals = scores_dict["eigenvals"]
    score_fun = scores_dict["score_fun"]
    stop = 0

    # Define ID data
    _, _, ID_loader = dataloader_from_string(
        args.ID_dataset,
        n_samples=None,
        batch_size=args.test_batch_size,
        shuffle=False,
        seed=args.model_seed,
        download=False,
        data_path=args.data_path,
    )
    # Define OOD data
    OOD_loaders = [
        dataloader_from_string(
            OOD_dataset,
            n_samples=None,
            batch_size=args.test_batch_size,
            shuffle=False,
            seed=0,
            download=True,
            data_path=args.data_path,
        )[2]
        for OOD_dataset in args_dict["OOD_datasets"]
    ]
    for d, loader in zip(args_dict["OOD_datasets"], OOD_loaders):
        print(
            f"Got OUT-of-distribution dataset {d} with {len(loader.dataset)} test data"
        )

    done = 0
    all_collected_data = defaultdict(lambda: {"X": [], "Y": [], "scores": []})

    for distribution, loader in [
        ("ID", ID_loader),
        *zip(args_dict["OOD_datasets"], OOD_loaders),
    ]:
        scores_dict[distribution] = []

        for batch in loader:
            # if done > 200:
            #     break
            X = jnp.array(batch[0].numpy())
            Y = jnp.array(batch[1].numpy())

            # Apply score function to batch of datapoints
            batch_scores = score_fun(X)
            scores_dict[distribution].append(batch_scores)

            # Collect data for later plotting
            all_collected_data[distribution]["X"].append(X)
            all_collected_data[distribution]["Y"].append(Y)
            all_collected_data[distribution]["scores"].append(batch_scores)

            # Original per-batch plotting (optional - you might want to comment these out)
            # plot_images_with_scores(X, Y, batch_scores, max_images=16,
            #                        title_prefix=f"{distribution}",
            #                        save_path=f"images_with_scores_{distribution}_batch_{done//X.shape[0]}.png")

            done += X.shape[0]

    # Concatenate all collected data
    for distribution in all_collected_data:
        all_collected_data[distribution]["X"] = np.concatenate(
            all_collected_data[distribution]["X"], axis=0
        )
        all_collected_data[distribution]["Y"] = np.concatenate(
            all_collected_data[distribution]["Y"], axis=0
        )
        all_collected_data[distribution]["scores"] = np.concatenate(
            all_collected_data[distribution]["scores"], axis=0
        )

    # Create the comprehensive plots
    print("Creating score category plots...")
    plot_score_categories(
        all_collected_data, save_path="score_categories_comparison.png"
    )

    print("Creating score distribution plots...")
    plot_score_distributions(all_collected_data, ood_classes=["MNIST"], save_path="score_distributions_MNIST.png")
    plot_score_distributions(all_collected_data, save_path="score_distributions_all.png")

    # print("Creating class specific comparison...")
    # create_all_class_comparisons(all_collected_data, save_dir="results", num_classes=Y.shape[1])

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for distribution, data in all_collected_data.items():
        scores = data["scores"]
        print(f"{distribution}:")
        print(f"  Mean score: {np.mean(scores):.4f}")
        print(f"  Std score: {np.std(scores):.4f}")
        print(f"  Min score: {np.min(scores):.4f}")
        print(f"  Max score: {np.max(scores):.4f}")
        print(f"  Number of samples: {len(scores)}")
        print()
