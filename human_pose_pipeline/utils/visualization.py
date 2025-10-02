"""
Visualization utilities for human pose estimation

This module contains functions for visualizing poses, uncertainties, and creating animations.
Extracted and organized from the examples to provide reusable visualization functionality.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image
from typing import List, Optional, Tuple, Union

# Skeleton connections for 13-joint pose visualization
CONNECTIONS_13 = [
    (0, 1), (0, 2),  # Nose to shoulders
    (1, 3), (3, 5),  # Left arm
    (2, 4), (4, 6),  # Right arm
    (1, 2), (1, 7), (2, 8),  # Shoulders to hips
    (7, 8),  # Connect hips
    (7, 9), (9, 11),  # Left leg
    (8, 10), (10, 12)  # Right leg
]

def draw_uncertainty_ellipse(ax, center: Tuple[float, float], uncertainties: Tuple[float, float],
                           covariance: float = None, n_std: float = 2, **kwargs):
    """
    Draw uncertainty ellipse for a joint using matplotlib

    Args:
        ax: matplotlib axis
        center: (x, y) center position
        uncertainties: (std_x, std_y) standard deviations
        covariance: covariance between x and y (optional)
        n_std: number of standard deviations for ellipse size
        **kwargs: additional arguments passed to Ellipse
    """
    if covariance is None:
        covariance = 0.0

    # Create covariance matrix
    cov_matrix = np.array([[uncertainties[0]**2, covariance],
                          [covariance, uncertainties[1]**2]])

    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)

    # Compute ellipse parameters
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    width = 2 * n_std * np.sqrt(eigenvals[0])
    height = 2 * n_std * np.sqrt(eigenvals[1])

    # Create and add ellipse
    ellipse = Ellipse(center, width, height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

    return ellipse

def draw_uncertainty_ellipse_cv2(image: np.ndarray, center: Tuple[int, int],
                                uncertainties: Tuple[float, float], covariance: float = None,
                                n_std: float = 2, color: Tuple[int, int, int] = (0, 255, 255),
                                thickness: int = 1):
    """
    Draw uncertainty ellipse for a joint using OpenCV

    Args:
        image: OpenCV image array
        center: (x, y) center position in pixels
        uncertainties: (std_x, std_y) standard deviations in pixels
        covariance: covariance between x and y (optional)
        n_std: number of standard deviations for ellipse size
        color: BGR color tuple
        thickness: line thickness
    """
    if covariance is None:
        covariance = 0.0

    std_x, std_y = uncertainties

    if std_x > 0 and std_y > 0:
        # Calculate the angle of the ellipse
        angle = 0.5 * np.arctan2(2 * covariance, (std_x**2 - std_y**2)) * (180 / np.pi)

        # Calculate the width and height of the ellipse based on standard deviations
        width = int(n_std * std_x)
        height = int(n_std * std_y)

        # Draw the uncertainty ellipse
        if width > 0 and height > 0:
            cv2.ellipse(image, center, (width, height), angle, 0, 360, color, thickness)

def visualize_single_pose_on_image(image: Union[np.ndarray, Image.Image],
                                  gt_pose: Optional[np.ndarray] = None,
                                  pred_pose: Optional[np.ndarray] = None,
                                  pred_uncertainties: Optional[np.ndarray] = None,
                                  pred_covariances: Optional[np.ndarray] = None,
                                  show_uncertainty: bool = True,
                                  uncertainty_n_std: float = 2) -> np.ndarray:
    """
    Visualize ground truth and/or predicted pose on a single image

    Args:
        image: Input image (PIL Image or numpy array)
        gt_pose: Ground truth pose (N, 2), optional
        pred_pose: Predicted pose (N, 2), optional
        pred_uncertainties: Prediction uncertainties (N, 2), optional
        pred_covariances: Prediction covariances (N,), optional
        show_uncertainty: Whether to draw uncertainty ellipses
        uncertainty_n_std: Number of standard deviations for uncertainty ellipses

    Returns:
        np.ndarray: Image with poses overlaid
    """
    # Convert PIL to numpy if needed
    if hasattr(image, 'mode'):
        image_array = np.array(image).copy()
    else:
        image_array = image.copy()

    # Get image dimensions
    image_height, image_width = image_array.shape[:2]

    # Draw ground truth pose
    if gt_pose is not None:
        gt_pose_clipped = np.clip(gt_pose, 0, [image_width - 1, image_height - 1])

        # Draw GT connections
        for connection in CONNECTIONS_13:
            start_idx, end_idx = connection
            if start_idx < len(gt_pose_clipped) and end_idx < len(gt_pose_clipped):
                start_point = tuple(gt_pose_clipped[start_idx].astype(int))
                end_point = tuple(gt_pose_clipped[end_idx].astype(int))
                cv2.line(image_array, start_point, end_point, color=(255, 0, 0), thickness=2)  # Red for GT

        # Draw GT keypoints
        for idx, (x, y) in enumerate(gt_pose_clipped):
            cv2.circle(image_array, (int(x), int(y)), radius=4, color=(0, 255, 0), thickness=-1)  # Green for GT
            cv2.putText(image_array, f"GT{idx}", (int(x)+5, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.3, (0, 0, 255), 1, cv2.LINE_AA)

    # Draw predicted pose
    if pred_pose is not None:
        pred_pose_clipped = np.clip(pred_pose, 0, [image_width - 1, image_height - 1])

        # Draw prediction connections
        for connection in CONNECTIONS_13:
            start_idx, end_idx = connection
            if start_idx < len(pred_pose_clipped) and end_idx < len(pred_pose_clipped):
                start_point = tuple(pred_pose_clipped[start_idx].astype(int))
                end_point = tuple(pred_pose_clipped[end_idx].astype(int))
                cv2.line(image_array, start_point, end_point, color=(0, 0, 255), thickness=2)  # Blue for prediction

        # Draw predicted keypoints and uncertainty ellipses
        for idx, (x, y) in enumerate(pred_pose_clipped):
            cv2.circle(image_array, (int(x), int(y)), radius=4, color=(255, 0, 0), thickness=-1)  # Blue for prediction
            cv2.putText(image_array, f"P{idx}", (int(x)+5, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.3, (255, 0, 0), 1, cv2.LINE_AA)

            # Draw uncertainty ellipse if available
            if (show_uncertainty and pred_uncertainties is not None and
                idx < len(pred_uncertainties)):
                uncertainty = pred_uncertainties[idx]
                covariance = pred_covariances[idx] if pred_covariances is not None else None

                draw_uncertainty_ellipse_cv2(
                    image_array, (int(x), int(y)), uncertainty, covariance,
                    n_std=uncertainty_n_std, color=(0, 255, 255), thickness=1  # Cyan for uncertainty
                )

    return image_array

def visualize_poses_matplotlib(image: Union[np.ndarray, Image.Image],
                              gt_pose: Optional[np.ndarray] = None,
                              pred_pose: Optional[np.ndarray] = None,
                              pred_uncertainties: Optional[np.ndarray] = None,
                              pred_covariances: Optional[np.ndarray] = None,
                              save_path: Optional[str] = None,
                              show_uncertainty: bool = True,
                              uncertainty_n_std: float = 3) -> None:
    """
    Create matplotlib visualization with multiple panels showing poses and uncertainties

    Args:
        image: Input image
        gt_pose: Ground truth pose (N, 2), optional
        pred_pose: Predicted pose (N, 2), optional
        pred_uncertainties: Prediction uncertainties (N, 2), optional
        pred_covariances: Prediction covariances (N,), optional
        save_path: Path to save the figure, optional
        show_uncertainty: Whether to show uncertainty panel
        uncertainty_n_std: Number of standard deviations for uncertainty ellipses
    """
    # Convert PIL to numpy if needed
    if hasattr(image, 'mode'):
        image_np = np.array(image)
    else:
        image_np = image

    # Determine number of panels
    n_panels = 1  # Original image
    if gt_pose is not None:
        n_panels += 1
    if pred_pose is not None:
        n_panels += 1
    if pred_pose is not None and show_uncertainty and pred_uncertainties is not None:
        n_panels += 1

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    # Original image
    axes[panel_idx].imshow(image_np)
    axes[panel_idx].set_title('Original Image')
    axes[panel_idx].axis('off')
    panel_idx += 1

    # Ground truth pose
    if gt_pose is not None:
        axes[panel_idx].imshow(image_np)
        axes[panel_idx].set_title('Ground Truth Pose')
        axes[panel_idx].axis('off')

        # Draw GT skeleton
        for connection in CONNECTIONS_13:
            start_idx, end_idx = connection
            if start_idx < len(gt_pose) and end_idx < len(gt_pose):
                start_point = gt_pose[start_idx]
                end_point = gt_pose[end_idx]
                axes[panel_idx].plot([start_point[0], end_point[0]],
                                   [start_point[1], end_point[1]], 'g-', linewidth=2)

        # Draw GT keypoints
        for i, point in enumerate(gt_pose):
            axes[panel_idx].scatter(point[0], point[1], c='red', s=50, zorder=5)
            axes[panel_idx].text(point[0]+5, point[1]-5, str(i), fontsize=8, color='white',
                                bbox=dict(boxstyle="round,pad=0.2", facecolor='red', alpha=0.7))
        panel_idx += 1

    # Predicted pose
    if pred_pose is not None:
        axes[panel_idx].imshow(image_np)
        axes[panel_idx].set_title('Predicted Pose')
        axes[panel_idx].axis('off')

        # Draw predicted skeleton
        for connection in CONNECTIONS_13:
            start_idx, end_idx = connection
            if start_idx < len(pred_pose) and end_idx < len(pred_pose):
                start_point = pred_pose[start_idx]
                end_point = pred_pose[end_idx]
                axes[panel_idx].plot([start_point[0], end_point[0]],
                                   [start_point[1], end_point[1]], 'b-', linewidth=2)

        # Draw predicted keypoints
        for i, point in enumerate(pred_pose):
            axes[panel_idx].scatter(point[0], point[1], c='yellow', s=50, zorder=5)
            axes[panel_idx].text(point[0]+5, point[1]-5, str(i), fontsize=8, color='white',
                                bbox=dict(boxstyle="round,pad=0.2", facecolor='blue', alpha=0.7))
        panel_idx += 1

        # Predicted pose with uncertainty ellipses
        if show_uncertainty and pred_uncertainties is not None:
            axes[panel_idx].imshow(image_np)
            axes[panel_idx].set_title('Predicted Pose + Uncertainty')
            axes[panel_idx].axis('off')

            # Draw predicted skeleton
            for connection in CONNECTIONS_13:
                start_idx, end_idx = connection
                if start_idx < len(pred_pose) and end_idx < len(pred_pose):
                    start_point = pred_pose[start_idx]
                    end_point = pred_pose[end_idx]
                    axes[panel_idx].plot([start_point[0], end_point[0]],
                                       [start_point[1], end_point[1]], 'b-', linewidth=2)

            # Draw uncertainty ellipses and keypoints
            for i, (point, uncertainty) in enumerate(zip(pred_pose, pred_uncertainties)):
                covariance = pred_covariances[i] if pred_covariances is not None else None

                # Draw uncertainty ellipse
                draw_uncertainty_ellipse(
                    axes[panel_idx], point, uncertainty, covariance,
                    n_std=uncertainty_n_std, facecolor='cyan', alpha=0.3,
                    edgecolor='blue', linewidth=1
                )

                # Draw keypoint
                axes[panel_idx].scatter(point[0], point[1], c='yellow', s=50, zorder=5)
                axes[panel_idx].text(point[0]+5, point[1]-5, str(i), fontsize=8, color='white',
                                    bbox=dict(boxstyle="round,pad=0.2", facecolor='blue', alpha=0.7))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    plt.show()

def visualize_pose_sequence(pose_sequence: np.ndarray,
                          images: List[Image.Image],
                          output_file: str,
                          num_frames: Optional[int] = None,
                          estimated_poses: Optional[List[np.ndarray]] = None,
                          estimated_uncertainties: Optional[List[np.ndarray]] = None,
                          estimated_covariances: Optional[List[np.ndarray]] = None,
                          show_uncertainty: bool = True,
                          uncertainty_n_std: float = 2,
                          duration: int = 100) -> None:
    """
    Create an animated GIF visualization of pose sequences overlaid on image frames

    Args:
        pose_sequence: Ground truth pose sequence (num_frames, num_joints, 2)
        images: List of PIL images for each frame
        output_file: Path to save the output GIF
        num_frames: Number of frames to process (default: all)
        estimated_poses: List of estimated poses for each frame, optional
        estimated_uncertainties: List of uncertainties for each frame, optional
        estimated_covariances: List of covariances for each frame, optional
        show_uncertainty: Whether to draw uncertainty ellipses
        uncertainty_n_std: Number of standard deviations for uncertainty ellipses
        duration: Duration between frames in milliseconds
    """
    if num_frames is None:
        num_frames = pose_sequence.shape[0]

    # Initialize a list to store individual frames for the GIF
    frames_for_gif = []

    for frame in range(num_frames):
        # Get the image frame
        image_pil = images[frame]

        # Get ground truth pose
        gt_pose = pose_sequence[frame]

        # Get estimated pose data if available
        pred_pose = estimated_poses[frame] if estimated_poses and frame < len(estimated_poses) else None
        pred_uncertainties = estimated_uncertainties[frame] if estimated_uncertainties and frame < len(estimated_uncertainties) else None
        pred_covariances = estimated_covariances[frame] if estimated_covariances and frame < len(estimated_covariances) else None

        # Create visualization for this frame
        image_with_pose = visualize_single_pose_on_image(
            image_pil, gt_pose, pred_pose, pred_uncertainties, pred_covariances,
            show_uncertainty=show_uncertainty, uncertainty_n_std=uncertainty_n_std
        )

        # Convert back to PIL Image for GIF creation
        image_with_pose_pil = Image.fromarray(image_with_pose)
        frames_for_gif.append(image_with_pose_pil)

    # Create an animated GIF with the overlaid poses
    if frames_for_gif:
        frames_for_gif[0].save(
            output_file,
            save_all=True,
            append_images=frames_for_gif[1:],
            duration=duration,
            loop=0
        )
        print(f"Animated GIF saved to: {output_file}")
    else:
        print("No frames to save!")


def draw_3d_pose_with_covariance(ax, points_3d, covariances, connections, scale=1.0):
    """
    Draw the 3D pose with covariance ellipsoids on a matplotlib axis.
    """
    ax.clear()

    # Draw skeleton connections
    for connection in connections:
        start, end = connection
        ax.plot([points_3d[start, 0], points_3d[end, 0]],
                [points_3d[start, 1], points_3d[end, 1]],
                [points_3d[start, 2], points_3d[end, 2]], 'r-')

    # Draw keypoints
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', marker='o')

    # Draw covariance ellipsoids (simplified version)
    for i in range(points_3d.shape[0]):
        cov = covariances[i]
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-8)  # Ensure positive eigenvalues

        # Create simplified ellipsoid representation
        radii = scale * np.sqrt(eigenvalues)

        # Draw uncertainty as simple lines along principal axes
        for j, (eigval, eigvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            start = points_3d[i] - radii[j] * eigvec
            end = points_3d[i] + radii[j] * eigvec
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                   'r-', alpha=0.5, linewidth=1)

    # Set consistent axis limits
    ax.set_xlim(-2000, 0)
    ax.set_ylim(-2000, 0)
    ax.set_zlim(1000, 2000)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Pose with Uncertainty')
