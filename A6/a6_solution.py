"""Generate upload files for CS 540 A6."""

from pathlib import Path

import numpy as np


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "vector_original25.txt"
ANSWER_DIR = ROOT / "answers"
NUM_AXES = 9
COMPETITION_EIGENFACES = 5
COMPETITION_SUSPECTS = np.array([0, 4, 13, 18, 24, 37, 46, 47, 53, 54])


def main() -> None:
    images = np.loadtxt(DATA_PATH, delimiter=",") / 255.0
    if images.shape != (25, 1024):
        raise ValueError(f"Expected data shape (25, 1024), got {images.shape}")

    mean_image = images.mean(axis=0)
    centered = images - mean_image

    # The PCA axes are the right singular vectors of the centered image matrix,
    # ordered by descending singular value.
    _, _, right_singular_vectors = np.linalg.svd(centered, full_matrices=False)
    axes = right_singular_vectors[:NUM_AXES]
    projections = centered @ axes.T

    # Question 3 depends on the four-decimal values submitted for Question 2.
    submitted_projections = np.round(projections, 4)
    reconstructions = submitted_projections @ axes + mean_image

    # Questions 5 and 6 explicitly depend on the four-decimal PCA directions
    # submitted for Question 4 and the four-decimal features submitted for Q5.
    submitted_principal_directions = np.round(axes, 4)
    pca_features = np.round(centered @ submitted_principal_directions.T, 4)
    pca_reconstructions = (
        pca_features @ submitted_principal_directions + mean_image
    )

    # Find each training image's nearest *other* image in PCA feature space.
    pairwise_distances = np.linalg.norm(
        pca_features[:, None, :] - pca_features[None, :, :], axis=2
    )
    np.fill_diagonal(pairwise_distances, np.inf)
    nearest_indices = np.argmin(pairwise_distances, axis=1)
    nearest_features = pca_features[nearest_indices]
    nearest_reconstructions = (
        nearest_features @ submitted_principal_directions + mean_image
    )

    ANSWER_DIR.mkdir(exist_ok=True)
    np.savetxt(ANSWER_DIR / "question1_axes.txt", axes, delimiter=",", fmt="%.17g")
    np.savetxt(
        ANSWER_DIR / "question2_projected_lengths.txt",
        submitted_projections,
        delimiter=",",
        fmt="%.4f",
    )
    np.savetxt(
        ANSWER_DIR / "question3_reconstructed_images.txt",
        reconstructions,
        delimiter=",",
        fmt="%.4f",
    )
    np.savetxt(
        ANSWER_DIR / "question4_principal_directions.txt",
        submitted_principal_directions,
        delimiter=",",
        fmt="%.4f",
    )
    np.savetxt(
        ANSWER_DIR / "question5_pca_features.txt",
        pca_features,
        delimiter=",",
        fmt="%.4f",
    )
    np.savetxt(
        ANSWER_DIR / "question6_pca_reconstructed_images.txt",
        pca_reconstructions,
        delimiter=",",
        fmt="%.4f",
    )
    np.savetxt(
        ANSWER_DIR / "question7_nearest_neighbor_features.txt",
        nearest_features,
        delimiter=",",
        fmt="%.4f",
    )
    np.savetxt(
        ANSWER_DIR / "question8_nearest_neighbor_reconstructions.txt",
        nearest_reconstructions,
        delimiter=",",
        fmt="%.4f",
    )

    competition_images = np.loadtxt(ROOT / "data" / "vector.txt", delimiter=",") / 255.0
    if competition_images.shape != (100, 1024):
        raise ValueError(
            "Expected Group 5 competition data shape (100, 1024), "
            f"got {competition_images.shape}"
        )
    competition_centered = competition_images - competition_images.mean(axis=0)
    _, _, competition_vectors = np.linalg.svd(
        competition_centered, full_matrices=False
    )
    competition_axes = competition_vectors[:COMPETITION_EIGENFACES]
    np.savetxt(
        ANSWER_DIR / "question9_competition_eigenfaces.txt",
        competition_axes,
        delimiter=",",
        fmt="%.17g",
    )
    np.savetxt(
        ANSWER_DIR / "question9_competition_suspects.txt",
        COMPETITION_SUSPECTS[None, :],
        delimiter=",",
        fmt="%d",
    )

    print(f"Question 1 shape: {axes.shape}")
    print(f"Question 2 shape: {projections.shape}")
    print(f"Question 3 shape: {reconstructions.shape}")
    print(f"Question 4 shape: {axes.shape}")
    print(f"Question 5 shape: {pca_features.shape}")
    print(f"Question 6 shape: {pca_reconstructions.shape}")
    print(f"Question 7 shape: {nearest_features.shape}")
    print(f"Question 8 shape: {nearest_reconstructions.shape}")
    print(f"Nearest-neighbor indices (zero-based): {nearest_indices.tolist()}")
    print(f"Competition eigenfaces shape: {competition_axes.shape}")
    print(f"Competition suspects: {COMPETITION_SUSPECTS.tolist()}")


if __name__ == "__main__":
    main()
