"""
Shared helpers for A5: loading sentences/embeddings and computing
cosine similarity (all embedding vectors here are already unit-norm,
so cosine similarity is just the dot product).
"""

import numpy as np

EMBED_DIM = 512


def load_sentences(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_vector(path):
    """A file containing a single comma-separated embedding vector."""
    with open(path) as f:
        values = f.read().strip().split(",")
    return np.array([float(v) for v in values])


def load_matrix(path):
    """A file containing one comma-separated embedding vector per line."""
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    return np.array([[float(v) for v in line.split(",")] for line in lines])


def cosine_similarity(vec, matrix):
    """`vec` is a single (512,) vector, `matrix` is (n, 512). Since all
    vectors are already unit-norm, this is just the dot product."""
    return matrix @ vec


def cosine_similarity_pair(a, b):
    """General cosine similarity between two vectors that are not
    necessarily unit-norm (e.g. an average of several unit vectors)."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
