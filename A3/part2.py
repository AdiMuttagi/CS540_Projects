"""
Part 2: Train a decision tree on the complete data set, restricted to the
same feature list allowed by the grader (x8, x2, x7, x4, x6, x10), then
prune it using a held-out validation set.

Data columns in breast-cancer-wisconsin.data:
  col 0        -> sample_code_number (id, not used as a feature)
  col 1..9     -> x2..x10 (Clump Thickness ... Mitoses)
  col 10       -> class (2 = benign, 4 = malignant)
"""

import math
import random

DATA_FILE = "data/breast+cancer+wisconsin+original/breast-cancer-wisconsin.data"
TEST_FILE = "data/test.txt"
FEATURES = [8, 2, 7, 4, 6, 10]
random.seed(0)


def load_data():
    rows = []
    with open(DATA_FILE) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 11:
                continue
            if "?" in parts:
                continue  # drop the 16 rows with a missing Bare Nuclei value
            sample_id = parts[0]
            x = {feat: int(parts[feat - 1]) for feat in range(2, 11)}
            label = int(parts[10])
            rows.append((sample_id, x, label))
    return rows


def load_test_ids():
    ids = []
    with open(TEST_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = line.split("\t")[-1]
            sample_id = row.split(",")[0].strip()
            ids.append(sample_id)
    return set(ids)


def entropy(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    n_benign = sum(1 for l in labels if l == 2)
    n_malignant = n - n_benign
    e = 0.0
    for c in (n_benign, n_malignant):
        if c > 0:
            p = c / n
            e -= p * math.log2(p)
    return e


def majority_label(rows):
    n_benign = sum(1 for _, _, l in rows if l == 2)
    n_malignant = len(rows) - n_benign
    return 2 if n_benign >= n_malignant else 4


class Node:
    def __init__(self, leaf, label=None, feature=None, threshold=None,
                 left=None, right=None):
        self.leaf = leaf
        self.label = label          # majority label at this node (used if pruned to leaf)
        self.feature = feature
        self.threshold = threshold
        self.left = left            # x <= threshold
        self.right = right          # x > threshold

    def count_nodes(self):
        if self.leaf:
            return 1
        return 1 + self.left.count_nodes() + self.right.count_nodes()

    def predict(self, x):
        if self.leaf:
            return self.label
        if x[self.feature] <= self.threshold:
            return self.left.predict(x)
        return self.right.predict(x)


def best_split(rows):
    labels = [l for _, _, l in rows]
    base_entropy = entropy(labels)
    best_gain = 0.0
    best_feature = None
    best_threshold = None

    for feature in FEATURES:
        values = sorted(set(x[feature] for _, x, _ in rows))
        for t in values:
            left_labels = [l for _, x, l in rows if x[feature] <= t]
            right_labels = [l for _, x, l in rows if x[feature] > t]
            if not left_labels or not right_labels:
                continue
            n = len(rows)
            weighted = (len(left_labels) / n) * entropy(left_labels) + \
                       (len(right_labels) / n) * entropy(right_labels)
            gain = base_entropy - weighted
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = t

    return best_feature, best_threshold, best_gain


def build_tree(rows):
    labels = [l for _, _, l in rows]
    label = majority_label(rows)

    if len(set(labels)) == 1:
        return Node(leaf=True, label=labels[0])

    feature, threshold, gain = best_split(rows)
    if feature is None or gain <= 0:
        return Node(leaf=True, label=label)

    left_rows = [r for r in rows if r[1][feature] <= threshold]
    right_rows = [r for r in rows if r[1][feature] > threshold]

    left_child = build_tree(left_rows)
    right_child = build_tree(right_rows)

    return Node(leaf=False, label=label, feature=feature, threshold=threshold,
                left=left_child, right=right_child)


def accuracy(node, rows):
    if not rows:
        return 1.0
    correct = sum(1 for _, x, l in rows if node.predict(x) == l)
    return correct / len(rows)


def prune_tree(node, val_rows):
    """Reduced-error pruning: bottom-up, replace a subtree with a leaf
    whenever doing so does not hurt accuracy on the validation set."""
    if node.leaf:
        return node

    left_val = [r for r in val_rows if r[1][node.feature] <= node.threshold]
    right_val = [r for r in val_rows if r[1][node.feature] > node.threshold]

    node.left = prune_tree(node.left, left_val)
    node.right = prune_tree(node.right, right_val)

    if not val_rows:
        return node

    subtree_acc = accuracy(node, val_rows)
    leaf_correct = sum(1 for _, x, l in val_rows if node.label == l)
    leaf_acc = leaf_correct / len(val_rows)

    if leaf_acc >= subtree_acc:
        return Node(leaf=True, label=node.label)
    return node


def tree_to_lines(node, indent=""):
    lines = []
    cond = "if (x{} <={})".format(node.feature, node.threshold)

    if node.left.leaf:
        lines.append(indent + cond + " return {}".format(node.left.label))
    else:
        lines.append(indent + cond)
        lines.extend(tree_to_lines(node.left, indent + "  "))

    if node.right.leaf:
        lines.append(indent + "else return {}".format(node.right.label))
    else:
        lines.append(indent + "else")
        lines.extend(tree_to_lines(node.right, indent + "  "))

    return lines


def max_depth(node, depth=0):
    if node.leaf:
        return depth
    return max(max_depth(node.left, depth + 1), max_depth(node.right, depth + 1))


def load_grading_test_rows():
    test_rows = []
    with open(TEST_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = line.split("\t")[-1]
            cols = [c.strip() for c in row.split(",")]
            x = {feat: int(cols[feat - 1]) for feat in range(2, 11)}
            label = int(cols[10])
            test_rows.append(("", x, label))
    return test_rows


def main():
    # "Complete data set": split ALL rows (including the ones that also
    # appear in the grading test.txt) into a train/validation partition.
    all_rows = load_data()
    random.shuffle(all_rows)
    split_point = int(len(all_rows) * 0.8)
    train_rows = all_rows[:split_point]
    val_rows = all_rows[split_point:]
    grading_test_rows = load_grading_test_rows()

    print("train size:", len(train_rows))
    print("validation size:", len(val_rows))
    print("grading test.txt size:", len(grading_test_rows))

    full_tree = build_tree(train_rows)
    print("nodes (unpruned):", full_tree.count_nodes())
    print("max depth (unpruned):", max_depth(full_tree))
    print("test.txt accuracy (unpruned):", accuracy(full_tree, grading_test_rows))

    tree_copy = build_tree(train_rows)  # fresh copy; prune_tree mutates in place
    pruned_tree = prune_tree(tree_copy, val_rows)
    print("nodes (pruned):", pruned_tree.count_nodes())
    print("max depth (pruned):", max_depth(pruned_tree))
    print("test.txt accuracy (pruned):", accuracy(pruned_tree, grading_test_rows))

    with open("answers/q4_tree.txt", "w") as f:
        f.write("\n".join(tree_to_lines(full_tree)) + "\n")

    with open("answers/q5.txt", "w") as f:
        f.write(str(max_depth(full_tree)) + "\n")

    with open("answers/q6_predictions.txt", "w") as f:
        preds = [full_tree.predict(x) for _, x, _ in grading_test_rows]
        f.write(", ".join(str(p) for p in preds) + "\n")

    with open("answers/q7_pruned_tree.txt", "w") as f:
        f.write("\n".join(tree_to_lines(pruned_tree)) + "\n")

    with open("answers/q8_predictions.txt", "w") as f:
        preds = [pruned_tree.predict(x) for _, x, _ in grading_test_rows]
        f.write(", ".join(str(p) for p in preds) + "\n")


if __name__ == "__main__":
    main()
