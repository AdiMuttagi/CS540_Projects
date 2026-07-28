"""
Question 9 (Competition): starting from a random image, run projected
gradient descent (PGD) against our team's network to find a typical
image of each digit 0-7.

Produces the two files the submission form asks for:
  answers/q9_rounded.txt  - 8 lines x 1024 ints, each 0 or 1
  answers/q9_original.txt - 8 lines x 1024 ints, each 0-255
"""

import numpy as np

from conv_net import load_net, prob_and_gradient, NUM_DIGITS, IMAGE_SIZE

TEAM = 0
NET_FILE = f"data/net{TEAM}.txt"
EPSILON = 0.02
ITERATIONS = 300
SEED = 0


def decode_digit(net, digit, rng):
    image = rng.uniform(0.0, 1.0, size=(IMAGE_SIZE, IMAGE_SIZE))
    for _ in range(ITERATIONS):
        _, grad = prob_and_gradient(image, net, digit)
        image = np.clip(image + EPSILON * np.sign(grad), 0.0, 1.0)
    return image


def write_lines(rows, out_file):
    lines = [", ".join(str(int(v)) for v in row) for row in rows]
    with open(out_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} lines to {out_file}")


def main():
    net = load_net(NET_FILE)
    rng = np.random.default_rng(SEED)

    rounded_rows = []
    original_rows = []
    for digit in range(NUM_DIGITS):
        image = decode_digit(net, digit, rng)
        rounded_rows.append(np.round(image).astype(int).flatten())
        original_rows.append(np.round(image * 255).astype(int).flatten())

    write_lines(rounded_rows, "answers/q9_rounded.txt")
    write_lines(original_rows, "answers/q9_original.txt")


if __name__ == "__main__":
    main()
