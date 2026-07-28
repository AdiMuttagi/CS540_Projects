"""
Part 2:
Q6-7: gradient of P(digit=0) with respect to every pixel of each test
      image, and one FGSM-style step in that direction.
Q8:   softmax output of the network on the Q7 (perturbed) images.
"""

import numpy as np

from conv_net import load_images, load_net, prob_and_gradient, forward, IMAGE_SIZE

IMAGES_FILE = "data/test.txt"
NET_FILE = "data/net.txt"
TARGET_DIGIT = 0
EPSILON = 0.1


def write_lines(rows, out_file, decimals):
    lines = [", ".join(f"{v:.{decimals}f}" for v in row) for row in rows]
    with open(out_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} lines to {out_file}")


def main():
    images = load_images(IMAGES_FILE)
    net = load_net(NET_FILE)

    grad_rows = []
    updated_rows = []
    for image in images:
        _, grad = prob_and_gradient(image, net, TARGET_DIGIT)
        grad_rows.append(grad.flatten())

        # Use the sign of the *rounded* (2-decimal) gradient, i.e. the
        # same value reported in Q6 -- gradients that round to 0.00
        # leave the pixel unchanged rather than nudging it by a
        # direction too small to be meaningful at that precision.
        rounded_grad = np.round(grad, 2)
        updated = image + EPSILON * np.sign(rounded_grad)
        updated = np.clip(updated, 0.0, 1.0)
        updated_rows.append(updated.flatten())

    write_lines(grad_rows, "answers/q6.txt", decimals=2)
    write_lines(updated_rows, "answers/q7.txt", decimals=4)

    prob_rows = [forward(row.reshape(IMAGE_SIZE, IMAGE_SIZE), net)[0] for row in updated_rows]
    write_lines(prob_rows, "answers/q8.txt", decimals=4)


if __name__ == "__main__":
    main()
