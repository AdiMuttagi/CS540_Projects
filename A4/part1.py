"""
Part 1 questions, all run against the shared test network (data/net.txt)
and the test images (data/test.txt).

Q1: convolution with the first filter only, no bias, no activation.
Q2: average-pooled first activation map, no bias, no activation.
Q3: average-pooled activation maps for all 3 filters, with the
    correct bias and ReLU activation.
Q4: softmax output layer values.
Q5: predicted digit (argmax of the softmax output) per image.
"""

import numpy as np

from conv_net import load_images, load_net, conv2d_same, avg_pool, relu, forward

IMAGES_FILE = "data/test.txt"
NET_FILE = "data/net.txt"


def write_lines(rows, out_file):
    lines = [", ".join(f"{v:.4f}" for v in row) for row in rows]
    with open(out_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} lines to {out_file}")


def question1(images, net):
    first_filter = net["filters"][0]
    rows = [conv2d_same(image, first_filter).flatten() for image in images]
    write_lines(rows, "answers/q1.txt")


def question2(images, net):
    first_filter = net["filters"][0]
    pool_size = net["pool_size"]
    rows = []
    for image in images:
        conv_out = conv2d_same(image, first_filter)
        pooled = avg_pool(conv_out, pool_size)
        rows.append(pooled.flatten())
    write_lines(rows, "answers/q2.txt")


def question3(images, net):
    """The 12 fully-connected hidden-layer activations, i.e. the next
    stage of the pipeline after pooling and flattening."""
    pool_size = net["pool_size"]
    rows = []
    for image in images:
        pooled_maps = []
        for kernel, bias in zip(net["filters"], net["biases"]):
            activation = relu(conv2d_same(image, kernel) + bias)
            pooled_maps.append(avg_pool(activation, pool_size))
        flat = [v for m in pooled_maps for v in m.flatten()]
        hidden = relu(np.array(flat) @ net["dense_W"] + net["dense_b"])
        rows.append(hidden)
    write_lines(rows, "answers/q3.txt")


def questions4_and_5(images, net):
    probs = [forward(image, net)[0] for image in images]
    write_lines(probs, "answers/q4.txt")

    digits = [int(np.argmax(p)) for p in probs]
    with open("answers/q5.txt", "w") as f:
        f.write(", ".join(str(d) for d in digits) + "\n")
    print(f"wrote {len(digits)} digits to answers/q5.txt")


def main():
    images = load_images(IMAGES_FILE)
    net = load_net(NET_FILE)

    question1(images, net)
    question2(images, net)
    question3(images, net)
    questions4_and_5(images, net)


if __name__ == "__main__":
    main()
