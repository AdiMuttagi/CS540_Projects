"""
Shared helpers for the alien-digit convolutional network:
loading the net weight files and the test images, plus the
conv / pool / dense / softmax building blocks.

Image size is fixed at 32x32 (deduced from test.txt: 1024 numbers
per line, and from the dense layer being 48 x 12 = (3 maps * 4 * 4) x 12,
which means each of the 3 pooled maps is 4x4, and 4 * pool_size(8) = 32).
"""

import numpy as np

IMAGE_SIZE = 32
NUM_FILTERS = 3
FILTER_SIZE = 5
NUM_DIGITS = 8


def load_images(path):
    """Pixel values in the file are 0-255; the network expects them
    normalized to 0-1."""
    images = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pixels = [float(x) / 255.0 for x in line.split(",")]
            images.append(np.array(pixels).reshape(IMAGE_SIZE, IMAGE_SIZE))
    return images


def load_net(path):
    """Parses a net file into its layers. See A4 instructions for the
    exact line-by-line layout."""
    with open(path) as f:
        lines = [line.strip() for line in f]

    def read_matrix(start, rows, cols):
        block = lines[start:start + rows]
        mat = np.array([[float(x) for x in row.split(",")] for row in block])
        assert mat.shape == (rows, cols)
        return mat

    idx = 0
    filters = []
    biases = []
    for _ in range(NUM_FILTERS):
        filters.append(read_matrix(idx, FILTER_SIZE, FILTER_SIZE))
        idx += FILTER_SIZE
        biases.append(float(lines[idx]))
        idx += 1

    idx += 1  # skip "-----"
    pool_size = int(lines[idx])
    idx += 1
    idx += 1  # skip "-----"

    # Figure out dense layer shape from the pooled map size.
    pooled_side = IMAGE_SIZE // pool_size
    dense_in = NUM_FILTERS * pooled_side * pooled_side

    # Peek at the first dense-weight line to get the hidden layer size.
    hidden_units = len(lines[idx].split(","))
    dense_W = read_matrix(idx, dense_in, hidden_units)
    idx += dense_in
    dense_b = np.array([float(x) for x in lines[idx].split(",")])
    idx += 1

    idx += 1  # skip "-----"
    softmax_W = read_matrix(idx, hidden_units, NUM_DIGITS)
    idx += hidden_units
    softmax_b = np.array([float(x) for x in lines[idx].split(",")])

    return {
        "filters": filters,
        "biases": biases,
        "pool_size": pool_size,
        "dense_W": dense_W,
        "dense_b": dense_b,
        "softmax_W": softmax_W,
        "softmax_b": softmax_b,
    }


def relu(x):
    return np.maximum(0.0, x)


def conv2d_same(image, kernel):
    """Zero-padded, stride-1 true convolution (kernel flipped both
    axes, as opposed to cross-correlation). Returns an array the same
    shape as `image`."""
    kernel = kernel[::-1, ::-1]
    k = kernel.shape[0]
    pad = k // 2
    padded = np.pad(image, pad, mode="constant")
    h, w = image.shape
    out = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            region = padded[i:i + k, j:j + k]
            out[i, j] = np.sum(region * kernel)
    return out


def conv2d_same_backward(grad_output, kernel):
    """Adjoint of conv2d_same: given dL/d(conv output), returns
    dL/d(image). Mirrors the padding/indexing of conv2d_same exactly,
    just distributing gradient instead of gathering values."""
    kernel = kernel[::-1, ::-1]
    k = kernel.shape[0]
    pad = k // 2
    h, w = grad_output.shape
    padded_grad = np.zeros((h + 2 * pad, w + 2 * pad))
    for a in range(k):
        for b in range(k):
            padded_grad[a:a + h, b:b + w] += grad_output * kernel[a, b]
    return padded_grad[pad:pad + h, pad:pad + w]


def avg_pool(feature_map, pool_size):
    h, w = feature_map.shape
    out_h, out_w = h // pool_size, w // pool_size
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            block = feature_map[i * pool_size:(i + 1) * pool_size,
                                 j * pool_size:(j + 1) * pool_size]
            out[i, j] = np.mean(block)
    return out


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def forward(image, net):
    """Runs one image through the full network, returning the class
    probabilities and the intermediate maps (useful for debugging /
    later parts of the assignment)."""
    conv_maps = []
    for kernel, bias in zip(net["filters"], net["biases"]):
        conv_maps.append(relu(conv2d_same(image, kernel) + bias))

    pooled_maps = [avg_pool(m, net["pool_size"]) for m in conv_maps]

    # Flatten in filter-major order to match the 48 = 3 * 16 weight layout.
    flat = np.concatenate([m.flatten() for m in pooled_maps])

    hidden = relu(flat @ net["dense_W"] + net["dense_b"])
    logits = hidden @ net["softmax_W"] + net["softmax_b"]
    probs = softmax(logits)

    return probs, {"conv_maps": conv_maps, "pooled_maps": pooled_maps, "hidden": hidden}


def prob_and_gradient(image, net, digit):
    """Runs `image` through the network and backprops the probability
    of `digit` all the way to the input pixels, returning
    (prob_of_digit, d prob_of_digit / d image)."""
    pool_size = net["pool_size"]

    conv_pre, act, pooled = [], [], []
    for kernel, bias in zip(net["filters"], net["biases"]):
        cp = conv2d_same(image, kernel) + bias
        conv_pre.append(cp)
        a = relu(cp)
        act.append(a)
        pooled.append(avg_pool(a, pool_size))

    flat = np.concatenate([m.flatten() for m in pooled])
    hidden_pre = flat @ net["dense_W"] + net["dense_b"]
    hidden = relu(hidden_pre)
    logits = hidden @ net["softmax_W"] + net["softmax_b"]
    probs = softmax(logits)

    # d prob[digit] / d logits, via the softmax Jacobian.
    one_hot = np.zeros_like(probs)
    one_hot[digit] = 1.0
    d_logits = probs[digit] * (one_hot - probs)

    d_hidden = net["softmax_W"] @ d_logits
    d_hidden_pre = d_hidden * (hidden_pre > 0)
    d_flat = net["dense_W"] @ d_hidden_pre

    pooled_side = pooled[0].shape[0]
    chunk_size = pooled_side * pooled_side
    grad_image = np.zeros_like(image)
    for i, kernel in enumerate(net["filters"]):
        d_pooled = d_flat[i * chunk_size:(i + 1) * chunk_size].reshape(pooled_side, pooled_side)
        # Average pooling spreads its incoming gradient evenly over the block.
        d_act = np.repeat(np.repeat(d_pooled, pool_size, axis=0), pool_size, axis=1)
        d_act = d_act / (pool_size ** 2)
        d_conv_pre = d_act * (conv_pre[i] > 0)
        grad_image += conv2d_same_backward(d_conv_pre, kernel)

    return probs[digit], grad_image
