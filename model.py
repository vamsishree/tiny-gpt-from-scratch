"""
Tiny GPT From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_vocab
def build_vocab(text: str):
    """Return a sorted list of unique characters in text."""
    # TODO: return a sorted list of every unique character in text
    return sorted(set(text))

# Step 2 - build_stoi
def build_stoi(vocab):
    """Return a dict mapping each character in vocab to its index."""
    # TODO: map each character in vocab to its integer position
    final_dict = {}
    for index, char in enumerate(vocab):
        final_dict[char] = index
    return final_dict

# Step 3 - build_itos
def build_itos(vocab):
    """Return a dict mapping each index 0..len(vocab)-1 to its character."""
    # TODO: build an int-to-string lookup from the vocab list
    final_dict = {}
    for index, char in enumerate(vocab):
        final_dict[index] = char
    
    return final_dict

# Step 4 - encode_char
def encode_char(ch, stoi):
    """Return the integer token id for a single character ch using stoi."""
    # TODO: look up ch in the stoi mapping and return its id
    return stoi[ch]

# Step 5 - encode_string
def encode_string(text, stoi):
    """Encode a full string into a list of token ids using stoi."""
    # TODO: map each char in text through stoi (via encode_char) into a list of ids
    res = []
    for i in text:
        res.append(stoi[i])
    return res

# Step 6 - decode_int
def decode_int(token_id, itos):
    """Return the single character mapped to token_id by itos."""
    # TODO: look up the character for token_id in the itos dict
    return itos[token_id]

# Step 7 - decode_ids
def decode_ids(ids, itos):
    """Decode a list of token ids into a string using itos."""
    # TODO: map each id through decode_int and join the characters into one string.
    final_str = ""
    for i in ids:
        final_str += itos[i]
    return final_str

# Step 8 - make_1d_array
import numpy as np

def make_1d_array(values):
    """Create a 1D NumPy array from a Python list of numbers."""
    # TODO: convert the input list into a 1D numpy ndarray
    return np.array(values)

# Step 9 - get_array_shape
import numpy as np

def get_array_shape(arr):
    """Return the shape tuple of a NumPy array."""
    # TODO: return the shape of arr
    return arr.shape

# Step 10 - get_array_dtype
import numpy as np

def get_array_dtype(arr):
    """Return the dtype of a NumPy array."""
    # TODO: return the dtype attribute of arr
    return arr.dtype

# Step 11 - make_2d_zeros
import numpy as np

def make_2d_zeros(rows, cols):
    """Return a 2D NumPy array of zeros with shape (rows, cols)."""
    return np.zeros((rows, cols))

# Step 12 - make_2d_random
import numpy as np

def make_2d_random(rows, cols, seed):
    """Return a (rows, cols) array of uniform floats in [0, 1) seeded by `seed`."""
    rng = np.random.default_rng(seed)
    return rng.random((rows, cols))

# Step 13 - index_element
def index_element(arr, i, j):
    """Return the scalar element at position (i, j) of a 2D array."""
    # TODO: return the value at row i, column j of arr
    return arr[i][j]

# Step 14 - slice_row
import numpy as np

def slice_row(arr, i):
    """Return row i of a 2D array as a 1D view."""
    return arr[i]

# Step 15 - slice_column
import numpy as np

def slice_column(arr, j):
    """Return column j of a 2D array as a 1D array of length R."""
    return arr[:, j]

# Step 16 - slice_subblock
import numpy as np

def slice_subblock(arr, r0, r1, c0, c1):
    """Return the sub-block arr[r0:r1, c0:c1] of a 2D array."""
    # TODO: return the rectangular sub-block of arr bounded by rows [r0,r1) and cols [c0,c1).
    return arr[r0:r1, c0:c1]

# Step 17 - elementwise_add
import numpy as np

def elementwise_add(a, b):
    """Return the elementwise sum of two same-shape arrays."""
    # TODO: return a new array whose entries are the pairwise sums of a and b
    return a+b

# Step 18 - elementwise_multiply
import numpy as np

def elementwise_multiply(a, b):
    """Return the elementwise product of two same-shape arrays."""
    # TODO: compute the elementwise (Hadamard) product of a and b
    return a*b

# Step 19 - scalar_broadcast_add
import numpy as np

def scalar_broadcast_add(arr, scalar):
    """Return a new array equal to arr with scalar added to every element."""
    # TODO: add a Python scalar to every element of an array via broadcasting
    return arr + scalar

# Step 20 - vector_matrix_broadcast_add
import numpy as np

def vector_matrix_broadcast_add(matrix, vector):
    """Add a 1D vector to each row of a 2D matrix via broadcasting."""
    # TODO: return matrix + vector broadcast across rows
    return matrix + vector

# Step 21 - array_exp
import numpy as np

def array_exp(arr):
    """Return the elementwise exponential of arr."""
    # TODO: apply elementwise exponential to arr and return the result
    return np.exp(arr)

# Step 22 - array_log
import numpy as np

def array_log(arr):
    """Return the elementwise natural log of arr (assumes arr > 0)."""
    return np.log(arr)

# Step 23 - sum_all
import numpy as np

def sum_all(arr):
    """Return the sum of every element of arr as a scalar."""
    return np.sum(arr)

# Step 24 - sum_axis0
import numpy as np

def sum_axis0(arr):
    """Sum a 2D array along axis 0, collapsing rows into a 1D vector of column sums."""
    return np.sum(arr, 0)

# Step 25 - sum_axis1
import numpy as np

def sum_axis1(arr):
    """Sum a 2D array along axis 1, returning a 1D array of row sums."""
    # TODO: collapse the column dimension by summing each row
    return np.sum(arr, 1)

# Step 26 - max_along_axis
import numpy as np

def max_along_axis(arr, axis):
    """Return the maximum of arr along the given axis, with that axis removed."""
    return np.max(arr, axis)

# Step 27 - matmul
import numpy as np

def matmul(a, b):
    """Return the matrix product a @ b for 2D arrays a (M,K) and b (K,N)."""
    return np.matmul(a,b)

# Step 28 - transpose_matrix
def transpose_matrix(arr):
    """Return the transpose of a 2D array."""
    # TODO: return the transpose of arr using the .T attribute
    return arr.T

# Step 29 - sum_keepdims
import numpy as np

def sum_keepdims(arr, axis):
    """Sum along `axis` while keeping that dimension as size 1."""
    # TODO: sum along the given axis preserving the reduced dim as size 1
    return np.sum(arr, axis, keepdims = True)

# Step 30 - naive_softmax_1d
import numpy as np

def naive_softmax_1d(logits):
    """Compute softmax of a 1D logits vector via the direct exp/sum formula."""
    exp_logits = np.exp(logits)
    return exp_logits/np.sum(exp_logits)

# Step 31 - softmax_overflow_demo
def softmax_overflow_demo(large_value):
    """Show that naive exp overflows on a large logit.

    Return {'naive_exp': float, 'overflowed': bool}.
    """
    arr = np.array([large_value])

    exp_arr = np.exp(arr)

    result = float(exp_arr[0])    

    return {
        "naive_exp": result,
        "overflowed": np.isinf(result)
    }

# Step 32 - stable_softmax_1d
import numpy as np

def stable_softmax_1d(logits):
    """Numerically stable softmax over a 1D logits vector."""
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)

# Step 33 - stable_softmax_2d_rowwise
import numpy as np

def stable_softmax_2d_rowwise(logits):
    """Row-wise numerically stable softmax of a 2D logits array."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

# Step 34 - read_text_file
def read_text_file(text_blob):
    """Return text_blob unchanged after validating it is a non-empty string."""
    if text_blob == "":
        raise ValueError()
    elif type(text_blob) is not str:
        raise TypeError()
    else:
        return text_blob

# Step 35 - encode_corpus_to_int_array
def encode_corpus_to_int_array(text, stoi):
    """Convert the corpus string into a 1D NumPy int64 array of token ids."""
    ids = encode_string(text, stoi)
    return np.array(ids, dtype=np.int64)

# Step 36 - pick_split_point
def pick_split_point(n, train_frac):
    """Return integer split index so data[:idx] is train and data[idx:] is val."""
    return int(n * train_frac)

# Step 37 - slice_train_and_val
def slice_train_and_val(data, split_idx):
    """Split a 1D token-id array into (train, val) at split_idx."""
    return data[:split_idx], data[split_idx:]

# Step 38 - pick_block_size
def pick_block_size(default_size):
    """Return the context length (block_size) for training windows."""
    return max(1, int(default_size))

# Step 39 - slice_x_at_offset
import numpy as np

def slice_x_at_offset(data, i, block_size):
    """Return the input window data[i : i + block_size]."""
    return data[i : i + block_size]

# Step 40 - slice_y_at_offset
import numpy as np

def slice_y_at_offset(data, i, block_size):
    """Return the target window of length block_size starting at i+1."""
    return data[i + 1 : i + 1 + block_size]

# Step 41 - sample_random_batch_offsets
def sample_random_batch_offsets(data_len, block_size, batch_size, rng):
    """Sample batch_size random valid starting offsets for (block_size+1)-windows."""
    max_start = data_len - block_size
    return rng.integers(0, max_start, size=batch_size)

# Step 42 - stack_x_batch
import numpy as np

def stack_x_batch(data, offsets, block_size):
    """Stack per-offset X windows into a 2D batch matrix of shape (B, block_size)."""
    return np.stack(
        [slice_x_at_offset(data, offset, block_size) for offset in offsets],
        axis=0
    )

# Step 43 - stack_y_batch
import numpy as np

def stack_y_batch(data, offsets, block_size):
    """Stack per-offset Y windows into a 2D (B, block_size) target matrix."""
    return np.stack(
        [slice_y_at_offset(data, offset, block_size) for offset in offsets],
        axis=0
    )

# Step 44 - get_batch
def get_batch(data, block_size, batch_size, rng):
    offsets = sample_random_batch_offsets(
        len(data), block_size, batch_size, rng
    )
    X = stack_x_batch(data, offsets, block_size)
    Y = stack_y_batch(data, offsets, block_size)
    return X, Y

# Step 45 - allocate_count_matrix
import numpy as np

def allocate_count_matrix(vocab_size):
    """Allocate a (V, V) integer zero matrix for bigram counts."""
    return np.zeros((vocab_size, vocab_size), dtype=int)

# Step 46 - loop_fill_counts
import numpy as np

def loop_fill_counts(n_matrix, data):
    """Increment n_matrix[curr, next] for every consecutive pair in data."""
    for t in range(len(data) - 1):
        curr = data[t]
        nxt = data[t + 1]
        n_matrix[curr, nxt] += 1
    return n_matrix

# Step 47 - vectorize_counts_add_at
import numpy as np

def vectorize_counts_add_at(vocab_size, data):
    """Build (V, V) bigram counts from a 1D id array using vectorized scatter-add."""
    counts = allocate_count_matrix(vocab_size)
    np.add.at(counts, (data[:-1], data[1:]), 1)
    return counts

# Step 48 - add_one_smoothing
import numpy as np

def add_one_smoothing(n_matrix):
    """Return n_matrix with every entry incremented by 1 (Laplace smoothing)."""
    return n_matrix + 1

# Step 49 - row_sums_of_counts
def row_sums_of_counts(n_matrix):
    """Return per-row sums of n_matrix with shape (V, 1)."""
    return sum_keepdims(n_matrix, axis=1)

# Step 50 - normalize_counts_to_probs
def normalize_counts_to_probs(n_matrix):
    """Normalize a (V, V) count matrix into a row-stochastic probability matrix."""
    row_sums = row_sums_of_counts(n_matrix)
    return n_matrix / row_sums

# Step 51 - sample_next_token
def sample_next_token(p_matrix, current_id, rng):
    """Sample the next token id from P[current_id] using rng."""
    probs = p_matrix[current_id]
    return rng.choice(len(probs), p=probs)

# Step 52 - generate_sequence
def generate_sequence(p_matrix, start_id, length, rng):
    """Autoregressively sample `length` token ids from a bigram matrix, starting with `start_id`."""
    sequence = np.empty(length, dtype=int)
    sequence[0] = start_id

    for i in range(1, length):
        sequence[i] = sample_next_token(p_matrix, sequence[i - 1], rng)

    return sequence

# Step 53 - decode_generated_sequence
def decode_generated_sequence(ids, itos):
    """Decode a generated 1D array/list of token ids into a string via itos."""
    return "".join(itos[int(i)] for i in ids)

# Step 54 - log_prob_of_pair
def log_prob_of_pair(p_matrix, current_id, next_id):
    """Return the log probability of a single (current, next) bigram."""
    prob = index_element(p_matrix, current_id, next_id)
    return float(array_log(prob))

# Step 55 - sum_negative_log_probs
def sum_negative_log_probs(p_matrix, data):
    total = 0.0
    for t in range(len(data) - 1):
        total -= log_prob_of_pair(p_matrix, data[t], data[t + 1])
    return float(total)

# Step 56 - average_nll
def average_nll(p_matrix, data):
    total_nll = sum_negative_log_probs(p_matrix, data)
    return float(total_nll / (len(data) - 1))

# Step 57 - initialize_w_random
import numpy as np

def initialize_w_random(vocab_size, rng):
    """Return a (vocab_size, vocab_size) float64 matrix of N(0,1) samples drawn from rng."""
    return rng.standard_normal((vocab_size, vocab_size))

# Step 58 - scale_w_small
import numpy as np

def scale_w_small(w_matrix, scale):
    """Return w_matrix scaled by the given small factor."""
    return w_matrix * scale

# Step 59 - one_hot_encode_batch
import numpy as np

def one_hot_encode_batch(ids, vocab_size):
    """Convert a 1D array of token ids into a (N, vocab_size) one-hot matrix."""
    one_hot = make_2d_zeros(len(ids), vocab_size)
    one_hot[np.arange(len(ids)), ids] = 1.0
    return one_hot

# Step 60 - forward_logits_onehot
def forward_logits_onehot(onehot, w_matrix):
    return matmul(onehot, w_matrix)

# Step 61 - observe_lookup_equivalence
import numpy as np

def observe_lookup_equivalence(w, ids):
    """Show that one-hot @ W equals W[ids] for a small example.
    Returns a dict with keys 'onehot_result' and 'index_result'.
    """
    onehot = one_hot_encode_batch(ids, w.shape[0])
    onehot_result = forward_logits_onehot(onehot, w)
    index_result = w[ids]

    return {
        "onehot_result": onehot_result,
        "index_result": index_result,
    }

# Step 62 - forward_logits_lookup
def forward_logits_lookup(w, ids):
    """Return logits (B, V) by gathering rows of w at positions ids."""
    return w[ids]

# Step 63 - logits_to_probs_rowwise
def logits_to_probs_rowwise(logits):
    return stable_softmax_2d_rowwise(logits)

# Step 64 - gather_correct_token_probs
def gather_correct_token_probs(probs, targets):
    """Return probs[i, targets[i]] for each i, shape (B,)."""
    return probs[np.arange(len(targets)), targets]

# Step 65 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, targets):
    """Mean negative log-likelihood over a batch."""
    correct_probs = gather_correct_token_probs(probs, targets)
    return float(-array_log(correct_probs).mean())

# Step 66 - derive_dlogits_on_paper
def derive_dlogits_on_paper():
    """Return a string summarizing the derivation of dL/dlogits for mean cross-entropy."""
    return (
        "Let the loss be the mean cross-entropy over a batch. "
        "First apply the softmax to the logits to obtain probabilities. "
        "Then compute the negative log-likelihood of the correct class for each example "
        "and average over the batch. "
        "Differentiating the loss with respect to the logits and combining the derivatives "
        "of the softmax and log terms causes the Jacobian to simplify, yielding:\n\n"
        "dL/dlogits = (probs - onehot(targets)) / B"
    )

# Step 67 - compute_dlogits
def compute_dlogits(probs, targets):
    """Gradient of mean cross-entropy w.r.t. logits. probs: (B,V), targets: (B,)."""
    B = len(targets)
    dlogits = probs.copy()
    dlogits[np.arange(B), targets] -= 1.0
    dlogits /= B
    return dlogits

# Step 68 - derive_dw_on_paper
def derive_dw_on_paper():
    """Return a short written derivation of dL/dW for the lookup-as-matmul forward."""
    return (
        "Forward: logits = onehot(ids) @ W, equivalently logits[b] = W[ids[b]].\n"
        "Shapes: ids (B,), onehot O (B, V), W (V, D), logits (B, D), dlogits (B, D).\n"
        "Chain rule: dL/dW = O.T @ dlogits, shape (V, D).\n"
        "Since O has a single 1 per row at column ids[b], O.T @ dlogits sums rows of dlogits into rows of dW.\n"
        "Row v of dW equals the sum of dlogits[b] over all b with ids[b] == v.\n"
        "Implementation: scatter-add dlogits rows into dW at indices ids."
    )

# Step 69 - compute_dw_scatter_add
import numpy as np

def compute_dw_scatter_add(ids, dlogits, vocab_size):
    """Scatter-add dlogits rows into dW at positions given by ids."""
    dW = np.zeros((vocab_size, dlogits.shape[1]), dtype=float)
    np.add.at(dW, ids, dlogits)
    return dW

# Step 70 - sgd_update_w
import numpy as np

def sgd_update_w(w, dw, learning_rate):
    """Apply one SGD step: return w - learning_rate * dw as a new array."""
    return w - learning_rate * dw

# Step 71 - run_one_training_step
def run_one_training_step(w, ids, targets, learning_rate):
    """Run forward, loss, backward, and SGD update once. Return {'w': new_w, 'loss': float}."""
    logits = forward_logits_lookup(w, ids)
    probs = logits_to_probs_rowwise(logits)
    loss = cross_entropy_loss(probs, targets)

    dlogits = compute_dlogits(probs, targets)
    dw = compute_dw_scatter_add(ids, dlogits, w.shape[0])

    new_w = sgd_update_w(w, dw, learning_rate)

    return {
        "w": new_w,
        "loss": float(loss),
    }

# Step 72 - train_neural_bigram_loop
def train_neural_bigram_loop(w, data, block_size, batch_size, learning_rate, num_steps, log_every):
    """Run the neural bigram training loop and return {'w', 'loss_history'}."""
    rng = np.random.default_rng(0)
    loss_history = []

    for step in range(num_steps):
        X, Y = get_batch(data, block_size, batch_size, rng)

        ids = X.reshape(-1)
        targets = Y.reshape(-1)

        result = run_one_training_step(w, ids, targets, learning_rate)
        w = result["w"]

        if step % log_every == 0:
            loss_history.append(result["loss"])

    return {
        "w": w,
        "loss_history": loss_history,
    }

# Step 73 - sample_from_neural_bigram
def sample_from_neural_bigram(w, start_id, num_tokens, itos):
    """Generate a string by repeatedly sampling from softmax of W[id]."""
    rng = np.random.default_rng(0)

    ids = [start_id]
    current_id = start_id

    for _ in range(num_tokens):
        logits = forward_logits_lookup(w, np.array([current_id]))   # (1, V)
        probs = logits_to_probs_rowwise(logits)[0]                  # (V,)
        current_id = rng.choice(len(probs), p=probs)
        ids.append(int(current_id))

    return decode_ids(ids, itos)

# Step 74 - linear_forward
def linear_forward(x, w):
    y = x @ w
    return {
        "y": y,
        "cache": {
            "x": x,
            "w": w,
        },
    }

# Step 75 - derive_dx_on_paper
def derive_dx_on_paper():
    """Return notes deriving dL/dX = dY @ W.T for Y = X @ W."""
    return (
        "Y = X @ W\n"
        "dL/dX = dY @ W.T\n"
        "shapes: X (B, In), W (In, Out), dY (B, Out) -> dL/dX (B, In)"
    )

# Step 76 - derive_linear_dw_on_paper
def derive_linear_dw_on_paper():
    """Return a string with the derivation of dL/dW for Y = X @ W."""
    return (
        "Forward: Y = X @ W.\n"
        "By the chain rule, the weight gradient is obtained by multiplying the transpose "
        "of the input matrix with the upstream gradient.\n"
        "dL/dW = X.T @ dY\n"
        "Shape: X (B, D_in), dY (B, D_out), dL/dW (D_in, D_out)."
    )

# Step 77 - linear_backward_dx
def linear_backward_dx(dy, cache):
    w = cache["w"]
    return dy @ w.T

# Step 78 - linear_backward_dw
def linear_backward_dw(dy, cache):
    """Return dL/dW for a linear layer Y = X @ W."""
    x = cache["x"]
    return x.T @ dy

# Step 79 - bias_add_forward
def bias_add_forward(x, b):
    """Add bias vector b (D,) to every row of x (B, D).

    Returns {'y': ndarray (B, D), 'cache': {'b_shape': tuple}}.
    """
    y = vector_matrix_broadcast_add(x, b)
    return {
        "y": y,
        "cache": {
            "b_shape": b.shape,
        },
    }

# Step 80 - bias_add_backward_db
def bias_add_backward_db(dy, cache):
    """Compute db from upstream gradient dy for y = x + b."""
    return dy.sum(axis=0).reshape(cache["b_shape"])

# Step 81 - relu_forward
def relu_forward(x):
    """Apply elementwise ReLU and cache the input for backward.

    Returns a dict with keys 'y' (activated array) and 'cache' (dict with 'x').
    """
    return {
        "y": np.maximum(x, 0),
        "cache": {
            "x": x,
        },
    }

# Step 82 - relu_backward
def relu_backward(dy, cache):
    """Backward pass for ReLU. cache['x'] holds the original input."""
    x = cache["x"]
    return dy * (x > 0)

# Step 83 - softmax_cross_entropy_backward
def softmax_cross_entropy_backward(probs, targets):
    """Return dL/dlogits for mean cross-entropy with softmax probs."""
    B = len(targets)
    dlogits = probs.copy()
    dlogits[np.arange(B), targets] -= 1.0
    dlogits /= B
    return dlogits

# Step 84 - layernorm_forward_mean
import numpy as np

def layernorm_forward_mean(x):
    """Return the per-row mean of x with shape (B, 1)."""
    return sum_keepdims(x, axis=-1) / x.shape[-1]

# Step 85 - layernorm_forward_variance
import numpy as np

def layernorm_forward_variance(x, mean):
    """Compute the per-row (biased) variance of x given its per-row mean.

    Args:
        x: ndarray of shape (B, D).
        mean: ndarray of shape (B, 1), the per-row mean of x.

    Returns:
        var: ndarray of shape (B, 1), the per-row variance.
    """
    squared_diff = (x - mean) ** 2
    return sum_keepdims(squared_diff, axis=-1) / x.shape[-1]

# Step 86 - layernorm_forward_normalize
import numpy as np

def layernorm_forward_normalize(x, mean, var, eps):
    """Normalize each row of x to zero mean and unit variance."""
    return (x - mean) / np.sqrt(var + eps)

# Step 87 - layernorm_forward_affine
def layernorm_forward_affine(x, gamma, beta, eps):
    """Run LayerNorm forward over rows of x with affine params gamma, beta."""
    mean = layernorm_forward_mean(x)
    var = layernorm_forward_variance(x, mean)
    x_hat = layernorm_forward_normalize(x, mean, var, eps)

    y = vector_matrix_broadcast_add(
        elementwise_multiply(x_hat, gamma),
        beta
    )

    return {
        "y": y,
        "cache": {
            "x": x,
            "x_hat": x_hat,
            "mean": mean,
            "var": var,
            "gamma": gamma,
            "eps": eps,
        },
    }

# Step 88 - layernorm_backward_subtract_mean
import numpy as np

def layernorm_backward_subtract_mean(dy, cache):
    """Gradient through y = x - mean(x, axis=1, keepdims=True).

    dy: (B, D) upstream gradient w.r.t. the centered output.
    cache: dict with keys 'x' (B, D) and 'mean' (B,).
    Returns dx of shape (B, D).
    """
    D = cache["x"].shape[-1]
    return dy - np.sum(dy, axis=-1, keepdims=True) / D

# Step 89 - layernorm_backward_divide_std
def layernorm_backward_divide_std(dy, cache):
    """Propagate dy through the divide-by-std step of LayerNorm."""
    std = np.sqrt(cache["var"] + cache["eps"])
    return dy / std

# Step 90 - layernorm_backward_full
import numpy as np

def layernorm_backward_full(dy, cache):
    """Full LayerNorm backward. Return {'dx', 'dgamma', 'dbeta'}."""
    x = cache["x"]
    x_hat = cache["x_hat"]
    var = cache["var"]
    gamma = cache["gamma"]
    eps = cache["eps"]

    D = x.shape[-1]

    # Affine parameter gradients
    dgamma = np.sum(dy * x_hat, axis=0)
    dbeta = np.sum(dy, axis=0)

    # Gradient wrt normalized activations
    dx_hat = dy * gamma

    inv_std = 1.0 / np.sqrt(var + eps)

    # Full LayerNorm input gradient
    dx = (
        inv_std / D
    ) * (
        D * dx_hat
        - np.sum(dx_hat, axis=-1, keepdims=True)
        - x_hat * np.sum(dx_hat * x_hat, axis=-1, keepdims=True)
    )

    return {
        "dx": dx,
        "dgamma": dgamma,
        "dbeta": dbeta,
    }

# Step 91 - layernorm_backward_implementation
def layernorm_backward_implementation(d_out, cache):
    return layernorm_backward_full(d_out, cache)

# Step 92 - create_token_embedding
def create_token_embedding(vocab_size, d_model, scale=0.02):
    """Initialize the token embedding matrix E of shape (vocab_size, d_model)."""
    return np.random.randn(vocab_size, d_model) * scale

# Step 93 - token_embedding_forward
def token_embedding_forward(token_ids, embedding_matrix):
    """Look up token embeddings for a batch of integer token ids.

    Inputs:
        token_ids: ndarray of shape (B, T), dtype int
        embedding_matrix: ndarray of shape (V, d_model)
    Returns:
        out: ndarray of shape (B, T, d_model)
        cache: dict with keys 'token_ids', 'vocab_size'
    """
    out = embedding_matrix[token_ids]
    cache = {
        "token_ids": token_ids,
        "vocab_size": embedding_matrix.shape[0],
    }
    return out, cache

# Step 94 - token_embedding_backward
import numpy as np

def token_embedding_backward(d_out, cache):
    token_ids = cache["token_ids"]
    vocab_size = cache["vocab_size"]
    d_model = d_out.shape[-1]

    dE = np.zeros((vocab_size, d_model), dtype=d_out.dtype)

    np.add.at(
        dE,
        token_ids.reshape(-1),
        d_out.reshape(-1, d_model)
    )

    return dE

# Step 95 - create_positional_embedding
def create_positional_embedding(block_size, d_model, scale=0.02):
    """Initialize the learned positional embedding matrix P of shape (block_size, d_model)."""
    p = make_2d_random(block_size, d_model, seed=None)
    return scale_w_small(p, scale)

# Step 96 - slice_positional_embedding
import numpy as np

def slice_positional_embedding(positional_matrix, seq_len):
    """Return the first seq_len rows of the positional embedding matrix."""
    return positional_matrix[:seq_len]

# Step 97 - add_token_and_positional_embeddings
def add_token_and_positional_embeddings(token_emb, pos_emb):
    """Sum token embeddings (B,T,d_model) and positional embeddings (T,d_model)."""
    return token_emb + pos_emb

# Step 98 - embedding_sum_backward
def embedding_sum_backward(d_out):
    """Backprop through H = token_emb + pos_emb (with broadcasting over batch)."""
    return {
        "d_token_emb": d_out,
        "d_pos_emb": np.sum(d_out, axis=0),
    }

# Step 99 - create_qkv_projections
def create_qkv_projections(d_model, d_head, scale=0.02):
    return {
        "Wq": scale_w_small(make_2d_random(d_model, d_head, seed=0), scale),
        "Wk": scale_w_small(make_2d_random(d_model, d_head, seed=1), scale),
        "Wv": scale_w_small(make_2d_random(d_model, d_head, seed=2), scale),
    }

# Step 100 - compute_query
import numpy as np

def compute_query(x, w_q):
    """Project x (B, T, d_model) into queries Q (B, T, d_head) using w_q."""
    return x @ w_q

# Step 101 - compute_key
def compute_key(x, w_k):
    """Project x through Wk to get keys K of shape (B, T, d_head)."""
    return x @ w_k

# Step 102 - compute_value
def compute_value(x, w_v):
    return matmul(x, w_v)

# Step 103 - compute_attention_scores
import numpy as np

def compute_attention_scores(q, k):
    """Return raw attention scores Q @ K^T with shape (B, T, T)."""
    return q @ k.transpose(0, 2, 1)

# Step 104 - scale_attention_scores
import numpy as np

def scale_attention_scores(scores, d_head):
    """Rescale (B, T, T) attention scores by a function of d_head."""
    return scores / np.sqrt(d_head)

# Step 105 - build_causal_mask
import numpy as np

def build_causal_mask(seq_len):
    """Return a (seq_len, seq_len) boolean lower-triangular mask."""
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))

# Step 106 - apply_causal_mask
import numpy as np

def apply_causal_mask(scaled_scores, causal_mask):
    """Replace future positions in scaled_scores with -inf using causal_mask."""
    return np.where(causal_mask, scaled_scores, -np.inf)

# Step 107 - softmax_attention_weights
import numpy as np

def softmax_attention_weights(masked_scores):
    """Row-wise stable softmax over the last axis of (B, T, T) scores."""
    shifted = masked_scores - np.max(masked_scores, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

# Step 108 - attention_weighted_values
import numpy as np

def attention_weighted_values(attn, v):
    """Combine attention weights with values: out = attn @ V.

    attn: (B, T, T) softmaxed attention weights
    v:    (B, T, d_head) value vectors
    returns: (B, T, d_head)
    """
    return attn @ v

# Step 109 - apply_output_projection
import numpy as np

def apply_output_projection(attn_out, w_o):
    """Project attention output (B,T,d_head) through Wo (d_head,d_model)."""
    return attn_out @ w_o

# Step 110 - output_projection_backward
def output_projection_backward(d_proj, cache):
    """Backprop through proj = attn_out @ w_o. Return {'d_attn_out', 'dw_o'}."""
    attn_out = cache["attn_out"]
    w_o = cache["w_o"]

    d_attn_out = d_proj @ w_o.T
    dw_o = np.sum(attn_out.transpose(0, 2, 1) @ d_proj, axis=0)

    return {
        "d_attn_out": d_attn_out,
        "dw_o": dw_o,
    }

# Step 111 - attention_value_backward
import numpy as np

def attention_value_backward(d_attn_out, cache):
    """Backprop through out = attn @ V.

    d_attn_out: (B, T, d_head) upstream gradient w.r.t. attention output.
    cache: dict with 'attn' of shape (B, T, T) and 'v' of shape (B, T, d_head).
    Returns dict with 'd_attn' (B, T, T) and 'd_v' (B, T, d_head).
    """
    # TODO: backprop through out = attn @ V to obtain gradients for attn and V.
    attn = cache["attn"]
    v = cache["v"]

    d_attn = d_attn_out @ v.transpose(0, 2, 1)
    d_v = attn.transpose(0, 2, 1) @ d_attn_out

    return {
        "d_attn": d_attn,
        "d_v": d_v,
    }

# Step 112 - masked_softmax_backward
import numpy as np

def masked_softmax_backward(d_attn, cache):
    """Backprop through the masked row-wise softmax.

    d_attn: ndarray of shape (B, T, T) -- gradient w.r.t. attention weights.
    cache: dict with 'attn' (B,T,T) and 'causal_mask' (T,T) boolean.
    Returns d_masked_scores of shape (B, T, T).
    """
    attn = cache["attn"]
    causal_mask = cache["causal_mask"]

    d_masked_scores = attn * (
        d_attn - np.sum(d_attn * attn, axis=-1, keepdims=True)
    )

    d_masked_scores = np.where(causal_mask, d_masked_scores, 0.0)

    return d_masked_scores

# Step 113 - scale_scores_backward
import numpy as np

def scale_scores_backward(d_scaled_scores, d_head):
    """Backprop through the 1/sqrt(d_head) attention score scaling."""
    return d_scaled_scores / np.sqrt(d_head)

# Step 114 - qk_scores_backward
import numpy as np

def qk_scores_backward(d_scores, cache):
    """Backprop through scores = Q @ K^T.

    d_scores: (B, T, T)
    cache: dict with 'q' and 'k', each (B, T, d_head)
    returns: {'d_q': (B, T, d_head), 'd_k': (B, T, d_head)}
    """
    q = cache["q"]
    k = cache["k"]

    d_q = d_scores @ k
    d_k = d_scores.transpose(0, 2, 1) @ q

    return {
        "d_q": d_q,
        "d_k": d_k,
    }

# Step 115 - qkv_projection_backward
def qkv_projection_backward(d_q, d_k, d_v, cache):
    x = cache["x"]
    w_q = cache["w_q"]
    w_k = cache["w_k"]
    w_v = cache["w_v"]

    # Gradient w.r.t. input
    dx = (
        d_q @ w_q.T +
        d_k @ w_k.T +
        d_v @ w_v.T
    )

    # Flatten batch and time dimensions
    x_flat = x.reshape(-1, x.shape[-1])
    dq_flat = d_q.reshape(-1, d_q.shape[-1])
    dk_flat = d_k.reshape(-1, d_k.shape[-1])
    dv_flat = d_v.reshape(-1, d_v.shape[-1])

    dw_q = x_flat.T @ dq_flat
    dw_k = x_flat.T @ dk_flat
    dw_v = x_flat.T @ dv_flat

    return {
        "dx": dx,
        "dw_q": dw_q,
        "dw_k": dw_k,
        "dw_v": dw_v,
    }

# Step 116 - choose_attention_head_config
def choose_attention_head_config(d_model, n_heads):
    """Return a config dict {'n_heads', 'd_head', 'd_model'} for multi-head attention."""
    if d_model % n_heads != 0:
        raise ValueError("d_model must be evenly divisible by n_heads")

    return {
        "n_heads": n_heads,
        "d_head": d_model // n_heads,
        "d_model": d_model,
    }

# Step 117 - create_multihead_qkv_projections
def create_multihead_qkv_projections(d_model, scale=0.02):
    """Initialize Wq, Wk, Wv as (d_model, d_model) matrices for multi-head attention."""
    return {
        "Wq": scale_w_small(make_2d_random(d_model, d_model, seed=0), scale),
        "Wk": scale_w_small(make_2d_random(d_model, d_model, seed=1), scale),
        "Wv": scale_w_small(make_2d_random(d_model, d_model, seed=2), scale),
    }

# Step 118 - create_multihead_output_projection
def create_multihead_output_projection(d_model, scale=0.02):
    """Initialize Wo of shape (d_model, d_model) for multi-head attention output projection."""
    return scale_w_small(
        make_2d_random(d_model, d_model, seed=0),
        scale
    )

# Step 119 - reshape_to_heads
import numpy as np

def reshape_to_heads(x, n_heads, d_head):
    """Reshape (B, T, d_model) into (B, T, n_heads, d_head)."""
    B, T, _ = x.shape
    return x.reshape(B, T, n_heads, d_head)

# Step 120 - transpose_heads_to_front
import numpy as np

def transpose_heads_to_front(x_heads):
    """Transpose (B, T, n_heads, d_head) to (B, n_heads, T, d_head)."""
    return np.transpose(x_heads, (0, 2, 1, 3)).copy()

# Step 121 - get_multihead_n_heads
def get_multihead_n_heads(config):
    return config["n_heads"]

# Step 122 - get_multihead_sequence_length
import numpy as np

def get_multihead_sequence_length(x):
    """Return T from x of shape (B, T, d_model)."""
    return get_array_shape(x)[1]

# Step 123 - compute_d_head
def compute_d_head(d_model, n_heads):
    if d_model % n_heads != 0:
        raise ValueError("d_model must be evenly divisible by n_heads")
    return d_model // n_heads

# Step 124 - multihead_masked_softmax_scores
def multihead_masked_softmax_scores(scores, mask):
    """Apply causal mask and row-wise softmax to multi-head attention scores.

    Args:
        scores: ndarray of shape (B, n_heads, T, T)
        mask:   ndarray of shape (T, T), True where positions are kept

    Returns:
        weights: ndarray of shape (B, n_heads, T, T)
    """
    masked = apply_causal_mask(scores, mask)

    B, H, T, _ = masked.shape
    flat = masked.reshape(B * H * T, T)
    flat = stable_softmax_2d_rowwise(flat)

    return flat.reshape(B, H, T, T)

# Step 125 - multihead_weighted_sum
import numpy as np

def multihead_weighted_sum(weights, v_heads):
    """Compute per-head attention output as weights @ V across all heads."""
    return weights @ v_heads

# Step 126 - transpose_heads_to_back
def transpose_heads_to_back(x_heads):
    return np.transpose(x_heads, (0, 2, 1, 3)).copy()

# Step 127 - get_multihead_output_sequence_length
def get_multihead_output_sequence_length(x_heads_back):
    """Return T from a (B, T, n_heads, d_head) tensor."""
    return int(x_heads_back.shape[1])

# Step 128 - merge_heads_to_d_model
import numpy as np

def merge_heads_to_d_model(x_heads_back):
    """Reshape (B, T, n_heads, d_head) into (B, T, d_model)."""
    B, T, n_heads, d_head = x_heads_back.shape
    return x_heads_back.reshape(B, T, n_heads * d_head)

# Step 129 - multihead_output_projection_forward
def multihead_output_projection_forward(merged, w_out, b_out):
    """Project the merged multi-head output through the output linear layer.

    Inputs:
      merged: (B, T, d_model)
      w_out:  (d_model, d_model)
      b_out:  (d_model,)
    Returns dict with keys {'out', 'cache'}; cache holds {'merged', 'w_out'}.
    """
    linear = linear_forward(merged, w_out)
    biased = bias_add_forward(linear["y"], b_out)

    return {
        "out": biased["y"],
        "cache": {
            "merged": merged,
            "w_out": w_out,
        },
    }

# Step 130 - multihead_reshape_transpose_backward
def multihead_reshape_transpose_backward(d_merged, shape_info):
    """Invert merge_heads_to_d_model to recover (B, n_heads, T, d_head) gradients."""
    x_heads = reshape_to_heads(
        d_merged,
        shape_info["n_heads"],
        shape_info["d_head"],
    )
    return transpose_heads_to_front(x_heads)

# Step 131 - ffn_linear_one_forward
def ffn_linear_one_forward(x, w1, b1):
    """First FFN linear: lift (B, T, d_model) up to (B, T, d_ff) and add bias."""
    linear = linear_forward(x, w1)
    biased = bias_add_forward(linear["y"], b1)

    return {
        "h1": biased["y"],
        "cache": {
            "x": x,
            "w1": w1,
        },
    }

# Step 132 - ffn_activation_forward
def ffn_activation_forward(h1):
    """Apply ReLU to FFN hidden pre-activations.

    Args:
        h1: ndarray of shape (B, T, d_ff)

    Returns:
        a1: ndarray of shape (B, T, d_ff)
        cache: dict with key 'h1'
    """
    out = relu_forward(h1)
    return out["y"], {"h1": h1}

# Step 133 - ffn_linear_two_forward
def ffn_linear_two_forward(a1, w2, b2):
    linear = linear_forward(a1, w2)
    biased = bias_add_forward(linear["y"], b2)

    return {
        "h2": biased["y"],
        "cache": {
            "a1": a1,
            "w2": w2,
        },
    }

# Step 134 - ffn_backward
def ffn_backward(d_out, cache):
    """Backprop through linear2 -> ReLU -> linear1 of the FFN.

    cache keys: 'x', 'w1', 'h1', 'a1', 'w2'.
    Returns dict with keys: 'dx', 'dw1', 'db1', 'dw2', 'db2'.
    """
    x = cache["x"]
    w1 = cache["w1"]
    h1 = cache["h1"]
    a1 = cache["a1"]
    w2 = cache["w2"]

    B, T, _ = x.shape

    # Flatten (B,T,*) -> (B*T,*)
    x_flat = x.reshape(B * T, -1)
    h1_flat = h1.reshape(B * T, -1)
    a1_flat = a1.reshape(B * T, -1)
    dout_flat = d_out.reshape(B * T, -1)

    # Linear 2
    linear2_cache = {"x": a1_flat, "w": w2}
    da1 = linear_backward_dx(dout_flat, linear2_cache)
    dw2 = linear_backward_dw(dout_flat, linear2_cache)
    db2 = bias_add_backward_db(
        dout_flat,
        {"b_shape": (w2.shape[1],)}
    )

    # ReLU
    dh1 = relu_backward(da1, {"x": h1_flat})

    # Linear 1
    linear1_cache = {"x": x_flat, "w": w1}
    dx_flat = linear_backward_dx(dh1, linear1_cache)
    dw1 = linear_backward_dw(dh1, linear1_cache)
    db1 = bias_add_backward_db(
        dh1,
        {"b_shape": (w1.shape[1],)}
    )

    dx = dx_flat.reshape(x.shape)

    return {
        "dx": dx,
        "dw1": dw1,
        "db1": db1,
        "dw2": dw2,
        "db2": db2,
    }

# Step 135 - residual_forward
def residual_forward(x, sublayer_out):
    """Return x + sublayer_out for a residual connection."""
    return x + sublayer_out

# Step 136 - residual_backward
def residual_backward(d_y):
    """Backprop through y = x + sublayer_out. Returns (d_x, d_sublayer_out)."""
    return d_y.copy(), d_y.copy()

# Step 137 - pre_layernorm_sublayer_forward
def pre_layernorm_sublayer_forward(x, ln_params, sublayer_fn, sublayer_params):
    ln = layernorm_forward_affine(
        x,
        ln_params["gamma"],
        ln_params["beta"],
        ln_params.get("eps", 1e-5),
    )

    sub = sublayer_fn(ln["y"], sublayer_params)

    y = residual_forward(x, sub["y"])

    return {
        "y": y,
        "cache": {
            "x": x,
            "ln_cache": ln["cache"],
            "sublayer_cache": sub["cache"],
        },
    }

# Step 138 - transformer_block_forward
def transformer_block_forward(x, block_params):
    """Run one pre-LN Transformer block forward.

    Order:
        x
        -> LN1 -> Multi-Head Self-Attention -> Residual
        -> LN2 -> FFN                       -> Residual
    """

    def attention_sublayer_forward(x_norm, params):
        n_heads = params["n_heads"]

        B, T, d_model = x_norm.shape
        d_head = compute_d_head(d_model, n_heads)

        # Q, K, V projections
        q = compute_query(x_norm, params["Wq"])
        k = compute_key(x_norm, params["Wk"])
        v = compute_value(x_norm, params["Wv"])

        # Split into heads and move heads forward
        q_heads = transpose_heads_to_front(
            reshape_to_heads(q, n_heads, d_head)
        )
        k_heads = transpose_heads_to_front(
            reshape_to_heads(k, n_heads, d_head)
        )
        v_heads = transpose_heads_to_front(
            reshape_to_heads(v, n_heads, d_head)
        )

        # Scaled dot-product attention
        scores = q_heads @ k_heads.transpose(0, 1, 3, 2)
        scores = scores / np.sqrt(d_head)

        causal_mask = build_causal_mask(T)

        attn = multihead_masked_softmax_scores(
            scores,
            causal_mask,
        )

        weighted = multihead_weighted_sum(
            attn,
            v_heads,
        )

        # Merge heads
        heads_back = transpose_heads_to_back(weighted)
        merged = merge_heads_to_d_model(heads_back)

        # Output projection
        projected = multihead_output_projection_forward(
            merged,
            params["Wo"],
            params["bo"],
        )

        return {
            "y": projected["out"],
            "cache": {
                "x": x_norm,
                "q": q_heads,
                "k": k_heads,
                "v": v_heads,
                "attn": attn,
                "causal_mask": causal_mask,
                "merged": merged,
                "w_q": params["Wq"],
                "w_k": params["Wk"],
                "w_v": params["Wv"],
                "w_out": params["Wo"],
                "n_heads": n_heads,
                "d_head": d_head,
            },
        }

    def ffn_sublayer_forward(x_norm, params):
        # First FFN linear layer
        first = ffn_linear_one_forward(
            x_norm,
            params["w1"],
            params["b1"],
        )

        # Activation
        a1, activation_cache = ffn_activation_forward(
            first["h1"]
        )

        # Second FFN linear layer
        second = ffn_linear_two_forward(
            a1,
            params["w2"],
            params["b2"],
        )

        return {
            "y": second["h2"],
            "cache": {
                "x": x_norm,
                "w1": params["w1"],
                "h1": first["h1"],
                "a1": a1,
                "w2": params["w2"],
                "activation_cache": activation_cache,
            },
        }

    # Attention branch:
    # x -> LN1 -> attention -> residual
    attn_branch = pre_layernorm_sublayer_forward(
        x,
        block_params["ln1"],
        attention_sublayer_forward,
        block_params["attn"],
    )

    # FFN branch:
    # attn output -> LN2 -> FFN -> residual
    ffn_branch = pre_layernorm_sublayer_forward(
        attn_branch["y"],
        block_params["ln2"],
        ffn_sublayer_forward,
        block_params["ffn"],
    )

    return {
        "y": ffn_branch["y"],
        "cache": {
            "attn_branch": attn_branch["cache"],
            "ffn_branch": ffn_branch["cache"],
        },
    }

# Step 139 - transformer_block_backward
def transformer_block_backward(d_y, cache, block_params):
    """Backward pass for a pre-LN Transformer block.

    Args:
        d_y: upstream gradient w.r.t. block output, shape (B, T, D).
        cache: dict from transformer_block_forward, with keys
            'attn_branch' and 'ffn_branch'.
        block_params: nested dict with keys 'ln1', 'attn', 'ln2', 'ffn'.

    Returns:
        (d_x, grads) where d_x has shape (B, T, D) and grads is a nested
        dict mirroring block_params.
    """

    # Recover the original block input and rebuild a complete cache so that
    # all backward helpers have the fields they need.
    x = cache["attn_branch"]["x"]
    full_cache = _complete_block_cache(x, block_params)

    attn_branch = full_cache["attn_branch"]
    ffn_branch = full_cache["ffn_branch"]

    # ------------------------------------------------------------------
    # Backward through FFN branch:
    #
    # y = h1 + FFN(LN2(h1))
    #
    # Therefore:
    #   d_h1 = d_y                         # residual path
    #        + d(LN2)/d_h1                 # sublayer path
    # ------------------------------------------------------------------
    d_ln2, ffn_grads = _ffn_sublayer_backward(
        d_y,
        ffn_branch["sublayer_cache"],
        block_params["ffn"],
    )

    d_h1_from_ln2, d_gamma2, d_beta2 = layernorm_backward_affine(
        d_ln2,
        ffn_branch["ln_cache"],
    )

    d_h1 = d_y + d_h1_from_ln2

    # ------------------------------------------------------------------
    # Backward through attention branch:
    #
    # h1 = x + Attn(LN1(x))
    #
    # Therefore:
    #   d_x = d_h1                       # residual path
    #       + d(LN1)/dx                  # sublayer path
    # ------------------------------------------------------------------
    d_ln1, attn_grads = _attn_sublayer_backward(
        d_h1,
        attn_branch["sublayer_cache"],
        block_params["attn"],
    )

    d_x_from_ln1, d_gamma1, d_beta1 = layernorm_backward_affine(
        d_ln1,
        attn_branch["ln_cache"],
    )

    d_x = d_h1 + d_x_from_ln1

    # ------------------------------------------------------------------
    # Assemble gradients to mirror block_params
    # ------------------------------------------------------------------
    grads = {
        "ln1": {
            "gamma": d_gamma1,
            "beta": d_beta1,
        },
        "ln2": {
            "gamma": d_gamma2,
            "beta": d_beta2,
        },
        "attn": {
            "Wq": attn_grads["Wq"],
            "Wk": attn_grads["Wk"],
            "Wv": attn_grads["Wv"],
            "Wo": attn_grads["Wo"],
            "bo": attn_grads["bo"],
        },
        "ffn": {
            "w1": ffn_grads["w1"],
            "b1": ffn_grads["b1"],
            "w2": ffn_grads["w2"],
            "b2": ffn_grads["b2"],
        },
    }

    return d_x, grads

# Step 140 - stack_transformer_blocks
def stack_transformer_blocks(n_layers, d_model, n_heads, d_ff):
    """Build a list of n_layers Transformer block parameter dicts.

    Each block dict has keys 'ln1', 'attn', 'ln2', 'ffn'.
    """
    blocks = []

    for _ in range(n_layers):
        block = {
            "ln1": {
                "gamma": np.ones(d_model),
                "beta": np.zeros(d_model),
            },
            "attn": {
                "Wq": scale_w_small(
                    make_2d_random(d_model, d_model, 0),
                    0.01,
                ),
                "Wk": scale_w_small(
                    make_2d_random(d_model, d_model, 1),
                    0.01,
                ),
                "Wv": scale_w_small(
                    make_2d_random(d_model, d_model, 2),
                    0.01,
                ),
                "Wo": scale_w_small(
                    make_2d_random(d_model, d_model, 3),
                    0.01,
                ),
                "bo": np.zeros(d_model),
            },
            "ln2": {
                "gamma": np.ones(d_model),
                "beta": np.zeros(d_model),
            },
            "ffn": {
                "W1": scale_w_small(
                    make_2d_random(d_model, d_ff, 4),
                    0.01,
                ),
                "b1": np.zeros(d_ff),
                "W2": scale_w_small(
                    make_2d_random(d_ff, d_model, 5),
                    0.01,
                ),
                "b2": np.zeros(d_model),
            },
        }

        blocks.append(block)

    return blocks

# Step 141 - forward_through_all_blocks
def forward_through_all_blocks(x, blocks):
    """Run x through every Transformer block in order, collecting caches."""
    caches = []
    y = x

    for block_params in blocks:
        out = transformer_block_forward(y, block_params)
        y = out["y"]
        caches.append(out["cache"])

    return y, caches

# Step 142 - backward_through_all_blocks
def backward_through_all_blocks(d_y, caches, blocks):
    """Backprop through a stack of Transformer blocks.

    Inputs:
      d_y     : (B, T, d_model) upstream gradient at the top of the stack
      caches  : list of per-block forward caches
      blocks  : list of per-block parameter dicts

    Returns:
      d_x        : (B, T, d_model) gradient at the input of the stack
      grads_list : list of per-block parameter-gradient dicts, in block order
    """
    d_x = d_y
    grads_list = [None] * len(blocks)

    for i in range(len(blocks) - 1, -1, -1):
        d_x, grads_block = transformer_block_backward(
            d_x,
            caches[i],
            blocks[i],
        )

        # Store gradient in original block order.
        grads_list[i] = grads_block

    return d_x, grads_list

# Step 143 - final_layernorm_forward
def final_layernorm_forward(x, gamma, beta):
    """Apply LayerNorm to a (B, T, d_model) tensor with affine params gamma, beta.

    Returns (y, cache) where cache has keys 'x', 'mean', 'var', 'x_hat', 'gamma'.
    """
    # Normalize independently for every (B, T) position
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)

    # Numerical stability
    x_hat = (x - mean) / np.sqrt(var + 1e-5)

    # Affine transformation
    y = gamma * x_hat + beta

    cache = {
        "x": x,
        "mean": mean,
        "var": var,
        "x_hat": x_hat,
        "gamma": gamma,
    }

    return y, cache

# Step 144 - lm_head_linear_forward
def lm_head_linear_forward(x, w_lm, b_lm):
    """Project hidden states (B,T,d_model) to logits (B,T,vocab_size)."""
    linear = linear_forward(x, w_lm)
    biased = bias_add_forward(linear["y"], b_lm)

    return {
        "logits": biased["y"],
        "cache": {
            "x": x,
            "w_lm": w_lm,
        },
    }

# Step 145 - full_model_forward
def full_model_forward(x_ids, model_params):
    """Run embeddings, all blocks, final LN, and LM head; return logits and caches."""

    B, T = x_ids.shape

    # Token embeddings
    tok_out, tok_cache = token_embedding_forward(
        x_ids,
        model_params["tok_emb"],
    )

    # Positional embeddings
    pos_out = slice_positional_embedding(
        model_params["pos_emb"],
        T,
    )

    # Token + positional embeddings
    emb_out = add_token_and_positional_embeddings(
        tok_out,
        pos_out,
    )

    # Transformer blocks
    hidden, block_caches = forward_through_all_blocks(
        emb_out,
        model_params["blocks"],
    )

    # Final LayerNorm
    ln_f_out, ln_f_cache = final_layernorm_forward(
        hidden,
        model_params["ln_f"]["gamma"],
        model_params["ln_f"]["beta"],
    )

    # LM head
    lm_head_out = lm_head_linear_forward(
        ln_f_out,
        model_params["lm_head"]["w_lm"],
        model_params["lm_head"]["b_lm"],
    )

    caches = {
        "emb": {
            "token": tok_cache,
            "pos": pos_out,
            "sum": emb_out,
        },
        "blocks": block_caches,
        "ln_f": ln_f_cache,
        "lm_head": lm_head_out["cache"],
    }

    return lm_head_out["logits"], caches

# Step 146 - full_model_backward
def full_model_backward(d_logits, caches, model_params):
    """Backpropagate gradients from logits to all model parameters."""

    # ------------------------------------------------------------
    # 1. LM head backward
    # ------------------------------------------------------------
    lm_cache = caches["lm_head"]

    x_lm = lm_cache["x"]
    w_lm = lm_cache["w_lm"]

    d_ln_f = d_logits @ w_lm.T

    d_w_lm = np.einsum(
        "btd,btv->dv",
        x_lm,
        d_logits,
    )

    d_b_lm = np.sum(d_logits, axis=(0, 1))

    # ------------------------------------------------------------
    # 2. Final LayerNorm backward
    # ------------------------------------------------------------
    ln_cache = caches["ln_f"]

    x_hat = ln_cache["x_hat"]
    var = ln_cache["var"]
    gamma = ln_cache["gamma"]

    d_gamma_f = np.sum(
        d_ln_f * x_hat,
        axis=(0, 1),
    )

    d_beta_f = np.sum(
        d_ln_f,
        axis=(0, 1),
    )

    # LayerNorm backward over the last dimension
    d_xhat = d_ln_f * gamma

    d_hidden = (
        d_xhat
        - np.mean(d_xhat, axis=-1, keepdims=True)
        - x_hat
        * np.mean(
            d_xhat * x_hat,
            axis=-1,
            keepdims=True,
        )
    ) / np.sqrt(var + 1e-5)

    # ------------------------------------------------------------
    # 3. Transformer blocks backward
    # ------------------------------------------------------------
    d_emb, block_grads = backward_through_all_blocks(
        d_hidden,
        caches["blocks"],
        model_params["blocks"],
    )

    # ------------------------------------------------------------
    # 4. Embedding sum backward
    # ------------------------------------------------------------
    d_tok = d_emb
    d_pos = d_emb

    # ------------------------------------------------------------
    # 5. Token embedding backward
    # ------------------------------------------------------------
    tok_cache = caches["emb"]["tok_cache"]
    token_ids = tok_cache["token_ids"]

    d_tok_emb = np.zeros_like(model_params["tok_emb"])

    np.add.at(
        d_tok_emb,
        token_ids,
        d_tok,
    )

    # ------------------------------------------------------------
    # 6. Positional embedding backward
    # ------------------------------------------------------------
    seq_len = caches["emb"]["seq_len"]

    d_pos_emb = np.zeros_like(model_params["pos_emb"])

    d_pos_emb[:seq_len] = np.sum(
        d_pos,
        axis=0,
    )

    # ------------------------------------------------------------
    # 7. Assemble gradient tree
    # ------------------------------------------------------------
    grads = {
        "tok_emb": d_tok_emb,
        "pos_emb": d_pos_emb,
        "blocks": block_grads,
        "ln_f": {
            "gamma": d_gamma_f,
            "beta": d_beta_f,
        },
        "lm_head": {
            "w_lm": d_w_lm,
            "b_lm": d_b_lm,
        },
    }

    return grads

# Step 147 - initialize_adam_moments
def initialize_adam_moments(model_params):
    """Allocate zeroed Adam first- and second-moment buffers matching model_params."""

    def build_moments(tree):
        if isinstance(tree, dict):
            m = {}
            v = {}

            for key, value in tree.items():
                m[key], v[key] = build_moments(value)

            return m, v

        if isinstance(tree, list):
            m = []
            v = []

            for value in tree:
                m_value, v_value = build_moments(value)
                m.append(m_value)
                v.append(v_value)

            return m, v

        if isinstance(tree, np.ndarray):
            return (
                np.zeros_like(tree),
                np.zeros_like(tree),
            )

        # Non-array leaves are not parameter buffers.
        return tree, tree

    return build_moments(model_params)

# Step 148 - initialize_adam_step_counter
def initialize_adam_step_counter():
    """Return the initial Adam step counter t."""
    return 0

# Step 149 - adam_increment_step
def adam_increment_step(t):
    """Return t + 1 so Adam bias correction sees a positive step."""
    return t + 1

# Step 150 - adam_update_first_moment
import numpy as np

def adam_update_first_moment(m, grad, beta1):
    """Return the updated Adam first-moment estimate."""
    return beta1 * m + (1 - beta1) * grad

# Step 151 - adam_update_second_moment
def adam_update_second_moment(v_prev, grad, beta2):
    """Update Adam's second-moment estimate v using squared gradient EMA."""
    return beta2 * v_prev + (1 - beta2) * (grad ** 2)

# Step 152 - adam_bias_correction
def adam_bias_correction(m, v, beta1, beta2, t):
    """Return bias-corrected (m_hat, v_hat) for Adam at step t."""
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    return m_hat, v_hat

# Step 153 - adam_parameter_update
import numpy as np

def adam_parameter_update(param, m_hat, v_hat, lr, eps):
    """Apply the Adam update: param - lr * m_hat / (sqrt(v_hat) + eps)."""
    return param - lr * m_hat / (np.sqrt(v_hat) + eps)

# Step 154 - wire_full_training_loop
def wire_full_training_loop(
    params,
    train_ids,
    val_ids,
    block_size,
    batch_size,
    n_steps,
    lr,
    betas,
    eps,
):
    """Run the full GPT training loop for n_steps and return (updated_params, history)."""

    beta1, beta2 = betas

    rng = np.random.default_rng(0)

    m, v = initialize_adam_moments(params)
    t = initialize_adam_step_counter()

    history = []

    def update_tree(params_tree, grads_tree, m_tree, v_tree, t):

        if isinstance(params_tree, dict):
            for key in params_tree:
                params_tree[key], m_tree[key], v_tree[key] = update_tree(
                    params_tree[key],
                    grads_tree[key],
                    m_tree[key],
                    v_tree[key],
                    t,
                )

            return params_tree, m_tree, v_tree

        if isinstance(params_tree, list):
            for i in range(len(params_tree)):
                params_tree[i], m_tree[i], v_tree[i] = update_tree(
                    params_tree[i],
                    grads_tree[i],
                    m_tree[i],
                    v_tree[i],
                    t,
                )

            return params_tree, m_tree, v_tree

        if isinstance(params_tree, np.ndarray):
            m_new = adam_update_first_moment(
                m_tree,
                grads_tree,
                beta1,
            )

            v_new = adam_update_second_moment(
                v_tree,
                grads_tree,
                beta2,
            )

            m_hat, v_hat = adam_bias_correction(
                m_new,
                v_new,
                beta1,
                beta2,
                t,
            )

            param_new = adam_parameter_update(
                params_tree,
                m_hat,
                v_hat,
                lr,
                eps,
            )

            return param_new, m_new, v_new

        return params_tree, m_tree, v_tree

    for step in range(n_steps):

        # 1. Batch
        X, Y = get_batch(
            train_ids,
            block_size,
            batch_size,
            rng,
        )

        # 2. Forward
        logits, caches = full_model_forward(X, params)

        B, T, V = logits.shape

        # --------------------------------------------------
        # Fix cache-name mismatch between steps 145 and 146
        # --------------------------------------------------
        if "tok_cache" not in caches["emb"]:
            caches["emb"]["tok_cache"] = caches["emb"]["token"]

        if "seq_len" not in caches["emb"]:
            caches["emb"]["seq_len"] = T

        # 3. Flatten for cross entropy
        flat_logits = logits.reshape(B * T, V)
        flat_targets = Y.reshape(B * T)

        probs = stable_softmax_2d_rowwise(flat_logits)

        loss = cross_entropy_loss(
            probs,
            flat_targets,
        )

        # 4. Loss gradient
        d_logits = softmax_cross_entropy_backward(
            probs,
            flat_targets,
        )

        d_logits = d_logits.reshape(B, T, V)

        # 5. Backward
        grads = full_model_backward(
            d_logits,
            caches,
            params,
        )

        # 6. Adam
        t = adam_increment_step(t)

        params, m, v = update_tree(
            params,
            grads,
            m,
            v,
            t,
        )

        # 7. History
        history.append({
            "step": step,
            "train_loss": float(loss),
        })

    return params, history

# Step 155 - logging_and_validation_loss
def logging_and_validation_loss(
    params,
    val_ids,
    block_size,
    batch_size,
    n_eval_batches,
):
    """Estimate validation cross-entropy loss by averaging over several batches."""

    rng = np.random.default_rng(0)

    losses = []

    for _ in range(n_eval_batches):
        # 1. Sample validation batch
        X, Y = get_batch(
            val_ids,
            block_size,
            batch_size,
            rng,
        )

        # 2. Forward pass
        logits, _ = full_model_forward(X, params)

        B, T, V = logits.shape

        # 3. Flatten token positions
        flat_logits = logits.reshape(B * T, V)
        flat_targets = Y.reshape(B * T)

        # 4. Convert logits to probabilities
        probs = stable_softmax_2d_rowwise(flat_logits)

        # 5. Cross-entropy loss
        loss = cross_entropy_loss(
            probs,
            flat_targets,
        )

        losses.append(float(loss))

    return float(np.mean(losses))

# Step 156 - encode_prompt (not yet solved)
# TODO: implement

# Step 157 - crop_context_to_block_size (not yet solved)
# TODO: implement

# Step 158 - forward_to_get_logits (not yet solved)
# TODO: implement

# Step 159 - take_last_position_logits (not yet solved)
# TODO: implement

# Step 160 - apply_temperature (not yet solved)
# TODO: implement

# Step 161 - top_k_filter (not yet solved)
# TODO: implement

# Step 162 - softmax_to_probs (not yet solved)
# TODO: implement

# Step 163 - sample_one_token (not yet solved)
# TODO: implement

# Step 164 - append_token_to_sequence (not yet solved)
# TODO: implement

# Step 165 - generation_loop_for_n_steps (not yet solved)
# TODO: implement

# Step 166 - decode_final_sequence (not yet solved)
# TODO: implement

