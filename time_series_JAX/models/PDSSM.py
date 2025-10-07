"""
An implementation of a PD-SSM block

Attributes of the `PDSSM` class:
- `linear_encoder`: The linear encoder applied to the input time series data.
- `blocks`: A list of `PDSSMBlock` instances, each containing the LRU layer, normalization, GLU, and dropout.
- `linear_layer`: The final linear layer that outputs the model predictions.
- `classification`: A boolean indicating whether the model is used for classification tasks.
- `output_step`: For regression tasks, specifies how many steps to skip before outputting a prediction.

The module also includes the following main classes and functions:
- `PDSSMLayer`: The main module, containing an implementation of PD-SSM that uses jax.lax.associative_scan and a custom gradient computation.
- `PDSSMBlock`: A wrapper around PDSSMLayer that applies LayerNorm to the SSM module's output.

- `binary_operator_efficient`: A helper function used in the associative scan operation within `PDSSMLayer` to process complex valued column one-hot elements.
- `binary_operator_efficient_grad`: A helper function used in the associative scan implementation of the gradient calculation.
- `custom_bwd`: Computes the gradients.
"""

from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import math

parallel_scan = jax.lax.associative_scan

# Parallel scan operations

@jax.vmap
def binary_operator_efficient(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence"""
    # All arrays are of shape (d_hidden,)
    P_i, D_i, b_i = q_i
    P_j, D_j, b_j = q_j

    P_new = jnp.take(P_j, indices=P_i)
    D_new = jnp.take(D_j, indices=P_i) * D_i
    b_new = jnp.zeros_like(b_i).at[P_j].add(D_j * b_i) + b_j 

    return P_new, D_new, b_new

@jax.vmap
def binary_operator_efficient_grad(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence"""
    # All arrays are of shape (d_hidden,)
    P_i, D_i, b_i = q_i
    P_j, D_j, b_j = q_j

    P_new = jnp.take(P_i, indices=P_j)

    D_new = jnp.take(D_i, indices=P_j) * D_j

    b_new = b_i[P_j] * D_j + b_j

    return P_new, D_new, b_new

def indexing_parallel_scan(P, D, b):
    _, _, y =  parallel_scan(binary_operator_efficient, (P, D, b), axis=0)
    return y

def indexing_grad_parallel_scan(P, D, b):
    _, _, y =  parallel_scan(binary_operator_efficient_grad, (P, D, b), axis=0)
    return y

"""
3: Custom implementation of indexing parallel scans
"""

@jax.custom_vjp
def custom_function(P, D, b):
    return forward_pass(P, D, b)

def forward_pass(P, D, b):
    return indexing_parallel_scan(jnp.argmax(P, axis=-2), D, b)

def custom_fwd(P, D, b):
    y = forward_pass(P, D, b)
    return y, (y, jnp.argmax(P, axis=-2), D)  # Save (y, P, D) for backward

def batch_index(b_row, p_row):
    return b_row[p_row]

def custom_bwd(residuals, grad_xs):

    y, P, D = residuals
    T, N = D.shape
    
    y_padded = jnp.concatenate([jnp.zeros((1, N)) , y[:-1,:]], axis=0)

    # 1: The parallel scan that computes dL/dx_k given dL/dy_k (which is the grad_xs array)
    # PSCAN recurrence
    pscan_inputs = jnp.flip(grad_xs, axis=0)
    pscan_Ps = jnp.concatenate([jnp.arange(N)[None, :], jnp.flip(P[1:], axis=0)], axis=0)
    pscan_Ds = jnp.concatenate([jnp.ones((1, N)), jnp.flip(D[1:], axis=0)], axis=0)
    grad_ys = jnp.flip(indexing_grad_parallel_scan(pscan_Ps, pscan_Ds, pscan_inputs), axis=0)
    
    # 2: Computing the total derivatives dL/dP_k, dL/dD_k, dL/db_k

    # Jacobian of loss w.r.t. P matrices
    grad_Ps = jnp.einsum('tj,tl->tjl', grad_ys, D*y_padded)

    # Jacobian of loss w.r.t. D matrices
    grad_Ds = jax.vmap(batch_index)(grad_ys, P) * y_padded

    # Jacobian of loss w.r.t input
    grad_bs = grad_ys 

    return grad_Ps.real, grad_Ds, grad_bs

custom_function.defvjp(custom_fwd, custom_bwd)



class PDSSMLayer(eqx.Module):
    B_re: jnp.ndarray
    B_im: jnp.ndarray
    C_re: jnp.ndarray
    C_im: jnp.ndarray
    D: jnp.ndarray

    P_dict: jnp.ndarray
    P_selector: jnp.ndarray

    W1_mag: jnp.ndarray
    W2_mag: jnp.ndarray
    b1_mag: jnp.ndarray
    b2_mag: jnp.ndarray

    W1_pha: jnp.ndarray
    W2_pha: jnp.ndarray
    b1_pha: jnp.ndarray
    b2_pha: jnp.ndarray

    def __init__(self, N, H, K=6, *, key):
        (B_re_key, B_im_key, 
         C_re_key, C_im_key, D_key, 
         P_dict_key, P_selector_key, 
         W1_mag_key, W2_mag_key,
         W1_pha_key, W2_pha_key,
         b1_mag_key, b2_mag_key,
         b1_pha_key, b2_pha_key) = jr.split(key, 15)

        # N: state dimension, H: model dimension

        # Normal initialized Input/Output projection matrices
        self.B_re = jr.normal(B_re_key, shape=(N, H)) / jnp.sqrt(2 * H)
        self.B_im = jr.normal(B_im_key, shape=(N, H)) / jnp.sqrt(2 * H)
        self.C_re = jr.normal(C_re_key, shape=(H, N)) / jnp.sqrt(N)
        self.C_im = jr.normal(C_im_key, shape=(H, N)) / jnp.sqrt(N)
        self.D = jr.normal(D_key, shape=(H,))

        # Permutation matrices
        self.P_dict = jr.normal(P_dict_key, shape=(K, N, N)) / jnp.sqrt(N)
        self.P_selector = jr.normal(P_selector_key, shape=(K, H))

        # Diagonal matrix generators
        self.W1_mag = jr.normal(W1_mag_key, shape=(N, H)) / jnp.sqrt(H)
        self.W2_mag = jr.normal(W2_mag_key, shape=(N, N)) / jnp.sqrt(N)
        self.b1_mag = jr.normal(b1_mag_key, shape=(N)) / jnp.sqrt(N)
        self.b2_mag = jr.normal(b2_mag_key, shape=(N)) / jnp.sqrt(N)

        self.W1_pha = jr.normal(W1_pha_key, shape=(N, H)) / jnp.sqrt(H)
        self.W2_pha = jr.normal(W2_pha_key, shape=(N, N)) / jnp.sqrt(N)
        self.b1_pha = jr.normal(b1_pha_key, shape=(N)) / jnp.sqrt(N)
        self.b2_pha = jr.normal(b2_pha_key, shape=(N)) / jnp.sqrt(N) - 2.5


    def __call__(self, x):

        B = self.B_re + 1j * self.B_im
        C = self.C_re + 1j * self.C_im

        # Generating the diagonal matrices
        magnitudes = jax.vmap(lambda u: jax.nn.sigmoid(self.W1_mag @ u + self.b1_mag))(x)
        phases = jax.vmap(lambda u: jax.nn.sigmoid(self.W1_pha @ u + self.b1_pha)*2*math.pi)(x)
        phases_complex = jax.vmap(lambda u: jnp.exp(1j * u))(phases)
        diagonal_matrices = magnitudes * phases_complex

        # Generating the column one-hot (P) matrices
        selection_weights = jax.vmap(lambda u: jax.nn.softmax(self.P_selector @ u, axis=-1))(x)
        permutation_matrices = jax.vmap(lambda u: jnp.einsum('kmn,k->mn', self.P_dict, u))(selection_weights)
        permutation_matrices_column_softmax = jax.vmap(lambda u: jax.nn.softmax(u, axis=0))(permutation_matrices)

        # Projecting the input
        Bu_elements = jax.vmap(lambda u: B @ u)(x)

        # Parallel scan
        hidden_states = custom_function(permutation_matrices_column_softmax, diagonal_matrices, Bu_elements)

        # Readout and residual
        y = jax.vmap(lambda h, x: (C @ h).real + self.D * x)(hidden_states, x)

        return y


class PDSSMBlock(eqx.Module):

    norm: eqx.nn.LayerNorm
    pdssm: PDSSMLayer

    def __init__(self, N, H, K=6, drop_rate=0.1, *, key):
        pdssmkey, _ = jr.split(key, 2)
        self.norm = eqx.nn.LayerNorm(shape=H)
        self.pdssm = PDSSMLayer(N, H, K, key=pdssmkey)

    def __call__(self, x, state, *, key):
        x = jax.vmap(self.norm)(self.pdssm(x))
        return x, state


class PDSSM(eqx.Module):
    linear_encoder: eqx.nn.Linear
    blocks: List[PDSSMBlock]
    linear_layer: eqx.nn.Linear
    classification: bool
    output_step: int
    stateful: bool = True
    nondeterministic: bool = True
    lip2: bool = False

    def __init__(
        self,
        num_blocks,
        data_dim,
        N,
        H,
        output_dim,
        classification,
        output_step,
        drop_rate=0.1,
        K=6,
        *,
        key
    ):
        linear_encoder_key, *block_keys, linear_layer_key = jr.split(
            key, num_blocks + 2
        )
        self.linear_encoder = eqx.nn.Linear(data_dim, H, key=linear_encoder_key)
        self.blocks = [
            PDSSMBlock(N, H, K, drop_rate, key=key)
            for key in block_keys
        ]
        self.linear_layer = eqx.nn.Linear(H, output_dim, key=linear_layer_key)
        self.classification = classification
        self.output_step = output_step

    def __call__(self, x, state, key):
        dropkeys = jr.split(key, len(self.blocks))
        x = jax.vmap(self.linear_encoder)(x)
        for block, key in zip(self.blocks, dropkeys):
            x, state = block(x, state, key=key)
        if self.classification:
            x = jnp.mean(x, axis=0)
            x = jax.nn.softmax(self.linear_layer(x), axis=0)
        else:
            x = x[self.output_step - 1 :: self.output_step]
            x = jax.nn.tanh(jax.vmap(self.linear_layer)(x))
        return x, state
