# This file contains a modified version of antoninschrab/mmdfuse.

# Copyright (c) 2023 Antonin Schrab

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import jax
import jax.numpy as jnp
from functools import partial
from jax.scipy.special import logsumexp


@partial(jax.jit, static_argnums=(4, 5, 6))
def mmdfuse(
    X,
    Y,
    weights,
    key,
    lambda_multiplier=1,
    number_bandwidths=10,
    number_permutations=20000,
):
    def compute_mmds_of_all_permutations(bw, pairwise_matrix, V01, V10, V11):
        K = squared_exp_kernel(pairwise_matrix, bw)
        K = K.at[jnp.diag_indices(K.shape[0])].set(0)
        unscaled_std = jnp.sqrt(jnp.sum(K**2))
        # see Schrab et al. MMDAgg Appendix C
        mmd = (
            (
                jnp.sum(V10 * (K @ V10), 0) * (n - m + 1) * (n - 1) / (m * (m - 1))
                + jnp.sum(V01 * (K @ V01), 0) * (m - n + 1) / m
                + jnp.sum(V11 * (K @ V11), 0) * (n - 1) / m
            )
            / unscaled_std
            * jnp.sqrt(n * (n - 1))
        )
        return mmd

    if Y.shape[0] > X.shape[0]:
        X, Y = Y, X
    m = X.shape[0]
    n = Y.shape[0]
    assert lambda_multiplier > 0
    assert number_bandwidths > 1 and type(number_bandwidths) == int
    assert number_permutations > 0 and type(number_permutations) == int

    key, subkey = jax.random.split(key)
    B = number_permutations
    # (B, m+n): rows of permuted indices
    idx = jax.random.permutation(
        subkey,
        jnp.repeat(jnp.arange(m + n).reshape(1, -1), B + 1, axis=0),
        axis=1,
        independent=True,
    )
    # 11
    v11 = jnp.concatenate((jnp.ones(m), -jnp.ones(n)))  # (m+n, )
    V11i = jnp.tile(v11, (B + 1, 1))  # (B, m+n)
    V11 = jnp.take_along_axis(
        V11i, idx, axis=1
    )  # (B, m+n): permute the entries of the rows
    V11 = V11.at[B].set(v11)  # (B+1)th entry is the original MMD (no permutation)
    V11 = V11.transpose()  # (m+n, B+1)
    # 10
    v10 = jnp.concatenate((jnp.ones(m), jnp.zeros(n)))
    V10i = jnp.tile(v10, (B + 1, 1))
    V10 = jnp.take_along_axis(V10i, idx, axis=1)
    V10 = V10.at[B].set(v10)
    V10 = V10.transpose()
    # 01
    v01 = jnp.concatenate((jnp.zeros(m), -jnp.ones(n)))
    V01i = jnp.tile(v01, (B + 1, 1))
    V01 = jnp.take_along_axis(V01i, idx, axis=1)
    V01 = V01.at[B].set(v01)
    V01 = V01.transpose()

    # Compute all permuted MMD estimates
    N = number_bandwidths
    M = jnp.zeros((N, B + 1))
    Z = jnp.concatenate((X, Y))
    pairwise_matrix = pairwise_l2_distances(Z, Z, weights)

    distances = pairwise_matrix[jnp.triu_indices(pairwise_matrix.shape[0])]
    bandwidths = _compute_bandwidths(distances, number_bandwidths)

    M = jax.lax.scan(
        lambda c, bw: (
            None,
            compute_mmds_of_all_permutations(bw, pairwise_matrix, V01, V10, V11),
        ),
        None,
        bandwidths,
    )[1]

    # Compute permuted and original statistics
    all_statistics = logsumexp(lambda_multiplier * M, axis=0, b=1 / N)  # (B1+1,)
    original_statistic = all_statistics[-1]  # (1,)

    # p_val = jnp.mean(all_statistics >= original_statistic)

    return all_statistics, original_statistic


def _compute_bandwidths(distances, number_bandwidths):
    median = jnp.median(distances)
    distances = distances + (distances == 0) * median
    dd = jnp.sort(distances)
    lambda_min = dd[(jnp.floor(len(dd) * 0.05).astype(int))] / 2
    lambda_max = dd[(jnp.floor(len(dd) * 0.95).astype(int))] * 2
    bandwidths = jnp.linspace(lambda_min, lambda_max, number_bandwidths)
    return bandwidths


def squared_exp_kernel(pairwise_matrix, bandwidth):
    return jnp.exp(-(pairwise_matrix / bandwidth**2) / 2)


def pairwise_l2_distances(X, Y, weights):
    @partial(jax.vmap, in_axes=(None, 0))
    @partial(jax.vmap, in_axes=(0, None))
    def pairwise_l2(x, y):
        z = x - y
        return jnp.sqrt(jnp.dot(jnp.square(z), weights))

    output = pairwise_l2(X, Y)
    return output
