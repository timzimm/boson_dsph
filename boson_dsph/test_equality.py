#!/usr/bin/env python
import argparse
import os

from mpi4py import MPI
import numpy as np
import jax.numpy as jnp
import jax
import ruamel.yaml

import jaxsp as jsp
from stats import mmdfuse


def load_parameters_from_config(path_to_file):
    yaml = ruamel.yaml.YAML(typ="safe", pure=True)
    with open(path_to_file, "r") as file:
        parameters = yaml.load(file)

    return parameters


eval_lib = jax.vmap(
    jax.vmap(jsp.eval_radial_eigenmode, in_axes=(None, 0)), in_axes=(0, None)
)
rho_psi = jax.vmap(jsp.rho_psi, in_axes=(0, None, None))
rho = jax.vmap(jsp.rho, in_axes=(0, None))


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


parser = argparse.ArgumentParser()
parser.add_argument("yaml_file", type=str, help="Path to YAML parameter file")
args = parser.parse_args()

params = load_parameters_from_config(args.yaml_file)
cache_dir = params["general"]["cache"]

m = params["uldm"]["m22"]
u = jsp.set_schroedinger_units(m)

weights = jsp.glw256
xi = jsp.glx256

names = [
    name
    for name in os.listdir(f"{cache_dir}/wavefunction_params")
    if not name.startswith(".")
]
N_samples = len(names)
batches = np.array_split(jnp.arange(N_samples), size)
batch_sizes = np.array([batch.shape[0] * weights.shape[0] for batch in batches])
disp = np.cumsum(batch_sizes) - batch_sizes[0]
sample_idx = batches[rank]


sqrt_jac_rhos_log_rj = None
sqrt_jac_rhos_psi_log_rj = None
if rank == 0:
    sqrt_jac_rhos_log_rj = np.empty((N_samples, weights.shape[0]))
    sqrt_jac_rhos_psi_log_rj = np.empty((N_samples, weights.shape[0]))

sqrt_jac_rhos_log_rj_loc = []
sqrt_jac_rhos_psi_log_rj_loc = []
for i, idx in enumerate(sample_idx):
    print(f"Rank {rank}: {i+1}/{len(sample_idx)}")
    wavefunction_params = jsp.load_model(f"{cache_dir}/wavefunction_params", names[idx])
    lib = jsp.load_model(
        f"{cache_dir}/eigenstate_library",
        wavefunction_params.eigenstate_library.item(),
    )
    density_params = jsp.load_model(
        f"{cache_dir}/density", wavefunction_params.density_params.item()
    )
    r_min = params["mmd_fuse"]["r_min"] * u.from_Kpc
    r_max = params["mmd_fuse"]["r_max"] * u.from_Kpc
    # r_max = jsp.enclosing_radius(params["mmd_fuse"]["r_max"], density_params)
    log_rj = jnp.log(r_min) + (jnp.log(r_max) - jnp.log(r_min)) * xi
    sqrt_jac = jnp.sqrt(
        4 * jnp.pi * (jnp.log(r_max) - jnp.log(r_min)) * jnp.exp(3 * log_rj)
    )
    sqrt_jac_rhos_log_rj_loc.append(sqrt_jac * rho(jnp.exp(log_rj), density_params))
    sqrt_jac_rhos_psi_log_rj_loc.append(
        sqrt_jac * rho_psi(jnp.exp(log_rj), wavefunction_params, lib)
    )
sqrt_jac_rhos_log_rj_loc = np.asarray(sqrt_jac_rhos_log_rj_loc)
sqrt_jac_rhos_psi_log_rj_loc = np.asarray(sqrt_jac_rhos_psi_log_rj_loc)

comm.Gatherv(
    sendbuf=sqrt_jac_rhos_log_rj_loc,
    recvbuf=(sqrt_jac_rhos_log_rj, batch_sizes, disp, MPI.DOUBLE),
    root=0,
)
comm.Gatherv(
    sendbuf=sqrt_jac_rhos_psi_log_rj_loc,
    recvbuf=(sqrt_jac_rhos_psi_log_rj, batch_sizes, disp, MPI.DOUBLE),
    root=0,
)

if rank == 0:
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    sqrt_jac_rhos_log_rj = jax.random.permutation(
        subkey, jnp.asarray(sqrt_jac_rhos_log_rj)
    )
    key, subkey = jax.random.split(key)
    sqrt_jac_rhos_psi_log_rj = jax.random.permutation(
        subkey, jnp.asarray(sqrt_jac_rhos_psi_log_rj)
    )
    mmds, mmd = mmdfuse(
        sqrt_jac_rhos_log_rj,
        sqrt_jac_rhos_psi_log_rj,
        weights,
        key,
    )

    jnp.savez(f"{cache_dir}/mmd_fuse/data", mmds=mmds, mmd=mmd)
