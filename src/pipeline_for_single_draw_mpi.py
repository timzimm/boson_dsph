#!/usr/bin/env python
import argparse

from mpi4py import MPI
import numpy as np
import jax.numpy as jnp
import jax
import ruamel.yaml

import jaxsp as jsp


def load_parameters_from_config(path_to_file):
    yaml = ruamel.yaml.YAML(typ="safe", pure=True)
    with open(path_to_file, "r") as file:
        parameters = yaml.load(file)

    return parameters


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

input_dir = params["density"]["gravsphere_chains"]
N_samples = params["density"]["samples"]
chain_data = np.empty((N_samples, 6))
if rank == 0:
    chain_data = np.loadtxt(input_dir)
    chain_data, c = np.unique(chain_data, axis=0, return_counts=True)
    chain_data[:, 0] *= u.from_Msun
    chain_data[:, 3] *= u.from_Kpc
    chain_data[:, 4] *= u.from_Kpc
    chain_data = chain_data[:N_samples]
comm.Bcast(chain_data, root=0)

sample_idx = np.array_split(jnp.arange(chain_data.shape[0]), size)[rank]
chain_data = chain_data[sample_idx]
N_samples = chain_data.shape[0]
for i, sample in enumerate(chain_data):
    print(f"Rank {rank}: {i}/{len(chain_data)}")
    i += 1
    if i % 16:
        jax.clear_caches()
    name = jsp.core_NFW_tides_params.compute_name(sample)
    density_params = jsp.load_or_compute_model(
        params["density"]["load_if_cached"],
        f"{cache_dir}/density",
        name,
        jsp.init_core_NFW_tides_params_from_sample,
        sample,
    )
    jsp.save_model(
        params["density"]["overwrite"], f"{cache_dir}/density", name, density_params
    )

    r999 = jsp.enclosing_radius(0.999, density_params)
    r_min = params["potential"]["r_min"] * u.from_Kpc * jnp.ones_like(r999)
    N_V = int(64 * (jnp.log10(jnp.max(r999)) - jnp.log10(jnp.min(r_min))))
    N_V = max(1 << (N_V - 1).bit_length(), 256)

    name = jsp.potential_params.compute_name(density_params, r_min, r999, N_V)
    potential_params = jsp.load_or_compute_model(
        params["potential"]["load_if_cached"],
        f"{cache_dir}/potential",
        name,
        jsp.init_potential_params,
        density_params,
        r_min,
        r999,
        N_V,
    )
    jsp.save_model(
        params["potential"]["overwrite"],
        f"{cache_dir}/potential",
        name,
        potential_params,
    )

    N_psi = params["eigenstate_library"]["N"]
    r_min = params["eigenstate_library"]["r_min"] * u.from_Kpc
    r99 = jsp.enclosing_radius(0.99, density_params)
    r995 = jsp.enclosing_radius(0.995, density_params)
    r999 = jsp.enclosing_radius(0.999, density_params)
    r9995 = jsp.enclosing_radius(0.9995, density_params)
    r9999 = jsp.enclosing_radius(0.9999, density_params)

    for r_max in [r99, r995, r999, r9995, r9999]:
        name = jsp.eigenstate_library.compute_name(
            potential_params, r_min, r_max, N_psi
        )

        library_params = jsp.load_or_compute_model(
            params["eigenstate_library"]["load_if_cached"],
            f"{cache_dir}/eigenstate_library",
            name,
            lambda *args: jsp.init_eigenstate_library(
                *args, batch_size=params["eigenstate_library"]["batch_size"]
            ),
            potential_params,
            r_min,
            r_max,
            N_psi,
        )
        if library_params:
            jsp.save_model(
                params["eigenstate_library"]["overwrite"],
                f"{cache_dir}/eigenstate_library",
                name,
                library_params,
            )
            break

    r_min = params["wave_function"]["r_min"] * u.from_Kpc
    tol = params["wave_function"]["tolerance"]
    name = jsp.wavefunction_params.compute_name(
        library_params, density_params, r_min, r99, tol
    )
    wavefunction_params = jsp.load_or_compute_model(
        params["wave_function"]["load_if_cached"],
        f"{cache_dir}/wavefunction_params",
        name,
        lambda *args: jsp.init_wavefunction_params_jensen_shannon(*args, verbose=False),
        library_params,
        density_params,
        r_min,
        r99,
        tol,
    )
    jsp.save_model(
        params["wave_function"]["overwrite"],
        f"{cache_dir}/wavefunction_params",
        name,
        wavefunction_params,
    )