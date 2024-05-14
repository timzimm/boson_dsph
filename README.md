## Summary
Code to reproduce the boson DM mass limit presented in:

TODO: Add bibtex entry

## What's in the Box
```bash
.
├── pyproject.toml
├── setup.cfg
├── README.md
├── configs                                 # config files r pipeline run
│   ├── leoii_15.yaml                           # 1.5 TByte on disk
│   ├── leoii_19.yaml                           # 2.3 TByte on disk
│   ├── leoii_20.yaml                           # 2.5 TByte on disk
│   ├── leoii_21.yaml                           # 2.9 TByte on disk
│   ├── leoii_22.yaml                           # 3.1 TByte on disk
│   ├── leoii_23.yaml                           # 3.4 TByte on disk
│   ├── leoii_24.yaml                           # 3.8 TByte on disk
├── data
│   ├── coreNFWtides_parameters             # coreNFWtides posterior samples 
│   │   ├── output_M200c200_chain_LeoII.txt
├── notebook
│   ├── mnras.mplstyle
│   ├── plots_prl.ipynb
└── src
    ├── gsph2wsph                           # run script for pipelin
    ├── pipeline_for_single_draw_mpi.py     # the pipeline
    ├── stats.py                            # mmd fuse implementation
    └── test_equality.py                    # driver for hypothesis test
```

## How to Install

```bash
$ git clone git@github.com:timzimm/boson_dsph.git
$ pip install -e .
```

## How to Run the Pipeline

Familiarise yourself with the runtime parameters of the pipeline by looking at
the provided yaml files. 
By default all models, i.e. density, potential, eigenstate library and
wavefunction parameters, are serialised, compressed and written to their
respective folder in `cache`. For this to work, make sure to:
```bash
$ mkdir cache_dir\{density,potential,eigenstate_library,wavefunction_params,mmd_fuse}
```
before exection.

To run the wave function reconstruction over all posterior samples, execute:
```bash
$ cd src 
$ ./gsph2wsph <NPROC> <HOSTFILE> ../configs/<YAML_CONFIG_FILE>
```

To run the hypothesis test, execute:
```bash
$ cd src 
$ JAX_ENABLE_X64=True mpirun -x JAX_ENABLE_X64 -n <NPROC> python test_equality.py ../configs/<YAML_CONFIG_FILE>
```

**NOTE:** Be aware that depending on the runtime (hyper)parameters, most notably
`NPROC` and `m22`, peak memory consumption can be O(TB).

With the pipeline data products available, you should be able to execute the
`plots_prl.pynb` notebook.
