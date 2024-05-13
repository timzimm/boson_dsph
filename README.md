## Summary
TODO

## What's in the Box
```bash
.
├── .md
├── pyproject.toml
├── setup.cfg
├── README.md
├── configs                                     # config files for pipeline run
│   ├── leoii_15.yaml
│   ├── leoii_19.yaml
│   ├── leoii_20.yaml
│   ├── leoii_21.yaml
│   ├── leoii_22.yaml
│   ├── leoii_23.yaml
│   ├── leoii_24.yaml
├── data
│   ├── coreNFWtides_parameters                 # coreNFWtides posterior samples 
│   │   ├── output_M200c200_chain_LeoII.txt
├── notebook
│   ├── mnras.mplstyle
│   ├── plots_prl.ipynb
└── src
    ├── gsph2wsph                               # run script for pipelin
    ├── pipeline_for_single_draw_mpi.py         # the pipeline
    ├── stats.py                                # mmd fuse implementation
    └── test_equality.py                        # driver for hypothesis test
```
## How to Install
```bash
$ git clone git@github.com:timzimm/boson_dsph.git
$ pip install -e .
```

## How to Run the Pipeline

Familiarise yourself with the runtime parameters of the pipeline by looking at
the provided yaml files. The `cache` directory specified therein is used to
store all pipeline results. Make sure to:
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
`NPROC` and `m22`, memory consumption can be O(TB).
