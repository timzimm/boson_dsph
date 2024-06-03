### Summary & Attribution
Code to reproduce the boson DM mass limit presented in (arXiv:2405.20374)[https://arxiv.org/pdf/2405.20374]:
We kindly ask you to cite this work as:
```
@article{Zimmermann:boson-dsph,
    author = "Zimmermann, Tim and Alvey, James and Marsh, David J. E. and Fairbairn, Malcolm and Read, Justin I.",
    title = "{Dwarf galaxies imply dark matter is heavier than $\mathbf{2.2 \times 10^{-21}} \, \mathbf{eV}$}",
    eprint = "2405.20374",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "5",
    year = "2024"
}
```

The analysis relies on our reconstruction tool 
[jaxsp](https://github.com/james-alvey-42/jaxsp).

### What's in the Box
```bash
.
├── pyproject.toml
├── setup.cfg
├── README.md
├── configs                                 # config files for pipeline run
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
│   ├── hypothesis_test                     # pipeline output of hypothesis test
│   │   ├── leoii_15.npz
│   │   ├── leoii_19.npz
│   │   ├── leoii_20.npz
│   │   ├── leoii_21.npz
│   │   ├── leoii_22.npz
│   │   ├── leoii_23.npz
│   │   ├── leoii_24.npz
│   ├── density
│   │   ├── density_002.3.npz 
│   │   ├── density_0023.0.npz 
├── notebook
│   ├── plots_prl.ipynb
└── boson_dsph
    ├── gsph2wsph                           # run script for pipeline
    ├── pipeline_for_single_draw_mpi.py     # the pipeline
    ├── stats.py                            # mmd fuse implementation
    └── test_equality.py                    # driver for hypothesis test
    └── density.py                          # generates ../data/density/*.npz
```

### How to Install

```bash
$ git clone git@github.com:timzimm/boson_dsph.git
$ cd boson_dsph
$ pip install -e .
```

### How to Reproduce the Plots
Given the total size of the pipeline data products, we only ship the MMDFuse
distribution of the hypothesis test. This is sufficient to execute
the `plots_prl.pynb` notebook. If you want access to the reconstructed wave
functions, you will have to run the pipeline yourself. See below for
instructions how to do this.

### How to Run the Pipeline
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
$ cd boson_dsph 
$ ./gsph2wsph <NPROC> <HOSTFILE> ../configs/<YAML_CONFIG_FILE>
```

To run the hypothesis test, execute:
```bash
$ cd boson_dsph 
$ JAX_ENABLE_X64=True mpirun -x JAX_ENABLE_X64 -n <NPROC> python test_equality.py ../configs/<YAML_CONFIG_FILE>
```

To generate the ensemble of spherically averaged wave densities (Fig. 1),
execute:
```bash
$ cd boson_dsph 
$ JAX_ENABLE_X64=True mpirun -x JAX_ENABLE_X64 -n <NPROC> python density.py ../configs/<YAML_CONFIG_FILE>
```

**NOTE:** Be aware that depending on the runtime (hyper)parameters, most notably
`NPROC` and `m22`, peak memory consumption can be O(TB).

### Contributors
Tim Zimmermann  
James Alvey  
David J.E. Marsh  
Malcolm Fairbarn  
Justin Read  

### Acknowledgement
![eu](https://github.com/timzimm/boson_dsph/blob/94c8984fca269edb8b5a47ca43b346f07e80e1cc/images/eu_acknowledgement_compsci_3.png#gh-light-mode-only)
![eu](https://github.com/timzimm/boson_dsph/blob/94c8984fca269edb8b5a47ca43b346f07e80e1cc/images/eu_acknowledgement_compsci_3_white.png#gh-dark-mode-only)
