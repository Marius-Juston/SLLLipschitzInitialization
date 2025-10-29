# SLL Lipschitz Initialization

## 

This repository is the code for the paper [1-Lipschitz Network Initialization for Certifiably Robust Classification Applications: A Decay Problem](https://arxiv.org/abs/2503.00240).

To cite please use,

```
@misc{juston20251lipschitznetworkinitializationcertifiably,
      title={1-Lipschitz Network Initialization for Certifiably Robust Classification Applications: A Decay Problem}, 
      author={Marius F. R. Juston and William R. Norris and Dustin Nottage and Ahmet Soylemezoglu},
      year={2025},
      eprint={2503.00240},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.00240}, 
}
```

## Paper figures

To generate the figures in the paper, run for generating the distribution of the intialization properties

```bash
python src/distribution/distribution.py
```

to run the training of the networks run

```bash
python src/train.py
```

and to generate the plots you would do,

```bash
python src/plot.py --runs_dir ../runs_lipschitz --outdir figs/tb_plots_lipschitz --compare_runs_dir runs --compare_outdir figs/tb_plots 
```