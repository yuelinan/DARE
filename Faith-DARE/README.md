Official implementation of "Towards Reliable and Faithful Explanations: A Disentanglement-Augmented Approach for Selective Rationalization".

## Data download
- Spurious-Motif: this dataset can be generated via `spmotif_gen/spmotif.ipynb` in [DIR](https://github.com/Wuyxin/DIR-GNN/tree/main). 
- Open Graph Benchmark (OGBG): this dataset can be downloaded when running mi_dare.sh.


## How to run Faith-DARE?

To train Faith-DARE on OGBG dataset:

```python
# cd ogbg
sh mi_dare.sh
```

To train Faith-DARE on Spurious-Motif dataset:

```python
# cd spmotif
sh mi_dare.sh
```




