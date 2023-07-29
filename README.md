# RAGCL

Source code for RAGCL in paper: 

**Propagation Tree is not Deep: Adaptive Graph Contrastive Learning Approach for Rumor Detection**

## Run

You need to compute node centrality with ```centrality.py``` before running the code.

The code can be run in the following ways:

```shell script
nohup python main\(sup\).py --gpu 0 &
```

## Dependencies

- [pytorch](https://pytorch.org/) == 1.12.0

- [torch-geometric](https://github.com/pyg-team/pytorch_geometric) == 2.1.0

- [gensim](https://radimrehurek.com/gensim/index.html) == 4.0.1