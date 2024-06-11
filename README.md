# RAGCL

Source code for RAGCL in paper: 

**Propagation Tree is not Deep: Adaptive Graph Contrastive Learning Approach for Rumor Detection**

## Run

You need to compute node centrality with ```centrality.py``` to process your own data before running the code.

The code can be run in the following ways:

```shell script
nohup python main\(sup\).py --gpu 0 &
```

Dataset is available at [https://pan.baidu.com/s/1Kl5IQjU3a_pdt90YmNNVsQ?pwd=qqul](https://pan.baidu.com/s/1Kl5IQjU3a_pdt90YmNNVsQ?pwd=qqul)

## Dependencies

- [pytorch](https://pytorch.org/) == 1.12.0

- [torch-geometric](https://github.com/pyg-team/pytorch_geometric) == 2.1.0

- [gensim](https://radimrehurek.com/gensim/index.html) == 4.0.1

## Citation

If this work is helpful, please kindly cite as:

```
@inproceedings{cui2024propagation,
  title={Propagation Tree Is Not Deep: Adaptive Graph Contrastive Learning Approach for Rumor Detection},
  author={Cui, Chaoqun and Jia, Caiyan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={1},
  pages={73--81},
  year={2024}
}
```
