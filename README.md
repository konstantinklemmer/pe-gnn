# PyTorch implementation of PE-GNN

![Architecture of a naive GCN versus that of PE-GNN, enhanced with a positional encoder.](https://raw.githubusercontent.com/konstantinklemmer/pe-gnn/main/images/pegnn.png)

*(Architecture of a naive GCN versus that of PE-GNN, enhanced with a positional encoder.)*

This is the official repository for the *AISTATS 2023* paper [Positional Encoder Graph Neural Networks for Geographic Data](https://arxiv.org/abs/2111.10144) (Konstantin Klemmer, Nathan Safir, Daniel B. Neill).


## Structure

The source code for *PE-GNN* (using `PyTorch`) can be found in the `src` folder. Its built on *[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)* (ICLR-W, 2019) and *[Space2Vec](https://github.com/gengchenmai/space2vec)* (ICLR, 2020).

We also provide an interactive example notebook to test *PE-GNN* via Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/konstantinklemmer/pe-gnn/blob/master/example.ipynb)
	
## Citation 

If you want to cite our work, you can use the following reference:

```
@InProceedings{klemmer2023positional,
  title = {Positional Encoder Graph Neural Networks for Geographic Data},
  author = {Konstantin Klemmer and Nathan Safir and Daniel B Neill},
  booktitle = {Proceedings of The 26th International Conference on Artificial Intelligence and Statistics},
  year = {2023},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url = {https://arxiv.org/abs/2111.10144},
}
```
