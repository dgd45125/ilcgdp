# ilcgdp
Incremental Left-Corner Generative Dependency Parser v3.0.

## Notes
This is the third iteration of the left-corner, generative, neural, locally-normalized, transition-based dependency parser presented in [Chapter 2 of Donnie Dunagan's dissertation](https://www.proquest.com/docview/3253537008/FEE81C1CFE1B4630PQ/1?accountid=11752&sourcetype=Dissertations%20&%20Theses). It differs from the implementation described there in that it no longer uses POS tags as features.

It is intended to work out of the box with either: 1) any Universal Dependencies dataset; or 2) dependency conversions of the Penn Treebank.

## Software Dependencies
- Cython
- PyTorch
- NumPy

## Cython Compilation
From the ilcgdp directory:
```
python setup.py build_ext --inplace
```
