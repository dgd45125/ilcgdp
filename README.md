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

## Prepare and Save the Training and Development Set Features
From the ilcgdp directory, e.g.: 
```
python main.py --prepare_features --analysis_dir ../ptb_ud --train_file ../ptb_dependencies/universal_dependencies/train/ptb_train.conllu --dev_file  ../ptb_dependencies/universal_dependencies/dev/ptb_dev.conllu
```

## Train a Model
Takes `-- num_epochs` `-- batch_size` 


From the ilcgdp directory, e.g.:
```
python main.py  --train_model --num_epochs 50 --analysis_dir ../ptb_ud --train_file ../ptb_dependencies/universal_dependencies/train/ptb_train.conllu --dev_file  ../ptb_dependencies/universal_dependencies/dev/ptb_dev.conllu --show_cuda_device_details
``
