# ilcgdp
Incremental Left-Corner Generative Dependency Parser v3.0.

## Notes
This is the third iteration of the dependency parser presented in [Chapter 2 of Donnie Dunagan's dissertation](https://www.proquest.com/docview/3253537008/FEE81C1CFE1B4630PQ/1?accountid=11752&sourcetype=Dissertations%20&%20Theses). It differs from the implementation presented there in that it no longer uses POS tags as features.

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
python main.py  --prepare_features --analysis_dir ../en_gum-ud --train_file ../UD_data/UD_English-GUM-master/en_gum-ud-train.conllu --dev_file  ../UD_data/UD_English-GUM-master/en_gum-ud-dev.conllu
```
This will create the analysis directory if it doesn't already exist.

## Train a Model
Accepts `--num_epochs` `--batch_size` `--word_embedding_size` `--dep_embedding_size` `--node_size` `--spine_size` and `--hidden_size` arguments. If not provided, reasonable defaults are used. `--save_all_models` save each epoch checkpoint rather than just models that improve one of the losses or labeled transition prediction accuracy.

From the ilcgdp directory, e.g.:
```
python main.py  --train_model --num_epochs 50 --analysis_dir ../en_gum-ud --train_file ../UD_data/UD_English-GUM-master/en_gum-ud-train.conllu --dev_file  ../UD_data/UD_English-GUM-master/en_gum-ud-dev.conllu --show_cuda_device_details
```
Models and a training log are stored in `analysis_dir/models`.
