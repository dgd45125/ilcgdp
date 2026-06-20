# ilcgdp
Incremental Left-Corner Generative Dependency Parser v3.0.

## Notes
This is the third iteration of the dependency parser presented in [Chapter 2 of Donnie Dunagan's dissertation](https://openscholar.uga.edu/record/26943?v=pdf). It differs from the implementation presented there in that it no longer uses POS tags as features.

It is intended to work out of the box with either: 1) any Universal Dependencies dataset; or 2) dependency conversions of the Penn Treebank.

PyTorch will automatically make use of any single compatible accelerator.

## Software Dependencies
- Cython
- PyTorch
- NumPy

## Cython Compilation
From the `ilcgdp` directory:
```
python setup.py build_ext --inplace
```

## Prepare and Save the Training and Development Set Features
From the `ilcgdp` directory, e.g.,
```
python main.py  --prepare_features
                --analysis_dir ../en_gum-ud
                --train_file ../UD_data/UD_English-GUM-master/en_gum-ud-train.conllu
                --dev_file ../UD_data/UD_English-GUM-master/en_gum-ud-dev.conllu
```
This will create the analysis directory if it doesn't already exist.

## Train a Model
Accepts `--num_epochs`, `--batch_size`, `--word_embedding_size`, `--dep_embedding_size`, `--node_size`, `--spine_size`, and `--hidden_size` arguments for the neural network and training process. If not provided, reasonable defaults are used. `--save_all_models` saves every epoch checkpoint, rather than just models that improve one of the losses or labeled transition prediction accuracy.

From the `ilcgdp` directory, e.g.,
```
python main.py  --train_model
                --num_epochs 50
                --analysis_dir ../en_gum-ud
                --train_file ../UD_data/UD_English-GUM-master/en_gum-ud-train.conllu
                --dev_file ../UD_data/UD_English-GUM-master/en_gum-ud-dev.conllu
                --show_cuda_device_details
```
PyTorch models and a training log are stored in `analysis_dir/models`.

## Beam Parse a CoNLL File
Takes `--beam_size` and `--model_number` arguments to parse the CoNLL-formatted file specified by `--conll_file_to_parse`.
From the `ilcgdp` directory, e.g.:
```
python main.py  --beam_parse_conll
                --analysis_dir ../en_gum-ud
                --train_file ../UD_data/UD_English-GUM-master/en_gum-ud-train.conllu
                --conll_file_to_parse ../UD_data/UD_English-GUM-master/en_gum-ud-dev.conllu
                --beam_size 2
                --model_number 26
                --show_cuda_device_details
```
Parsing can also be split by sentence index using the `--start_sentence_index` and `--end_sentence_index` arguments, in order to make maximum use of computing (GPU) resources.
From the `ilcgdp` directory, e.g.,
```
python main.py  --beam_parse_conll
                --analysis_dir ../en_gum-ud
                --train_file ../UD_data/UD_English-GUM-master/en_gum-ud-train.conllu
                --conll_file_to_parse ../UD_data/UD_English-GUM-master/en_gum-ud-dev.conllu
                --start_sentence_index 0
                --end_sentence_index 425
                --beam_size 16
                --model_number 26
                --show_cuda_device_details
```
One example use case is splitting parsing into multiple separate jobs when the beam size is increased, in order to reduce wall time cost. Remember to concatenate your files together once you are finished, e.g., 
```
cat en_gum-ud-dev_beam_size16_model26_0_394 en_gum-ud-dev_beam_size16_model26_394_788 en_gum-ud-dev_beam_size16_model26_788_1182 en_gum-ud-dev_beam_size16_model26_1182_1575 > en_gum-ud-dev_beam_size16_model26_0_1575.conllx
```

## Evaluate System Parses against Gold Parses
Evaluate a system-parse CoNLL file against a gold- or reference-parse CoNLL file, in terms of labelled and unlabelled attachment score.
From the `ilcgdp` directory, e.g.,
```
python main.py  --evaluate_conll
                --system_parses ../en_gum-ud/en_gum-ud-dev_beam_size2_model26_0_1575.conllx
                --gold_parses ../UD_data/UD_English-GUM-master/en_gum-ud-dev.conllu
```


## Parse Arbitrary Sentences and Compute Word-by-Word Complexity Metrics
`--sentences_to_parse` is a .txt file with one sentence per line. Sentences should already be tokenized --- space-separated --- according to the tokenizaton scheme of the training corpus. The input will automatically be downcased and UNK-ified. From the `ilcgdp` directory, e.g.,
>
>
>
```
python main.py  --parse_sentences
                --analysis_dir ../en_gum-ud
                --train_file ../UD_data/UD_English-GUM-master/en_gum-ud-train.conllu
                --sentences_to_parse ../en_gum-ud/my_sentences.txt
                --beam_size 64
                --model_number 26
                --show_cuda_device_details
```
One output file records the CoNLL-formated dependency parses, while anothe records word-by-word complexity metrics. Currently, only surprisal is implemented --- as the negative log ratio of the current to former prefix probability, in bits, for beams resulting from SHIFT actions.



## Full Help Menu
```
usage: main.py [-h] [--train_file TRAIN_FILE] [--dev_file DEV_FILE] [--test_file TEST_FILE] [--analysis_dir ANALYSIS_DIR] [--prepare_features]
               [--train_model] [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE] [--word_embedding_size WORD_EMBEDDING_SIZE]
               [--dep_embedding_size DEP_EMBEDDING_SIZE] [--node_size NODE_SIZE] [--spine_size SPINE_SIZE] [--hidden_size HIDDEN_SIZE] [--save_all_models]
               [--beam_parse_conll] [--conll_file_to_parse CONLL_FILE_TO_PARSE] [--start_sentence_index START_SENTENCE_INDEX]
               [--end_sentence_index END_SENTENCE_INDEX] [--beam_size BEAM_SIZE] [--model_number MODEL_NUMBER] [--evaluate_conll]
               [--system_parses SYSTEM_PARSES] [--gold_parses GOLD_PARSES] [--eval_log_file EVAL_LOG_FILE] [--parse_sentences]
               [--sentences_to_parse SENTENCES_TO_PARSE] [--show_cuda_device_details]

options:
  -h, --help            show this help message and exit
  --train_file TRAIN_FILE
                        training set dependency data, in conll format
  --dev_file DEV_FILE   development set dependency data, in conll format
  --test_file TEST_FILE
                        test set dependency data, in conll format
  --analysis_dir ANALYSIS_DIR
                        directory to store features, models, logs, etc.
  --prepare_features    prepare and save the tabular training and dev feature data
  --train_model         train a neural net for locally-normalized transition prediction and language modeling
  --num_epochs NUM_EPOCHS
                        number of epochs to train the model
  --batch_size BATCH_SIZE
                        training batch size
  --word_embedding_size WORD_EMBEDDING_SIZE
                        word embedding size
  --dep_embedding_size DEP_EMBEDDING_SIZE
                        dependency embedding size
  --node_size NODE_SIZE
                        node size
  --spine_size SPINE_SIZE
                        right spine size
  --hidden_size HIDDEN_SIZE
                        hidden size
  --save_all_models     saves the model after every epoch; default is to only save if an improvement is made on one of the evaluation metrics
  --beam_parse_conll    beam parse the sentence of a conll file
  --conll_file_to_parse CONLL_FILE_TO_PARSE
                        connl file to beam parse
  --start_sentence_index START_SENTENCE_INDEX
                        sentence index to begin parsing at
  --end_sentence_index END_SENTENCE_INDEX
                        sentence index to stop parsing at
  --beam_size BEAM_SIZE
                        beam size
  --model_number MODEL_NUMBER
                        epoch number for the model to load for parsing
  --evaluate_conll      evaluate LAS and UAS for system parses, against the gold-standard parses, excluding punctuation
  --system_parses SYSTEM_PARSES
                        CoNLL-formatted system output file to be evaluated
  --gold_parses GOLD_PARSES
                        CoNLL-formatted gold-standard parses
  --eval_log_file EVAL_LOG_FILE
                        file name to write the evaluation details to
  --parse_sentences     parse sentences in a text file. separately writes out parses and word-by-word complexity metrics
  --sentences_to_parse SENTENCES_TO_PARSE
                        .txt file containing sentences to parse, one per line and pre-tokenized (space-separated) in accoradance with your
                        language/treebank
  --show_cuda_device_details
                        show CUDA device details
```
