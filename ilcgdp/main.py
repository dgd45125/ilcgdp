#!/usr/bin/env python

import argparse, os, sys
import numpy as np
import torch
import string
import subprocess

from utils import process_conll_data, get_training_data, tokenized_sent_to_Sentence
from parser import Vocabulary

from modeling import  train_model, get_data_loader, model, device, beam_parse

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', type = str, help='training set dependency data, in conll format')
    parser.add_argument('--dev_file', type = str, help='development set dependency data, in conll format')
    parser.add_argument('--test_file', type = str, help='test set dependency data, in conll format')

    parser.add_argument('--analysis_dir', type = str, help='directory to store features, models, logs, etc.')

    parser.add_argument('--prepare_features', help="prepare and save the tabular training and dev feature data", action="store_true")

    parser.add_argument('--train_model', help="train a neural net for locally-normalized transition prediction and language modeling", action="store_true")
    parser.add_argument('--num_epochs', help="number of epochs to train the model", default = 10, type=int)
    parser.add_argument('--batch_size', help="training batch size", default = 2048, type=int)   
    parser.add_argument('--word_embedding_size', help="word embedding size", default = 300, type=int)
    parser.add_argument('--dep_embedding_size', help="dependency embedding size", default = 50, type=int)     
    parser.add_argument('--node_size', help="node size", default = 500, type=int)                 #(3*word_embed + 3*dep_embed, node_size)   
    parser.add_argument('--spine_size', help="right spine size", default = 1000, type=int)         #(4*node_size, spine_size)
    parser.add_argument('--hidden_size', help="hidden size", default = 2000, type=int)            #(4*spine_size + word_embed + dep_embed, hidden_size)
    parser.add_argument('--save_all_models', help="saves the model after every epoch; default is to only save if an improvement is made on one of the evaluation metrics", action='store_true')
    
    parser.add_argument('--beam_parse_conll', help="beam parse the sentence of a conll file", action="store_true")
    parser.add_argument('--conll_file_to_parse', type = str, help="connl file to beam parse")
    parser.add_argument('--start_sentence_index', type = int, help="sentence index to begin parsing at")
    parser.add_argument('--end_sentence_index', type = int, help="sentence index to stop parsing at")
    parser.add_argument('--beam_size', help="beam size", default = 1, type=int)
    parser.add_argument('--model_number', help="epoch number for the model to load for parsing", default = 1)

    parser.add_argument('--evaluate_conll', help="evaluate LAS and UAS for system parses, against the gold-standard parses, excluding punctuation", action="store_true")
    parser.add_argument('--system_parses', help="CoNLL-formatted system output file to be evaluated")
    parser.add_argument('--gold_parses', help="CoNLL-formatted gold-standard parses")
    parser.add_argument('--eval_log_file', help="file name to write the evaluation details to")

    parser.add_argument('--parse_sentences', help="parse sentences in a text file. separately writes out parses and word-by-word complexity metrics", action = "store_true")
    parser.add_argument('--sentences_to_parse', help='.txt file containing sentences to parse, one per line and pre-tokenized (space-separated) in accoradance with your language/treebank')


    parser.add_argument('--show_cuda_device_details', help='show CUDA device details', action="store_true")


    args = parser.parse_args()


    print("------------------------------------------------------------------------\n")


    if args.show_cuda_device_details:
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Current CUDA device: {torch.cuda.current_device()}")
        print("Setting CUDA device if it is available; else, cpu")
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

        print("\n------------------------------------------------------------------------\n")
    
    
    if args.prepare_features:
        
        if not args.train_file:
            print("A training data file is required with --train_file to prepare features.")
            sys.exit()
            
        if not args.dev_file:
            print("A development data file is required with --dev_file to prepare features.")
            sys.exit()

        if args.analysis_dir:
            if not os.path.isdir(args.analysis_dir):
                print(f"Creating analysis directory: {args.analysis_dir}\n")
                os.makedirs(args.analysis_dir)
        else:
            print(f"No analysis directory was specified. Using ../DEPENDENCY_PARSING_ANALYSIS as default")
            if not os.path.isdir("../DEPENDENCY_PARSING_ANALYSIS"):
                print(f"Creating analysis directory ../DEPENDENCY_PARSING_ANALYSIS\n")
                os.makedirs("../DEPENDENCY_PARSING_ANALYSIS")
            args.analysis_dir = "../DEPENDENCY_PARSING_ANALYSIS"


        
        train_sentences = process_conll_data(args.train_file)
        vocab = Vocabulary()
        vocab.populate_vocabulary(train_sentences)

        dev_sentences = process_conll_data(args.dev_file)
        vocab.unkify(dev_sentences)


        #there are 2n-1 transitions for each sentence, where n is the number of tokens on the stack, including the ROOT node
        ttl_num_train_transitions = sum([(2*len(sent.buffer))-1 for sent in train_sentences])
        print(f"Number of train transitions: {ttl_num_train_transitions}")
        ttl_num_dev_transitions = sum([(2*len(sent.buffer))-1 for sent in dev_sentences])
        print(f"Number of dev transitions: {ttl_num_dev_transitions}\n")

        print(f"\nExtracting Train Features and saving at {args.analysis_dir}/train_feats.npy \n")
        get_training_data(train_sentences,ttl_num_train_transitions, f'{args.analysis_dir}/train_feats', vocab)
        print("Done.")
        print(f"\n\nExtracting Dev Features and saving at {args.analysis_dir}/dev_feats.npy \n")
        get_training_data(dev_sentences,ttl_num_dev_transitions, f'{args.analysis_dir}/dev_feats', vocab)
        print("Done.")
        print("\n------------------------------------------------------------------------\n")


    
    if args.train_model:
        
        if not args.train_file:
            print("A training data file is required with --train_file to train a model.")
            sys.exit()
            
        if (not os.path.isfile(f'{args.analysis_dir}/train_feats.npy')) or (not os.path.isfile(f'{args.analysis_dir}/dev_feats.npy')):
            print("\nWhoops! It looks like you don't haven't prepared the training and dev feature data yet. Try running again with the --prepare_features flag.")
            sys.exit()

        if not os.path.isdir(f"{args.analysis_dir}/models"):
            print(f"Creating directory 'models' inside of : {args.analysis_dir} to save the models and training log file in\n")
            os.makedirs(f"{args.analysis_dir}/models")

        
        train_sentences = process_conll_data(args.train_file)
        vocab = Vocabulary()
        vocab.populate_vocabulary(train_sentences)

        print(f"Preparing training and dev dataloaders with batch size: {args.batch_size}\n")
        train_dataloader = get_data_loader(np.load(f'{args.analysis_dir}/train_feats.npy'), args.batch_size)
        dev_dataloader = get_data_loader(np.load(f'{args.analysis_dir}/dev_feats.npy'), args.batch_size)

        print(f"Initializing the Torch model with:\n\tword embeddings size: {args.word_embedding_size}\n\tdependency embedding size: {args.dep_embedding_size}\n\tnode size: {args.node_size}\n\tspine size: {args.spine_size}\n\thidden size: {args.hidden_size}\n")
        my_model = model(num_form_embeddings = len(vocab.form2idx), form_embedding_size = args.word_embedding_size, 
              num_dep_embeddings=len(vocab.dep2idx), dep_embedding_size = args.dep_embedding_size,
              node_size = args.node_size, spine_size = args.spine_size,
             hidden_size = args.hidden_size, num_transitions = len(vocab.transition2idx)).to(device)
    
       
        print(f"Launching the training loop.")
        if args.save_all_models:
            print(f"Models are saved after every epoch to: {args.analysis_dir}/models.")
        else:
            print(f"Models that improve Dev Transition Accuracy, Dev Transition Loss, or Dev Lexical LM Loss will be saved to: {args.analysis_dir}/models.")
        print(f"The training log is being recorded at: {args.analysis_dir}/train_log.txt.")
        print(f"Currently set to train for {args.num_epochs} epochs.\n\n")
        
        train_model(_model = my_model, _train_dataloader = train_dataloader, 
                    _dev_dataloader = dev_dataloader, _num_epochs = args.num_epochs,
                     _save_all_models = args.save_all_models, _model_save_location = f"{args.analysis_dir}/models")

        print("\n------------------------------------------------------------------------\n")


    if args.beam_parse_conll:

        if not args.train_file:
            print("A training data file is required with --train_file to parse with a model.")
            sys.exit()

        train_sentences = process_conll_data(args.train_file)
        vocab = Vocabulary()
        vocab.populate_vocabulary(train_sentences)

        if not args.conll_file_to_parse:
            print("A CoNLL format file of sentence to parse is required with the --conll_file_to_parse argument.")
            sys.exit(0)

        sentences_to_parse = process_conll_data(args.conll_file_to_parse)
        vocab.unkify(sentences_to_parse)
                

        if not os.path.isfile(f"{args.analysis_dir}/models/{args.model_number}.pth"):
            print(f"\nWhoops! It looks like the specified model doesn't exist: {args.analysis_dir}/models/{MODEL_NUM}.pth")
            sys.exit(0)

        print("\nLoading the Torch model\n")

        my_model = model(num_form_embeddings = len(vocab.form2idx), form_embedding_size = args.word_embedding_size, 
              num_dep_embeddings=len(vocab.dep2idx), dep_embedding_size = args.dep_embedding_size,
              node_size = args.node_size, spine_size = args.spine_size,
             hidden_size = args.hidden_size, num_transitions = len(vocab.transition2idx)).to(device)
        
        my_model.load_state_dict(torch.load(f"{args.analysis_dir}/models/{args.model_number}.pth",weights_only=True))
        my_model.eval()


        out_file_name = args.conll_file_to_parse.split("/")[-1].split(".")[0] #e.g., ptb_dev

        if args.start_sentence_index and not args.end_sentence_index:
            out_conll_file = f"{args.analysis_dir}/{out_file_name}_beam_size{args.beam_size}_model{args.model_number}_{args.start_sentence_index}_{len(sentences_to_parse)}.conllx"

        elif args.end_sentence_index and not args.start_sentence_index:
            out_conll_file = f"{args.analysis_dir}/{out_file_name}_beam_size{args.beam_size}_model{args.model_number}_0_{args.end_sentence_index}.conllx"

        elif args.end_sentence_index and args.start_sentence_index:
            out_conll_file = f"{args.analysis_dir}/{out_file_name}_beam_size{args.beam_size}_model{args.model_number}_{args.start_sentence_index}_{args.end_sentence_index}.conllx"

        else:
            out_conll_file = f"{args.analysis_dir}/{out_file_name}_beam_size{args.beam_size}_model{args.model_number}_0_{len(sentences_to_parse)}.conllx"


        print("Parsing")
        print(f"CoNLL output is being written to: {out_conll_file}\n")
        if os.path.isfile(out_conll_file):
            print(f"Overwriting file: {out_conll_file}")
            os.remove(out_conll_file)

        if args.start_sentence_index:
            strt_idx = args.start_sentence_index
        else:
            strt_idx = 0
        if args.end_sentence_index:
            end_idx = args.end_sentence_index
        else:
            end_idx = len(sentences_to_parse)
        
        for i in range(strt_idx, end_idx):    
            #print(f"Sentence number: {i}")
            out_stream = open(out_conll_file,'a')
            parsed = beam_parse(sentences_to_parse[i], my_model, vocab, beam_size=args.beam_size, verbose=False, stat_file=None)
            parsed.write_conll(out_stream)
            out_stream.close()
            
            if i%100 == 0:
                print(f"Finished sentence at index: {i} --- Going to sentence at index: {end_idx}")

        
        print("\n------------------------------------------------------------------------\n")
        

    if args.evaluate_conll:

        if not args.system_parses:
            print("A CoNLL system parses file is required with --system_parses in order to evaluate.")
            sys.exit()
        if not args.gold_parses:
            print("A CoNLL gold standard parses file is required with --gold_parses in order to evaluate.")
            sys.exit()
        if not args.eval_log_file:
            print("A file name to write the evaluation details to is required with --eval_log_file in order to evaluate.")
            sys.exit()

            
        print(f"\nUsing the ../eval.pl script to evaluate System parses: {args.system_parses}, against Gold standard parses: {args.gold_parses}.")
        print(f"Details will be written to: {args.eval_log_file}\n")

        try:
            result = subprocess.run(["perl", "../eval.pl", "-o", args.eval_log_file, "-g", args.gold_parses, "-s", args.system_parses], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"../eval.pl failed with exit code {e.returncode}")
            sys.exit()
     
        print("\n------------------------------------------------------------------------\n")


    if args.parse_sentences:

        if not args.analysis_dir:
            print("An analysis directory is required with --analysis_dir to parse sentences.")
            sys.exit()

        if not args.train_file:
            print("A training data file is required with --train_file to parse sentences.")
            sys.exit()

        if not args.sentences_to_parse:
            print("A .txt file is required with --sentences_to_parse.")
            sys.exit()


        train_sentences = process_conll_data(args.train_file)
        vocab = Vocabulary()
        vocab.populate_vocabulary(train_sentences)

        f = open(args.sentences_to_parse, 'r')
        tokenized_sentences = [line.strip().split() for line in f.readlines()]
        f.close()

        sentences_to_parse = [tokenized_sent_to_Sentence(sent, vocab) for sent in tokenized_sentences]

        if not os.path.isfile(f"{args.analysis_dir}/models/{args.model_number}.pth"):
            print(f"\nWhoops! It looks like the specified model doesn't exist: {args.analysis_dir}/models/{MODEL_NUM}.pth")
            sys.exit(0)

        print("\nLoading the Torch model\n")

        my_model = model(num_form_embeddings = len(vocab.form2idx), form_embedding_size = args.word_embedding_size, 
              num_dep_embeddings=len(vocab.dep2idx), dep_embedding_size = args.dep_embedding_size,
              node_size = args.node_size, spine_size = args.spine_size,
             hidden_size = args.hidden_size, num_transitions = len(vocab.transition2idx)).to(device)
        
        my_model.load_state_dict(torch.load(f"{args.analysis_dir}/models/{args.model_number}.pth",weights_only=True))
        my_model.eval()

        out_file_name = args.sentences_to_parse.split("/")[-1].split(".")[0] #e.g., 'my_sentences' from '../ptb_ud/my_sentences.txt' 
        out_conll_file = f"{args.analysis_dir}/{out_file_name}_beam_size{args.beam_size}_model{args.model_number}.conllx"
        out_stat_file = f"{args.analysis_dir}/{out_file_name}_beam_size{args.beam_size}_model{args.model_number}_stats.txt"

        
        print("Parsing")
        print(f"CoNLL output is being written to: {out_conll_file}")
        print(f"Complexity metric stats are being written to: {out_stat_file}\n")
        if os.path.isfile(out_conll_file):
            print(f"Overwriting file: {out_conll_file}")
            os.remove(out_conll_file)
        if os.path.isfile(out_stat_file):
            print(f"Overwriting file: {out_stat_file}")
            os.remove(out_stat_file)

        
        for i in range(len(sentences_to_parse)):    
            #print(f"Sentence number: {i}")
            out_stream = open(out_conll_file,'a')
            parsed = beam_parse(sentences_to_parse[i], my_model, vocab, beam_size=args.beam_size, verbose=False, stat_file=out_stat_file)
            parsed.write_conll(out_stream)
            out_stream.close()
            #if i%100 == 0:
            print(f"{i+1}/{len(sentences_to_parse)} sentences")

        
        print("\n------------------------------------------------------------------------\n")
        
