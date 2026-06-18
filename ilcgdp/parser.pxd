import numpy as np
cimport numpy as np

'''
an "extension type" for representing a CONLL token.
initialized with the CONLL features of a token (i.e., a line in a CONLL file)
'''
cdef class Token:
    cdef public int idx, gold_head    #unfortunately, id is already a built-in function that returns the unique integer identity of an object in memory, so Tokens get an integer attribute idx, instead
    cdef public str form, lemma, upos, xpos, feats, gold_deprel, deps, misc

    cpdef Token copy(self)            #makes a deep copy of the Token

    

'''
an extension type for representing a node in a RightSpine
'''
cdef class Node:
    cdef public int token_idx, parent_idx, form_idx, pos_idx, dep_idx    
    cdef public bint is_complete_node, is_labeled_dummy_node, is_unlabeled_dummy_node
    cdef public str form, pos, dep
    cdef public list left_children              #list of left children Nodes

    cpdef void print_state(self, int depth)     #print state
    cpdef Node copy(self)                       #makes a deep copy of the Node

    #get the features of the node
    cpdef np.ndarray[np.int32_t, ndim=1] get_node_features_state(self, Vocabulary train_vocab)

    
    
'''
An extension type for representing a Right Spine on the stack
'''

cdef class RightSpine:
    
    cdef public list node_list    

    cpdef void print_state(self)       #print the Right Spine
    cpdef RightSpine copy(self)        #make a deep copy  

    #get the features of the RightSpine
    cpdef np.ndarray[np.int32_t, ndim=1] get_right_spine_features_state(self, Vocabulary train_vocab)



'''
An "extension type" for handling and representing the (un)observed tokens, POS tags, and dependencies from our dependency data.    
For forms, POSs, dependencies, and transitions, creates dicitonaries which map back and forth to indices.

Takes no initialization arguments. Class attributes are instead instantiated through functions    
'''
cdef class Vocabulary:
    cdef public dict form2idx, idx2form, pos2idx, idx2pos, dep2idx, idx2dep, transition2idx, idx2transition 

    cpdef void populate_vocabulary(self, list training_sentences)
    cpdef void unkify(self, list sentences)


    
    

'''
an "extension type" for representing the state of a sentence and its parse;
initialized as the initial configuration of the parse state
'''
cdef class Sentence:

    cdef public int num_tokens           #number of tokens in the sentence
    cdef public list buffer              #buffer of to-be-processed Tokens
    cdef public list stack               #stack of RightSpines
    cdef public double log_probability    #log probability of the parse
    cdef public list assigned_deps       #list of assigned dependencies, dependencies are represented as a 3-tuple: (head_idx, label, child_idx)
    cdef public list gold_deps           #list of gold dependencies, dependencies are represented as a 3-tuple: (head_idx, label, child_idx)
    cdef public list pretty_stack        #non-necessary stack representation that allows for easily viewing the stack state
    cdef public str identifier           #a string for identifying this parse as well as its children
    cdef public double most_recent_word_gen_log_prob     #keeps track of the most recent word generation prob so we can compare with/without
    cdef public str most_recent_transition #keep track of the most recent transition
    cdef public dict idx2form            #dictionary for mapping token index position in the sentence to the token form

    cpdef bint is_final_configuration(self)   #returns whether or not parse state is in the final configuration



    
    
    '''
    this method prints the current parse state for a Sentence. I.e., the state of the (pretty print) stack, buffer, and dependencies
    in terms of the (IDs and forms of the Tokens)
    '''
    cpdef void print_state(self)
    
    cpdef void print_stack(self)                 #prints the functional representation of the stack
    cpdef void write_conll(self, file_stream)    #write out the CoNLL-formatted parse
    cpdef void print_conll(self)                 #print CoNLL-formatted parse 
    cpdef Sentence copy(self)                    #makes a deep copy of the Sentence

    cpdef np.ndarray[np.int32_t, ndim=1] get_features_state(self, Vocabulary train_vocab)     #returns a 1-dimensional numpy array containing feature attributes

    cpdef tuple get_valid_transitions(self, Vocabulary train_vocab) #returns a list of strings of the valid transitions
    
    '''
    Labeled left-corner transition system
    '''
    #shifts token at the front of the buffer onto the stack as a new Right Spine
    cpdef int shift(self, Vocabulary train_vocab)
    
    #Insert into an existing partial tree. I.E. the inserted node will be assigned a governor and a labeled arc
    cpdef int insert_into_tree(self, Vocabulary train_vocab)

    #Insert into an UNlabled Dummy Node. I.E. the inserted node with NOT be assigned a governor or a labeled arc
    cpdef int insert_as_head(self, Vocabulary train_vocab)
    
    #predicts that the head of the right-spine on the top of the stack has a new right-most child/terminal, which is a labeled dummy node
    cpdef int right_pred(self, str label, Vocabulary train_vocab)

    #construct an UNlabeled Dummy Node and place the head of the right spine on the stack as a left child of that Dummy Node
    cpdef int left_pred(self, str label, Vocabulary train_vocab)

    #Given that second-to-topmost right spine has a dummy node terminal (labled or unlabeled), add head of topmost right spine as a left child of that dummy node
    cpdef int left_comp(self, str label, Vocabulary train_vocab)

    #Given that the second to top right-spine on the stack has a Labled Dummy Node as the terminal child, insert the head of the 
    #topmost right-spine on the stack ino that DUMMY NODE and add a new LABELED Dummy Node to the right-spine that you are inserting into
    cpdef int right_comp(self, str label, Vocabulary train_vocab)


    cpdef str get_oracle_transition(self) #get the oracle transition, given the curent parset state
        
    
        



    