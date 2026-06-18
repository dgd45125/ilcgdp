import numpy as np
cimport numpy as np

from parser cimport Token, Sentence, Vocabulary, RightSpine


cpdef list process_conll_data(str file_path):
    '''
    This function takes the file path for a conll file, and returns a list of Sentence objects, each of which 
    represents the state of a sentence and its parse (in this case, the initial configuration of the parse state).
    '''
    cdef list sentences = [], current_sentence_tokens = [], parts, _buffer, _stack, _assigned_deps, _gold_deps = [], _pretty_stack
    cdef str line,
    cdef double _log_probability
    cdef Token root_token
    cdef dict idx2form = {}
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            #remove white space, including new line chars
            line = line.strip()
            #empty line indicates the end of a sentence
            if not line:  
                #if current_sentence is not empty, prepare the Tokens in the buffer and prepare and create a Sentence
                if current_sentence_tokens:   
                    #print([token.idx for token in current_sentence_tokens])
                    #create a ROOT token to go at the front of the buffer
                    root_token = Token(_idx = 0, _form= "ROOT", _lemma="ROOT", _upos= "ROOT", _xpos= "ROOT",  _feats= "ROOT", _gold_head= -1, _gold_deprel= "ROOT", _deps= "ROOT", _misc= "ROOT")

                    #add ROOT form to the idx2form dict
                    idx2form[0] = 'ROOT'
                        
                    #create the buffer that will be passed to the Sentence constructor
                    _buffer = [root_token] + current_sentence_tokens            
                    _stack = []
                    _log_probability = 0.0
                    _assigned_deps = []
                    _pretty_stack = []
                    sentences.append(Sentence(_num_tokens = len(current_sentence_tokens), _buffer = _buffer, _stack = _stack, _log_probability = _log_probability, _assigned_deps = _assigned_deps, _gold_deps = _gold_deps, _pretty_stack = _pretty_stack, _identifier = '1', _most_recent_word_gen_log_prob=0.0, _most_recent_transition = "", _idx2form = idx2form))
                    current_sentence_tokens = []
                    _gold_deps = []
                    idx2form = {}
                    
            #we do have another line/token
            else:
                parts = line.split('\t')
                #ensure it's a valid CoNLL line (i.e. has 10 values) and not e.g. a comment (comments begin with #)
                #also make sure that it is not an index range multiword token or an empty node (i.e., doesn't have no head; UD problem)
                if len(parts) == 10 and not parts[6]=='_':  
                    #make a new Token and add it to the running Token list for this sentence
                    current_sentence_tokens.append(Token(_idx = int(parts[0]), _form= parts[1].lower(), _lemma=parts[2], _upos= parts[3], _xpos= parts[4], _feats= parts[5], _gold_head= int(parts[6]), _gold_deprel= parts[7], _deps= parts[8], _misc= parts[9]))
                    #add the gold labeled dependency relation to the running list of gold sentence labeled dependencies
                    _gold_deps.append((int(parts[6]), parts[7], int(parts[0])))
                    #add this token from at the index to the idx2form dictionary
                    idx2form[int(parts[0])] = parts[1].lower()
                                    
             
    return sentences




cpdef Sentence tokenized_sent_to_Sentence(list tokenized_sent, Vocabulary train_vocab):
    '''
    This function takes a tokenized input sentence (list of strings) and returns a Sentence object (Parse State)
    '''
    cdef Sentence sentence
    cdef list current_sentence_tokens = [],  _buffer, _stack, _assigned_deps, _gold_deps = [], _pretty_stack
    cdef int i = 0
    cdef Token root_token
    cdef double _log_probability
    cdef dict idx2form = {}
    
    current_sentence_tokens = []
    tokenized_sent = [tok.lower() for tok in tokenized_sent]

    root_token = Token(_idx = 0, _form= "ROOT", _lemma="ROOT", _upos= "ROOT", _xpos= "ROOT",  _feats= "ROOT", _gold_head= -1, _gold_deprel= "ROOT", _deps= "ROOT", _misc= "ROOT")

    #add ROOT form to the idx2form dict
    idx2form[0] = 'ROOT'
    
    for i in range(len(tokenized_sent)):
        current_sentence_tokens.append(Token(_idx = i+1, 
                                  _form= tokenized_sent[i], 
                                  _lemma= '_', 
                                  _upos= 'not_available', 
                                  _xpos= '_', 
                                  _feats='_', 
                                  _gold_head= 0, 
                                  _gold_deprel= '_', 
                                  _deps= '_' , 
                                  _misc= '_')
            )
        idx2form[i+1] = tokenized_sent[i]
                                   
    #create the buffer that will be passed to the Sentence constructor
    _buffer = [root_token] + current_sentence_tokens            
    _stack = []
    _log_probability = 0.0
    _assigned_deps = []
    _pretty_stack = []
    

    
    sentence = Sentence(_num_tokens = len(current_sentence_tokens), _buffer = _buffer, _stack = _stack, _log_probability = _log_probability, _assigned_deps = _assigned_deps, _gold_deps = _gold_deps, _pretty_stack = _pretty_stack, _identifier = '1',_most_recent_word_gen_log_prob=0.0, _most_recent_transition = "",  _idx2form = idx2form)

    train_vocab.unkify([sentence])
    
    return sentence


    

cpdef void print_oracle_sequence(Sentence sentence, Vocabulary train_vocab, int num_transitions, bint silence, bint print_stack_details):
    '''
    This functon takes a Sentence object and walks through the Labeled Left Corner Oracle parse, printing
    out the transitions and state along the way.
    '''
    cdef int i
    cdef str transition, action, label
    
    if sentence.is_final_configuration():
        if not silence: 
            sentence.print_state()
            if print_stack_details: sentence.print_stack()
        return
    if not silence: 
        sentence.print_state()
        if print_stack_details: sentence.print_stack()

    if num_transitions ==-1:
        while True:
            transition = sentence.get_oracle_transition()
            action = transition.split("(")[0]
            label = transition.split("(")[-1][:-1]
                       
            if action=='shift':
                if not silence: print("\nshift")
                sentence.shift(train_vocab)   
            if action =='insert_as_head':
                if not silence: print(f'\ninsert_as_head')
                sentence.insert_as_head(train_vocab)
            if action == 'insert_into_tree':
                if not silence: print('\ninsert_into_tree')
                sentence.insert_into_tree(train_vocab)  
            if action =='right_comp':
                if not silence: print(f"\nright_comp({label})")
                sentence.right_comp(label, train_vocab)
            if action =='left_comp':
                if not silence: print(f"\nleft_comp({label})")
                sentence.left_comp(label, train_vocab)
            if action =='right_pred':
                if not silence: print(f'\nright_pred({label})')
                sentence.right_pred(label, train_vocab)
            if action =='left_pred':
                if not silence: print(f'\nleft_pred({label})')
                sentence.left_pred(label, train_vocab)  
            if sentence.is_final_configuration():
                if not silence: 
                    print("")
                    sentence.print_state()
                    if print_stack_details: sentence.print_stack()
                break
            else:
                if not silence: 
                    print("")
                    sentence.print_state()
                    if print_stack_details: sentence.print_stack()
    
    else:
        for i in range(num_transitions):
            transition = sentence.get_oracle_transition()
            action = transition.split("(")[0]
            label = transition.split("(")[-1][:-1]
    
            if action=='shift':
                if not silence: print("\nshift")
                sentence.shift(train_vocab)
            if action =='insert_as_head':
                if not silence: print('\ninsert_as_head')
                sentence.insert_as_head(train_vocab)
            if action == 'insert_into_tree':
                if not silence: print('\ninsert_into_tree')
                sentence.insert_into_tree(train_vocab)
    
            if action =='right_comp':
                if not silence: print(f"\nright_comp({label})")
                sentence.right_comp(label, train_vocab)
            if action =='left_comp':
                if not silence: print(f"\nleft_comp({label})")
                sentence.left_comp(label, train_vocab)
            if action == 'right_pred':
                if not silence: print(f"\nright_pred({label})")
                sentence.right_pred(label, train_vocab)
            if action =='left_pred':
                if not silence: print(f"\nleft_pred({label})")
                sentence.left_pred(label, train_vocab)
    
            if sentence.is_final_configuration():
                if not silence: 
                    sentence.print_state()
                    if print_stack_details: sentence.print_stack()
                break
            else:
                if not silence: 
                    print("\n")
                    sentence.print_state()
                    if print_stack_details: sentence.print_stack()





                    

                    

cpdef void get_training_data(list sentences, int ttl_num_transitions, str save_file_name, Vocabulary train_vocab):    
    '''
    This function takes as input the sentences (list of Sentence objects), the total number of 
    transitions (2n-1; including ROOT on the buffer) and then, for each sentence, walks through the
    oracle parse, inserting into a numpy array, the current parse features state and the corresponding oracle transition

    When finished, saves the numpy array
    '''
    cdef int i, transition_counter = 0                       
    cdef np.ndarray[np.int32_t, ndim=2] training_data = np.empty((ttl_num_transitions,99), dtype = np.int32)   #number of stack features (96) + next word as feature + next_next word to predict + labeled transition to predict
    cdef np.ndarray[np.int32_t, ndim=1] feats
    cdef np.ndarray[np.int32_t, ndim=1] transition_idx
    cdef str transition, action, label
    cdef Sentence sentence


    for i in range(len(sentences)):

        sentence = sentences[i]

        try:
            while True:
                #print(f"training_data: {training_data}")
                feats = sentence.get_features_state(train_vocab)
                #print(feats)
                transition = sentence.get_oracle_transition()
                #print(transition)
                action = transition.split("(")[0]
                label = transition.split("(")[-1][:-1]
    
                transition_idx = np.array([train_vocab.transition2idx[transition]],dtype = np.int32)
                #print(f"transition_idx: {transition_idx}")
                feats = np.concatenate((feats,transition_idx))
                #print(feats)
                if action=='shift':
                    sentence.shift(train_vocab)   
                if action =='insert_as_head':
                    sentence.insert_as_head(train_vocab)
                if action == 'insert_into_tree':
                    sentence.insert_into_tree(train_vocab)  
                if action =='right_comp':
                    sentence.right_comp(label, train_vocab)
                if action =='left_comp':
                    sentence.left_comp(label, train_vocab)
                if action =='right_pred':
                    sentence.right_pred(label, train_vocab)
                if action =='left_pred':
                    sentence.left_pred(label, train_vocab)  
                if sentence.is_final_configuration():
                    training_data[transition_counter] = feats
                    transition_counter +=1
                    break
                else:
                    training_data[transition_counter] = feats
                    transition_counter +=1

        except:
            continue
                
        if i%1000 == 0:
            print(f"{i+1}/{len(sentences)} sentences")
 
    np.save(file = save_file_name, arr = training_data[:transition_counter,:])

  

