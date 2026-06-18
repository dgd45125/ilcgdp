import numpy as np
cimport numpy as np

import pickle

import copy

cdef class Token:

    def __cinit__(self, int _idx, str _form, str _lemma, str _upos, str _xpos, str _feats, int _gold_head, str _gold_deprel, str _deps, str _misc):
        self.idx = _idx
        self.form = _form
        self.lemma = _lemma
        self.upos = _upos
        self.xpos = _xpos
        self.feats = _feats
        self.gold_head = _gold_head
        self.gold_deprel = _gold_deprel
        self.deps = _deps
        self.misc = _misc

    #method for making a deep copy
    cpdef Token copy(self):
        return Token(_idx = self.idx, _form = self.form, _lemma = self.lemma, _upos = self.upos, _xpos =  self.xpos, _feats = self.feats, _gold_head = self.gold_head, _gold_deprel = self.gold_deprel, _deps = self.deps, _misc = self.misc)




    
cdef class Node:

    def __cinit__(self, int _token_idx, int _parent_idx, bint _is_complete_node, bint _is_labeled_dummy_node, bint _is_unlabeled_dummy_node, str _form, int _form_idx, str _pos, int _pos_idx, str _dep, int _dep_idx, list _left_children):        
        self.token_idx = _token_idx
        self.parent_idx = _parent_idx
        self.is_complete_node = _is_complete_node
        self.is_labeled_dummy_node = _is_labeled_dummy_node
        self.is_unlabeled_dummy_node = _is_unlabeled_dummy_node
        self.form = _form
        self.form_idx = _form_idx
        self.pos = _pos
        self.pos_idx = _pos_idx
        self.dep = _dep
        self.dep_idx = _dep_idx
        self.left_children = _left_children

    #print state
    cpdef void print_state(self, int depth):
        cdef Node node
        print(f"Token idx:{self.token_idx}, ",end="")
        print(f"Form:{self.form}/{self.form_idx}, ",end="")     
        #print(f"Pos:{self.pos}/{self.pos_idx}, ",end="")
        print(f"and Dep:{self.dep}/{self.dep_idx}, ",end="")
        if self.is_complete_node: print("Is Complete, ",end="")
        if self.is_labeled_dummy_node: print("Is Labeled Dummy, ",end="")
        if self.is_unlabeled_dummy_node: print("Is Unlabeled Dummy, ",end="")
        print(f"Parent idx:{self.parent_idx}, ", end="")
        print(f"w/Left Children:", end="")
        for node in self.left_children:
            print("\n"," "*(depth+1),"↳", end="")
            node.print_state(depth+1)
        print("")

 
    #method for making a deep copy
    cpdef Node copy(self):
        cdef Node node
        cdef list left_children_copy = [node.copy() for node in self.left_children]
        return Node(_token_idx = self.token_idx, _parent_idx = self.parent_idx, _is_complete_node = self.is_complete_node, _is_labeled_dummy_node = self.is_labeled_dummy_node, _is_unlabeled_dummy_node = self.is_unlabeled_dummy_node, _form = self.form, _form_idx = self.form_idx, _pos = self.pos, _pos_idx = self.pos_idx, _dep = self.dep, _dep_idx = self.dep_idx, _left_children = left_children_copy)

    #gets the feaures for a node, q: 
        #- form index and dependency index for q and q's leftmost (l1) and second-leftmost (l2) children
    cpdef np.ndarray[np.int32_t, ndim=1] get_node_features_state(self, Vocabulary train_vocab):
        cdef np.ndarray[np.int32_t, ndim=1] q_feats
        cdef np.ndarray[np.int32_t, ndim=1] ql1_feats
        cdef np.ndarray[np.int32_t, ndim=1] ql2_feats
        
                            #q_form_idx,   q_dep_idx
        q_feats = np.array([self.form_idx, self.dep_idx], dtype = np.int32)
            
        if self.left_children:        
            ql1_feats = np.array([self.left_children[0].form_idx, self.left_children[0].dep_idx], dtype = np.int32)   #ql1_form_idx,     ql1_dep_idx
    
            if len(self.left_children) > 1:
                ql2_feats = np.array([self.left_children[1].form_idx, self.left_children[1].dep_idx], dtype = np.int32)     #ql2_form_idx,     ql2_dep_idx
            else:
                ql2_feats = np.array([train_vocab.form2idx['not_available'], train_vocab.dep2idx['not_available']], dtype = np.int32)
            
        else:
            ql1_feats = np.array([train_vocab.form2idx['not_available'], train_vocab.dep2idx['not_available']], dtype = np.int32)
            ql2_feats = np.array([train_vocab.form2idx['not_available'], train_vocab.dep2idx['not_available']], dtype = np.int32)

        return np.concatenate((ql1_feats, ql2_feats, q_feats))
        
        
        

cdef class RightSpine:

    def __cinit__(self, list _node_list):
        self.node_list = _node_list
            
    #method for making a deep copy
    cpdef RightSpine copy(self):
        cdef Node node
        cdef list node_list_copy = [node.copy() for node in self.node_list]
        return RightSpine(node_list_copy)

    #print state
    cpdef void print_state(self):
        cdef Node node
        cdef int depth
        for node in self.node_list:
            node.print_state(depth = 0)


            
    #gets the feaures for a RightSpine
    cpdef np.ndarray[np.int32_t, ndim=1] get_right_spine_features_state(self, Vocabulary train_vocab):    
        cdef np.ndarray[np.int32_t, ndim=1] q_feats
        cdef np.ndarray[np.int32_t, ndim=1] p_feats
        cdef np.ndarray[np.int32_t, ndim=1] gp_feats
        cdef np.ndarray[np.int32_t, ndim=1] ggp_feats
        cdef int node_list_len = len(self.node_list)

        
                        #[ql1_form_idx, ql1_dep_idx, ql2_form_idx, ql2_dep_idx, q_form_idx,  q_dep_idx]
        q_feats = self.node_list[-1].get_node_features_state(train_vocab)

    
                         #[pl1_form_idx, pl1_dep_idx, pl2_form_idx, pl2_dep_idx, p_form_idx,  p_dep_idx]
        if node_list_len > 1:
            p_feats = self.node_list[-2].get_node_features_state(train_vocab)
        else:
            p_feats = np.array([train_vocab.form2idx['not_available'], train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'], train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'], train_vocab.dep2idx['not_available']], dtype = np.int32)

                        #[gpl1_form_idx, gpl1_dep_idx, gpl2_form_idx, gpl2_dep_idx, gp_form_idx,  gp_dep_idx]
        if node_list_len > 2:
            gp_feats = self.node_list[-3].get_node_features_state(train_vocab)
        else:
            gp_feats = np.array([train_vocab.form2idx['not_available'], train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'], train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'], train_vocab.dep2idx['not_available']], dtype = np.int32)

                        #[ggpl1_form_idx, ggpl1_dep_idx, ggpl2_form_idx, ggpl2_dep_idx, ggp_form_idx,  ggp_dep_idx]
        if node_list_len > 3:
            ggp_feats = self.node_list[-4].get_node_features_state(train_vocab)
        else:
            ggp_feats = np.array([train_vocab.form2idx['not_available'], train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'], train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'], train_vocab.dep2idx['not_available']], dtype = np.int32)

        
        return np.concatenate((ggp_feats, gp_feats, p_feats, q_feats))
        


cdef class Sentence:
    def __cinit__(self, int _num_tokens, list _buffer, list _stack, double _log_probability, list _assigned_deps, list _gold_deps, list _pretty_stack, str _identifier, double _most_recent_word_gen_log_prob, str _most_recent_transition, dict _idx2form):

        self.num_tokens = _num_tokens
        self.buffer = _buffer
        self.stack = _stack
        self.log_probability = _log_probability
        self.assigned_deps = _assigned_deps
        self.gold_deps = _gold_deps
        self.pretty_stack = _pretty_stack
        self.identifier = _identifier
        self.most_recent_word_gen_log_prob = _most_recent_word_gen_log_prob
        self.most_recent_transition = _most_recent_transition
        self.idx2form = _idx2form

    cpdef bint is_final_configuration(self):
        '''
        Returns true if the parse state is in the terminal configuration <[[ROOT, ...]], [], A>
        otherwise, false
        '''
        return len(self.stack)==1 and len(self.buffer) == 0 and self.stack[0].node_list[0].token_idx ==0

    cpdef void print_state(self):
        cdef Token token
        cdef tuple dep
        print(f"Stack: {self.pretty_stack}")       #print the pretty stack
        print(f"Buffer: {[(token.idx, token.form) for token in self.buffer]}")
        print(f"Dependencies: {[(dep[0], dep[1], dep[2]) for dep in self.assigned_deps]}")
        print(f"Is final configuration: {self.is_final_configuration()}")


    cpdef void print_stack(self):
        cdef int i
        for i in range(1, len(self.stack)+1):
            print(f"Right Spine at stack index {-i}:")
            self.stack[-i].print_state()
            print("")

    cpdef Sentence copy(self):
        cdef Token token
        cdef RightSpine right_spine
        cdef tuple tpl
        cdef list buffer_copy = [token.copy() for token in self.buffer]
        cdef list stack_copy = [right_spine.copy() for right_spine in self.stack]
        cdef list assigned_deps_copy = [tpl for tpl in self.assigned_deps]
        cdef list pretty_stack_copy = copy.deepcopy(self.pretty_stack)      #[spine_representation for spine_representation in self.pretty_stack]
        cdef dict idx2form_copy = self.idx2form
        
        return Sentence(_num_tokens = self.num_tokens, _buffer = buffer_copy, _stack = stack_copy, _log_probability = self.log_probability, _assigned_deps = assigned_deps_copy, _gold_deps = self.gold_deps, _pretty_stack = pretty_stack_copy, _identifier = self.identifier, _most_recent_word_gen_log_prob=self.most_recent_word_gen_log_prob, _most_recent_transition = self.most_recent_transition, _idx2form = idx2form_copy)


    cpdef tuple get_valid_transitions(self, Vocabulary train_vocab):
        '''
        returns a list of strings of the valid UNlabeled transitions, given the current parse state 
        AND a list containing the corresponding  LABELED vocab transition indices.
        '''
        cdef list valid_unlabeled_transitions = []     
        cdef list valid_transition_indices
        cdef list valid_transitions = []
        cdef str transition
        

        if self.stack == []:
            valid_unlabeled_transitions.append('shift')
        
        
        elif len(self.buffer) >= 1:
        
            if self.stack[-1].node_list[-1].is_labeled_dummy_node:  
                valid_unlabeled_transitions.append('shift')
                valid_unlabeled_transitions.append('insert_into_tree')

            if self.stack[-1].node_list[-1].is_unlabeled_dummy_node:
                valid_unlabeled_transitions.append('shift')
                valid_unlabeled_transitions.append('insert_as_head')

            if self.stack[-1].node_list[-1].is_complete_node:
                valid_unlabeled_transitions.append('right_pred')
                valid_unlabeled_transitions.append('left_pred')          

                if len(self.stack) >=2:
        
                    if self.stack[-2].node_list[-1].is_labeled_dummy_node:
                        valid_unlabeled_transitions.append('right_comp')
                        valid_unlabeled_transitions.append('left_comp')    
            
                    if self.stack[-2].node_list[-1].is_unlabeled_dummy_node:
                        valid_unlabeled_transitions.append('left_comp')
        else:
            pass

        valid_transitions = [transition for transition in train_vocab.transition2idx.keys() if  transition.split("(")[0] in valid_unlabeled_transitions]
        
        valid_transition_indices = [train_vocab.transition2idx[transition] for transition in train_vocab.transition2idx.keys() if  transition.split("(")[0] in valid_unlabeled_transitions]

                    
        return valid_unlabeled_transitions, valid_transitions, valid_transition_indices


    

    '''
    Labeled left-corner transition system
    '''
    
    cpdef int shift(self, Vocabulary train_vocab):
        '''
        shifts the token at the front of the buffer onto the stack as a new complete RightSpine.
        Returns 0 if successful and -1 otherwise
        '''
        cdef Node node
        cdef RightSpine right_spine
        if len(self.buffer) >=1:      
            node = Node(_token_idx = self.buffer[0].idx, _parent_idx = -1, _is_complete_node = True, _is_labeled_dummy_node = False, _is_unlabeled_dummy_node = False, _form = self.buffer[0].form, _form_idx = train_vocab.form2idx[self.buffer[0].form], _pos = self.buffer[0].upos, _pos_idx = train_vocab.pos2idx[self.buffer[0].upos], _dep = 'not_available', _dep_idx = train_vocab.dep2idx['not_available'], _left_children = [])
            if self.buffer[0].form=="ROOT":
                node.dep='ROOT'
                node.dep_idx = train_vocab.dep2idx['ROOT']
            right_spine = RightSpine([node])
            self.stack.append(right_spine)
            self.pretty_stack.append([self.buffer[0].idx])
            self.buffer.pop(0)
            return 0
        else:
            return -1

    cpdef int insert_into_tree(self, Vocabulary train_vocab):
        '''
        This transition is for inserting INTO AN EXISTING PARTIAL TREE. That is, into a Labeled Dummy Node.
        That is, the inserted node will be ASSIGNED A GOVERNOR AND A LABELED ARC.
        
        Inserts the word index at the front of the buffer into the LABELED dummy node that
        is the terminal child of the right-spine at the top of the stack.
        
        Adds the to-be-inserted index as a labeled child of the next-higher-up-node in the right-spine
        
        Adds each index that is currently a left-child of that dummy node
        as a left dependent of the inserted index.
        '''
        cdef Node node
        if len(self.buffer) >=1 and self.stack and self.stack[-1].node_list[-1].is_labeled_dummy_node:

            #handle children of the to-be-inserted-into node first. I.e. assign them as children of the to-be-inserted node
            for node in self.stack[-1].node_list[-1].left_children:
                #add the dependency relation to the list of dependencies
                self.assigned_deps.append((self.buffer[0].idx, node.dep, node.token_idx))
                #add predicted head to the child
                node.parent_idx = self.buffer[0].idx
            #sort the children?
            #self.stack[-1].node_list[-1].left_children.sort()

            #add dependency relation
            self.assigned_deps.append((self.stack[-1].node_list[-2].token_idx, self.stack[-1].node_list[-1].dep, self.buffer[0].idx))
            #add predicted head to the child
            #self.stack[-1].node_list[-1].parent_idx = self.stack[-1].node_list[-2].token_idx

            #insert into Parse State and pop from the buffer
            self.stack[-1].node_list[-1].token_idx = self.buffer[0].idx
            self.stack[-1].node_list[-1].form = self.buffer[0].form
            self.stack[-1].node_list[-1].form_idx = train_vocab.form2idx[self.buffer[0].form]
            self.stack[-1].node_list[-1].pos = self.buffer[0].upos
            self.stack[-1].node_list[-1].pos_idx = train_vocab.pos2idx[self.buffer[0].upos]
            
            self.stack[-1].node_list[-1].is_labeled_dummy_node =False
            self.stack[-1].node_list[-1].is_complete_node =True
            self.pretty_stack[-1][-1]=self.buffer[0].idx
            self.buffer.pop(0)
            return 0
        else:
            return -1



    cpdef int insert_as_head(self, Vocabulary train_vocab):
        '''
        This transition is for inserting into an UNLABLED DUMMY NODE. That is, the
        inserted node will NOT BE ASSIGNED A GOVERNOR OR A LABELED ARC.
        
        Inserts the word index at the front of the buffer into the UNLABLED Dummy Node 
        that is the terminal child of the right-spine at the top of the stack.
        
        Adds each index that is currently a left-child of that dummy node as 
        a left dependent of the inserted index.        
        '''
        cdef Node node                                                                      #is_unlabeled_dummy_node
        if len(self.buffer)>=1 and self.stack and self.stack[-1].node_list[-1].is_unlabeled_dummy_node:

            #handle children of the to-be-inserted-into node first. I.e. assign them as children of the to-be-inserted node
            for node in self.stack[-1].node_list[-1].left_children:
                #add the dependency relation to the list of dependencies
                self.assigned_deps.append((self.buffer[0].idx, node.dep, node.token_idx))
                #add predicted head to the child
                node.parent_idx = self.buffer[0].idx   
            #sort the children?
            #self.stack[-1].node_list[-1].left_children.sort()
            
            #insert into Parse State and pop from the buffer
            self.stack[-1].node_list[-1].token_idx = self.buffer[0].idx
            self.stack[-1].node_list[-1].form = self.buffer[0].form
            self.stack[-1].node_list[-1].form_idx = train_vocab.form2idx[self.buffer[0].form]
            self.stack[-1].node_list[-1].pos = self.buffer[0].upos
            self.stack[-1].node_list[-1].pos_idx = train_vocab.pos2idx[self.buffer[0].upos]
            
            self.stack[-1].node_list[-1].is_unlabeled_dummy_node =False
            self.stack[-1].node_list[-1].is_complete_node =True

            self.pretty_stack[-1] = [self.buffer[0].idx]
            self.buffer.pop(0)
            return 0
        else:
            return -1
        
        

    cpdef int right_pred(self, str label, Vocabulary train_vocab):
        '''
        Predicts that the head of the right-spine on the top of the stack has a new right-most child/terminal, which is a labled dummy node.
        I.e., resets the right-spine at the top of the stack to JUST its current head plus a LABELED dummy node as its right-most chlid.
        '''
        cdef Node node
        if self.stack and self.stack[-1].node_list[-1].is_complete_node:    
            self.stack[-1].node_list[-1].is_complete_node = False
            #for node in self.stack[-1].node_list:
            #    node.is_complete_node = False
            
            self.stack[-1].node_list = self.stack[-1].node_list[:1]
            node = Node(_token_idx = -1, _parent_idx = self.stack[-1].node_list[0].token_idx, _is_complete_node = False, _is_labeled_dummy_node = True, _is_unlabeled_dummy_node = False, _form = 'is_LDN', _form_idx = train_vocab.form2idx['is_LDN'], _pos = 'is_LDN',  _pos_idx = train_vocab.pos2idx['is_LDN'], _dep = label, _dep_idx = train_vocab.dep2idx[label],  _left_children = [])
            self.stack[-1].node_list.append(node)
            self.pretty_stack[-1] = [self.pretty_stack[-1][0],(label,[])]
            return 0
        else:
            return -1

    
    cpdef int right_comp(self, str label, Vocabulary train_vocab):
        '''
        Given that the second to top right-spine on the stack has a Labled Dummy Node as the terminal child, insert the head of the 
        topmost right-spine on the stack into that Labeled DUMMY NODE and add a new LABELED Dummy Node to the right-spine that you are inserting into
        '''
        cdef Node node
        if len(self.stack) >=2 and self.stack[-1].node_list[-1].is_complete_node and self.stack[-2].node_list[-1].is_labeled_dummy_node:
            
            #handle children of the to-be-inserted-into node first. I.e. assign them as children of the to-be-inserted node
            for node in self.stack[-2].node_list[-1].left_children:
                #add the dependency relation to the list of dependencies
                self.assigned_deps.append((self.stack[-1].node_list[0].token_idx, node.dep, node.token_idx))
                #add predicted head to the child
                node.parent_idx = self.stack[-1].node_list[0].token_idx
            #sort the children?
            #self.stack[?].node_list[?].left_children.sort()

            #insert into Parse State and pop from the stack
            self.stack[-2].node_list[-1].token_idx = self.stack[-1].node_list[0].token_idx
            self.stack[-2].node_list[-1].form = self.stack[-1].node_list[0].form
            self.stack[-2].node_list[-1].form_idx = self.stack[-1].node_list[0].form_idx
            self.stack[-2].node_list[-1].pos = self.stack[-1].node_list[0].pos
            self.stack[-2].node_list[-1].pos_idx = self.stack[-1].node_list[0].pos_idx

            self.assigned_deps.append((self.stack[-2].node_list[-1].parent_idx, self.stack[-2].node_list[-1].dep, self.stack[-1].node_list[0].token_idx))
            self.stack[-2].node_list[-1].is_labeled_dummy_node =False
            #self.stack[-1].node_list[-1].is_complete_node =False
            self.stack.pop(-1)


            #add the new Labled Dummy Node to the right spine that we are inserting into
            node = Node(_token_idx = -1, _parent_idx = self.stack[-1].node_list[-1].token_idx, _is_complete_node = False, _is_labeled_dummy_node = True, _is_unlabeled_dummy_node = False, _form = 'is_LDN', _form_idx = train_vocab.form2idx['is_LDN'],  _pos = 'is_LDN',  _pos_idx = train_vocab.pos2idx['is_LDN'], _dep = label, _dep_idx = train_vocab.dep2idx[label], _left_children = [])
            self.stack[-1].node_list.append(node)


            #Pretty Stack
            #set the terminal child of the second-to-the-top right-spine on the stack to the head of
            #the top-most right-spine on the stack
            self.pretty_stack[-2][-1] = self.pretty_stack[-1][0]     
            #add a new DUMMY NODE to the second-to-the-top right-spine on the stack
            self.pretty_stack[-2].append((label,[]))
            #pop the topmost right-spine from the stack
            self.pretty_stack.pop(-1)

            return 0

        else:
            return -1
    

    cpdef int left_pred(self, str label, Vocabulary train_vocab):
        '''
        Given that the right spine on the top of the stack is complete, construct an UNlabeled Dummy Node
        and place the head of the right spine on the top of the stack as a left child of that unlabeled Dummy Node
        '''
        cdef Node node
        #cdef RightSpine right_spine
        if self.stack and self.stack[-1].node_list[-1].is_complete_node:
            self.stack[-1].node_list[0].is_complete_node = False
            #for node in self.stack[-1].node_list:
            #    node.is_complete_node = False
            
            #add the dependency label to the head of the right spine who is now becoming a left child of the unlabeled dummy node
            self.stack[-1].node_list[0].dep = label
            #create the new node that the head becomes a left child of
            node = Node(_token_idx = -1, _parent_idx = -1, _is_complete_node = False, _is_labeled_dummy_node = False, _is_unlabeled_dummy_node = True, _form = 'is_UDN', _form_idx = train_vocab.form2idx['is_UDN'],  _pos = 'is_UDN',  _pos_idx = train_vocab.pos2idx['is_UDN'], _dep = 'not_available', _dep_idx = train_vocab.dep2idx['not_available'], _left_children = [self.stack[-1].node_list[0]])
            
            self.stack[-1].node_list = [node]
            self.pretty_stack[-1] = [(label, self.pretty_stack[-1][0])]

            return 0
        else:
            return -1

    
    cpdef int left_comp(self, str label, Vocabulary train_vocab):
        '''
        Given that the second-to-topmost right-spine on the stack has a Dummy Node as the terminal child, either LABELED or UNLABELED,
        add the head of the topmost right-spine on the stack as a left-child of that Dummy Node
        '''
        if len(self.stack)>=2 and self.stack[-1].node_list[-1].is_complete_node and (self.stack[-2].node_list[-1].is_labeled_dummy_node or self.stack[-2].node_list[-1].is_unlabeled_dummy_node):

            self.stack[-1].node_list[0].is_complete_node = False
            self.stack[-1].node_list[0].dep = label
            self.stack[-1].node_list[0].dep_idx = train_vocab.dep2idx[label]
            self.stack[-2].node_list[-1].left_children.append(self.stack[-1].node_list[0])
            
            #pretty stack
            if self.stack[-2].node_list[-1].is_labeled_dummy_node: self.pretty_stack[-2][-1][-1].append((label,self.pretty_stack[-1][0]))
            elif self.stack[-2].node_list[-1].is_unlabeled_dummy_node: self.pretty_stack[-2].append((label,self.pretty_stack[-1][0]))
            else: print("SOMETHING HAS GONE VERY WRONG WITH LEFT COMP OPERATION")

            self.stack.pop(-1)
            self.pretty_stack.pop(-1)
            return 0
        else:
            return -1
    



    '''
    Below, I implement a labeled version of the oracle from Noji and Miayo 2014/6
    '''
    cpdef str get_oracle_transition(self):
        cdef Node node
        cdef tuple tpl
        cdef Token token
        
        cdef bint top_spine_is_complete
        cdef bint top_spine_is_labeled_dummy
        cdef bint top_spine_is_unlabeled_dummy
        
        cdef bint second_top_spine_is_complete
        cdef bint second_top_spine_is_labeled_dummy
        cdef bint second_top_spine_is_unlabeled_dummy

        cdef int stack_len = len(self.stack)
        
        '''
        Transitions must alternate between SHIFT (shift, insert_as_head, insert_into_tree) and
        REDUCE (left_pred, right_pred, left_comp, right_comp) actions.
        Because we initialize the stack as empty, the first two actions will always be to: 1) shift ROOT onto the stack,
        creating a new RightSpine with ROOT as its head; and 2) Right-Pred.
        '''

        #very first action is to shift ROOT onto the stack, creating a new RightSpine with ROOT as its head
        if self.stack == []:
            return 'shift'
        #second action: predict emtpy node that will become the child of ROOT
        if (stack_len==1 
            and len(self.stack[0].node_list) ==1 
            and self.stack[0].node_list[0].token_idx ==0):
            tpl = [tpl for tpl in self.gold_deps if tpl[0] == 0][0]
            return f"right_pred({tpl[1]})"

        top_spine_is_complete = self.stack[-1].node_list[-1].is_complete_node
        top_spine_is_labeled_dummy = self.stack[-1].node_list[-1].is_labeled_dummy_node
        top_spine_is_unlabeled_dummy = self.stack[-1].node_list[-1].is_unlabeled_dummy_node

        #if terminal child of right-spine on the top of the stack is a DUMMY NODE (either labeled or unlabeled),
        #either shift, insert_as_head, or insert_into_tree
        if (top_spine_is_labeled_dummy 
            or top_spine_is_unlabeled_dummy):
            
            #if terminal child of the right-spine on the top of the stack is an UNlabeled dummy node
            #AND the front of the buffer is parent of one of the unclaimed left children, then insert_as_head
            if (top_spine_is_unlabeled_dummy 
                and [tpl for tpl in self.gold_deps if tpl[0] ==self.buffer[0].idx and tpl[2] in [node.token_idx for node in self.stack[-1].node_list[-1].left_children]]):
                return 'insert_as_head'
                
            #now considering between insert_into_tree and shift
            #if token at the front of the buffer still has at least one child in the buffer, then must shift

            #if the dependency actually exists AND no remaining children, then insert_into_tree
            if (top_spine_is_labeled_dummy 
                and [tpl for tpl in self.gold_deps if tpl[0]==self.stack[-1].node_list[-2].token_idx and tpl[2] == self.buffer[0].idx] 
                and not [tpl for tpl in self.gold_deps if tpl[0]==self.buffer[0].idx and tpl[2] in [token.idx for token in self.buffer]]):
                return 'insert_into_tree'
            
            #else, shift
            else:
                return 'shift'

        #else, perform a reduce action (left_pred, left_comp, right_pred, right_comp)
        else:

            second_top_spine_is_complete = self.stack[-2].node_list[-1].is_complete_node
            second_top_spine_is_labeled_dummy = self.stack[-2].node_list[-1].is_labeled_dummy_node
            second_top_spine_is_unlabeled_dummy = self.stack[-2].node_list[-1].is_unlabeled_dummy_node

            ###
            #right_comp
            ###
            #if the second-to-top-most right-spine on the stack is a Labeled Dummy Node
            #and the top of the stack is a complete node
            #and the head of the right-spine on the top of the stack has exactly 1 remaining child in the buffer
            #and the parent of the dummy node is the parent of the head of the right-spine on the stop of the stack
            if (second_top_spine_is_labeled_dummy 
                and top_spine_is_complete 
                and len([tpl for tpl in self.gold_deps if tpl[0] ==self.stack[-1].node_list[0].token_idx and tpl[2] in [token.idx for token in self.buffer]]) ==1 
                and [tpl for tpl in self.gold_deps if tpl[0] ==self.stack[-2].node_list[-2].token_idx and tpl[2]==self.stack[-1].node_list[0].token_idx]):
                return f'right_comp({[tpl for tpl in self.gold_deps if tpl[0] ==self.stack[-1].node_list[0].token_idx and tpl[2] in [token.idx for token in self.buffer]][0][1]})'
                                                                                                                             
                                                                                                                             
                        
            ###
            #left-comp     -> breaks out to left_pred if left_comp conditions are not satisfied because
            #these two have similiar set-up conditions
            ###
            #if the second-to-top-most right-spine on the stack is either a LABELED or an UNLABELED Dummy Node
            #AND the top of the stack is a complete node,                                                                                                     #AND the head of the right-spine at the top of the stack has no remaining children in the buffer           
            elif stack_len>=2 and (second_top_spine_is_labeled_dummy or second_top_spine_is_unlabeled_dummy) and top_spine_is_complete and not [tpl for tpl in self.gold_deps if tpl[0]==self.stack[-1].node_list[0].token_idx and tpl[2] in [token.idx for token in self.buffer]]:                                                                                                                                                                                          
                #if the second-to-top-most right spine is a labeled Dummy Node, check if the governor of the 
                #head of the right-spine at the top of the stack is governed by whoever governs the dummy node
                if second_top_spine_is_labeled_dummy and [tpl for tpl in self.gold_deps if tpl[2]==self.stack[-1].node_list[0].token_idx][0][0] in [tpl[2] for tpl in self.gold_deps if tpl[0]==self.stack[-2].node_list[-2].token_idx]:     
                    return f"left_comp({[tpl[1] for tpl in self.gold_deps if tpl[2]==self.stack[-1].node_list[0].token_idx][0]})"
                
                #if the second-to-top-most right-spine on the stack is an Unlabeled Dummy Node
                #and the head of the right-spine at the top of the stack has the same head as one 
                #of the children in the unlabeled Dummy Node
                elif second_top_spine_is_unlabeled_dummy and [tpl for tpl in self.gold_deps if tpl[2]==self.stack[-1].node_list[0].token_idx][0][0] == [tpl for tpl in self.gold_deps if tpl[2]==self.stack[-2].node_list[0].left_children[0].token_idx][0][0]:
                    return f"left_comp({[tpl[1] for tpl in self.gold_deps if tpl[2]==self.stack[-1].node_list[0].token_idx][0]})"
                else:
                    return f"left_pred({[tpl[1] for tpl in self.gold_deps if tpl[2]==self.stack[-1].node_list[0].token_idx][0]})"    
                                                                                                                                        
            ###          
            #right_pred
            ###                                                                                        
            #if the top of the stack is a complete node and the head of this right-spine still has
            #at least one remaining child in the buffer
            elif top_spine_is_complete and [tpl for tpl in self.gold_deps if tpl[0] ==self.stack[-1].node_list[0].token_idx and tpl[2] in [token.idx for token in self.buffer]]:    
                return f"right_pred({[tpl for tpl in self.gold_deps if tpl[0] ==self.stack[-1].node_list[0].token_idx and tpl[2] in [token.idx for token in self.buffer]][0][1]})"
            
                                                                                                                                                
            ###
            #left_pred
            ###
            else:
                print("inside of left_pred logical else")                                                                           
                return f"left_pred({[tpl[1] for tpl in self.gold_deps if tpl[2]==self.stack[-1].node_list[0].token_idx][0]})"



                
    cpdef void write_conll(self, file_stream):
        cdef int i
        cdef tuple dep
        '''
        Once the sentence has been parsed, use the (now populated) self.assigned_deps to write out the CoNLL-formatted parse.
        Takes as input an append-mode file stream to write the parse to
        '''
        for i in range(1, self.num_tokens+1):
            for dep in self.assigned_deps:
                if dep[2]==i:
                    #file_stream.write(f"{i}\t{self.idx2form[i]}\tlemma\tupos\txpos\tfeats\t{dep[0]}\t{dep[1]}\tdeps\tmisc\n")
                    file_stream.write(f"{i}\t{self.idx2form[i]}\t_\t_\t_\t_\t{dep[0]}\t{dep[1]}\t_\t_\n")
        file_stream.write("\n")

        
    cpdef void print_conll(self):
        cdef int i
        cdef tuple dep
        '''
        Once the sentence has been parsed, use the (now populated) self.assigned_deps to print out the ConLL-formatted parse.
        '''
        for i in range(1, self.num_tokens+1):
            for dep in self.assigned_deps:
                if dep[2]==i:
                    #print(f"{i}\t{self.idx2form[i]}\tlemma\tupos\txpos\tfeats\t{dep[0]}\t{dep[1]}\tdeps\tmisc")
                    print(f"{i}\t{self.idx2form[i]}\t_\t_\t_\t_\t{dep[0]}\t{dep[1]}\t_\t_\n")



                    
    cpdef np.ndarray[np.int32_t, ndim=1] get_features_state(self, Vocabulary train_vocab):     
        '''
        This function returns a numpy array of features extracted from the current Sentence (parse state) used for training or inference
        '''
        cdef int next_word_form_idx
        cdef int next_next_word_form_idx
        cdef np.ndarray[np.int32_t, ndim=1] buffer_feats
        cdef np.ndarray[np.int32_t, ndim=1] stack_feats
        cdef np.ndarray[np.int32_t, ndim=1] s0_feats
        cdef np.ndarray[np.int32_t, ndim=1] s1_feats
        cdef np.ndarray[np.int32_t, ndim=1] s2_feats
        cdef np.ndarray[np.int32_t, ndim=1] s3_feats
        cdef int buffer_len = len(self.buffer)
        cdef int stack_len = len(self.stack)

        #get the next word (aka the element at the front of the buffer)
        #this is an inference feature
        if buffer_len > 0:
            next_word_form_idx = train_vocab.form2idx[self.buffer[0].form]
        else:
            next_word_form_idx = train_vocab.form2idx['not_available']

        #get the next, next word and its pos
        #this is the language model target
        if buffer_len > 1:
            next_next_word_form_idx  = train_vocab.form2idx[self.buffer[1].form]
        else:
            next_next_word_form_idx = train_vocab.form2idx['not_available']

        buffer_feats = np.array([next_word_form_idx, next_next_word_form_idx], dtype = np.int32)


        
        #-----------------------------------------------------------------------------------------------    
        
        ####
        #right spine at the TOP of the stack
        ####

                        #s0_ggpl1_form_idx, s0_ggpl1_dep_idx, s0_ggpl2_form_idx, s0_ggpl2_dep_idx, s0_ggp_form_idx,  s0_ggp_dep_idx
                        #s0_gpl1_form_idx, s0_gpl1_dep_idx, s0_gpl2_form_idx, s0_gpl2_dep_idx, s0_gp_form_idx,  s0_gp_dep_idx
                        #s0_pl1_form_idx, s0_pl1_dep_idx, s0_pl2_form_idx, s0_pl2_dep_idx, s0_p_form_idx,  s0_p_dep_idx
                        #s0_ql1_form_idx, s0_ql1_dep_idx, s0_ql2_form_idx, s0_ql2_dep_idx, s0_q_form_idx,  s0_q_dep_idx

 
        if stack_len >0:
            s0_feats = self.stack[-1].get_right_spine_features_state(train_vocab)
        else:
            s0_feats = np.array([train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available']] , dtype = np.int32)



        ####
        #right spine at the SECOND to top of the stack
        ####
                        #s1_ggpl1_form_idx, s1_ggpl1_dep_idx, s1_ggpl2_form_idx, s1_ggpl2_dep_idx, s1_ggp_form_idx,  s1_ggp_dep_idx
                        #s1_gpl1_form_idx, s1_gpl1_dep_idx, s1_gpl2_form_idx, s1_gpl2_dep_idx, s1_gp_form_idx,  s1_gp_dep_idx
                        #s1_pl1_form_idx, s1_pl1_dep_idx, s1_pl2_form_idx, s1_pl2_dep_idx, s1_p_form_idx,  s1_p_dep_idx
                        #s1_ql1_form_idx, s1_ql1_dep_idx, s1_ql2_form_idx, s1_ql2_dep_idx, s1_q_form_idx,  s1_q_dep_idx
        if stack_len >1:
            s1_feats = self.stack[-2].get_right_spine_features_state(train_vocab)
        else:
            s1_feats = np.array([train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available']] , dtype = np.int32)

        ####
        #right spine at the THIRD to top of the stack
        ####
                        #s2_ggpl1_form_idx, s2_ggpl1_dep_idx, s2_ggpl2_form_idx, s2_ggpl2_dep_idx, s2_ggp_form_idx,  s2_ggp_dep_idx
                        #s2_gpl1_form_idx, s2_gpl1_dep_idx, s2_gpl2_form_idx, s2_gpl2_dep_idx, s2_gp_form_idx,  s2_gp_dep_idx
                        #s2_pl1_form_idx, s2_pl1_dep_idx, s2_pl2_form_idx, s2_pl2_dep_idx, s2_p_form_idx,  s2_p_dep_idx
                        #s2_ql1_form_idx, s2_ql1_dep_idx, s2_ql2_form_idx, s2_ql2_dep_idx, s2_q_form_idx,  s2_q_dep_idx
        if stack_len >2:
            s2_feats = self.stack[-3].get_right_spine_features_state(train_vocab)
        else:
            s2_feats = np.array([train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available']] , dtype = np.int32)

        ####
        #right spine at the fourth to top of the stack
        ####
                        #s3_ggpl1_form_idx, s3_ggpl1_dep_idx, s3_ggpl2_form_idx, s3_ggpl2_dep_idx, s3_ggp_form_idx,  s3_ggp_dep_idx
                        #s3_gpl1_form_idx, s3_gpl1_dep_idx, s3_gpl2_form_idx, s3_gpl2_dep_idx, s3_gp_form_idx,  s3_gp_dep_idx
                        #s3_pl1_form_idx, s3_pl1_dep_idx, s3_pl2_form_idx, s3_pl2_dep_idx, s3_p_form_idx,  s3_p_dep_idx
                        #s3_ql1_form_idx, s3_ql1_dep_idx, s3_ql2_form_idx, s3_ql2_dep_idx, s3_q_form_idx,  s3_q_dep_idx
        if stack_len >3:
            s3_feats = self.stack[-4].get_right_spine_features_state(train_vocab)
        else:
            s3_feats = np.array([train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available'],train_vocab.form2idx['not_available'],train_vocab.dep2idx['not_available']] , dtype = np.int32)
        
            
        feats = np.concatenate((s3_feats, s2_feats, s1_feats, s0_feats, buffer_feats))
            
        return feats
           

    
                                                                                                                             
cdef class Vocabulary:
    def __cinit__(self):

           
        self.form2idx = {'ROOT': 0, 'UNK': 1, 'is_UDN': 2, 'is_LDN': 3, 'not_available': 4}
        self.idx2form = {}

        self.pos2idx = {'ROOT': 0, 'is_UDN': 1, 'is_LDN': 2, 'not_available': 3}
        self.idx2pos = {}

        self.dep2idx = {'ROOT': 0, 'not_available': 1}
        self.idx2dep = {}

        self.transition2idx = {'shift': 0, 'insert_as_head': 1, 'insert_into_tree': 2}
        self.idx2transition = {}


    #Function to populate vocabulary
    cpdef void populate_vocabulary(self, list training_sentences):
        cdef Sentence sentence
        cdef Token token
        cdef str key
        cdef int value
        
        for sentence in training_sentences:
            for token in sentence.buffer[1:]:

                # Add form to form2idx and count occurrences
                if token.form not in self.form2idx:
                    self.form2idx[token.form] = len(self.form2idx)
                    
                # Add UPOS to pos2idx
                if token.upos not in self.pos2idx:
                    self.pos2idx[token.upos] = len(self.pos2idx)

                # Add dependency labels to dep2idx
                if token.gold_deprel not in self.dep2idx:
                    self.dep2idx[token.gold_deprel] = len(self.dep2idx)

                    # Add dependency-labeled transitions to transition2idx
                    self.transition2idx[f'left_pred({token.gold_deprel})'] = len(self.transition2idx)
                    self.transition2idx[f'right_pred({token.gold_deprel})'] = len(self.transition2idx)
                    self.transition2idx[f'left_comp({token.gold_deprel})'] = len(self.transition2idx)
                    self.transition2idx[f'right_comp({token.gold_deprel})'] = len(self.transition2idx)

        # Populate reverse dictionaries
        for key, value in self.form2idx.items():
            self.idx2form[value] = key
        
        for key, value in self.pos2idx.items():
            self.idx2pos[value] = key

        for key, value in self.dep2idx.items():
            self.idx2dep[value] = key

        for key, value in self.transition2idx.items():
            self.idx2transition[value] = key





    cpdef void unkify(self, list sentences):
        '''
        This function is called after the Vocabulary attributes are populated with `populate_vocabulary()`.
        Takes a list of Sentence objects, goes through each of the Tokens, and unkify's the .form if the form
        was not observed in the training data
        ''' 
        cdef Sentence sentence
        cdef Token token

        # Iterate over each sentence in the list of sentences
        for sentence in sentences:
            for token in sentence.buffer[1:]:

                if not token.form in self.form2idx:
                    token.form = "UNK"


            

    