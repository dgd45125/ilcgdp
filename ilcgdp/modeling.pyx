import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
cimport numpy as np
from scipy import stats

import time

import random

from parser cimport Token, Node, Sentence, Vocabulary, RightSpine

import operator



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



#Create a Pytorch Dataset object for the local transition data
class LocalDataset(Dataset):
    def __init__(self,np.ndarray[np.int32_t, ndim=2] transition_data):
        self.transition_data = transition_data
    
    def __len__(self):
        return len(self.transition_data)
    
    def __getitem__(self, idx):
        
        item = self.transition_data[idx]
        return item



# get a DataLoader, given an input 2-dimensional numpy array and a batch size
def get_data_loader(np.ndarray[np.int32_t, ndim=2] data, int batch_size):

    return DataLoader(LocalDataset(data),batch_size = batch_size, shuffle=True)





#neural net for locally-normalized training and inference
class model(nn.Module):
    def __init__(self, 
                 num_form_embeddings, form_embedding_size, 
                 num_dep_embeddings, dep_embedding_size,
                 node_size,
                 spine_size,
                 hidden_size, 
                 num_transitions):
        super(model, self).__init__()


        self.form_embeddings = nn.Embedding(num_form_embeddings, form_embedding_size)
        self.dep_embeddings = nn.Embedding(num_dep_embeddings, dep_embedding_size)
        
        self.activation = torch.nn.ReLU()

        self.node_former = nn.Linear(3*form_embedding_size + 3* dep_embedding_size, node_size)
        self.node_former_dropout = nn.Dropout(p= 0.2)

        self.spine_former = nn.Linear(4*node_size, spine_size)
        self.spine_former_dropout = nn.Dropout(p=0.2)
        
        self.fc1 = nn.Linear(4*spine_size + form_embedding_size, hidden_size)
        self.fc1_dropout = nn.Dropout(p=0.2)
         
        self.fc1_to_transitions = nn.Linear(hidden_size, num_transitions)
        self.fc1_to_vocabulary = nn.Linear(hidden_size, num_form_embeddings)

    def forward(self, batch):


        s3_ggpl1_form = self.form_embeddings(batch[:,0])
        s3_ggpl1_dep = self.dep_embeddings(batch[:,1])
        s3_ggpl2_form = self.form_embeddings(batch[:,2])
        s3_ggpl2_dep = self.dep_embeddings(batch[:,3])
        s3_ggp_form = self.form_embeddings(batch[:,4])
        s3_ggp_dep = self.dep_embeddings(batch[:,5])
        s3ggp = self.node_former_dropout(self.activation(self.node_former(torch.cat((s3_ggpl1_form,s3_ggpl1_dep,s3_ggpl2_form,s3_ggpl2_dep,s3_ggp_form,s3_ggp_dep),dim=1))))

        s3_gpl1_form = self.form_embeddings(batch[:,6])
        s3_gpl1_dep = self.dep_embeddings(batch[:,7])
        s3_gpl2_form = self.form_embeddings(batch[:,8])
        s3_gpl2_dep = self.dep_embeddings(batch[:,9])
        s3_gp_form = self.form_embeddings(batch[:,10])
        s3_gp_dep = self.dep_embeddings(batch[:,11])
        s3gp = self.node_former_dropout(self.activation(self.node_former(torch.cat((s3_gpl1_form,s3_gpl1_dep,s3_gpl2_form,s3_gpl2_dep,s3_gp_form,s3_gp_dep),dim=1))))

        s3_pl1_form = self.form_embeddings(batch[:,12])
        s3_pl1_dep = self.dep_embeddings(batch[:,13])
        s3_pl2_form = self.form_embeddings(batch[:,14])
        s3_pl2_dep = self.dep_embeddings(batch[:,15])
        s3_p_form = self.form_embeddings(batch[:,16])
        s3_p_dep = self.dep_embeddings(batch[:,17])
        s3p = self.node_former_dropout(self.activation(self.node_former(torch.cat((s3_pl1_form,s3_pl1_dep,s3_pl2_form,s3_pl2_dep,s3_p_form,s3_p_dep),dim=1))))

        s3_ql1_form = self.form_embeddings(batch[:,18])
        s3_ql1_dep = self.dep_embeddings(batch[:,19])
        s3_ql2_form = self.form_embeddings(batch[:,20])
        s3_ql2_dep = self.dep_embeddings(batch[:,21])
        s3_q_form = self.form_embeddings(batch[:,22])
        s3_q_dep = self.dep_embeddings(batch[:,23])
        s3q = self.node_former_dropout(self.activation(self.node_former(torch.cat((s3_ql1_form,s3_ql1_dep,s3_ql2_form,s3_ql2_dep,s3_q_form,s3_q_dep),dim=1))))

        
        s3 = self.spine_former_dropout(self.activation(self.spine_former(torch.cat((s3ggp, s3gp, s3p, s3q),dim=1))))



        s2_ggpl1_form = self.form_embeddings(batch[:,24])
        s2_ggpl1_dep = self.dep_embeddings(batch[:,25])
        s2_ggpl2_form = self.form_embeddings(batch[:,26])
        s2_ggpl2_dep = self.dep_embeddings(batch[:,27])
        s2_ggp_form = self.form_embeddings(batch[:,28])
        s2_ggp_dep = self.dep_embeddings(batch[:,29])
        s2ggp = self.node_former_dropout(self.activation(self.node_former(torch.cat((s2_ggpl1_form,s2_ggpl1_dep,s2_ggpl2_form,s2_ggpl2_dep,s2_ggp_form,s2_ggp_dep),dim=1))))

        s2_gpl1_form = self.form_embeddings(batch[:,30])
        s2_gpl1_dep = self.dep_embeddings(batch[:,31])
        s2_gpl2_form = self.form_embeddings(batch[:,32])
        s2_gpl2_dep = self.dep_embeddings(batch[:,33])
        s2_gp_form = self.form_embeddings(batch[:,34])
        s2_gp_dep = self.dep_embeddings(batch[:,35])
        s2gp = self.node_former_dropout(self.activation(self.node_former(torch.cat((s2_gpl1_form,s2_gpl1_dep,s2_gpl2_form,s2_gpl2_dep,s2_gp_form,s2_gp_dep),dim=1))))

        s2_pl1_form = self.form_embeddings(batch[:,36])
        s2_pl1_dep = self.dep_embeddings(batch[:,37])
        s2_pl2_form = self.form_embeddings(batch[:,38])
        s2_pl2_dep = self.dep_embeddings(batch[:,39])
        s2_p_form = self.form_embeddings(batch[:,40])
        s2_p_dep = self.dep_embeddings(batch[:,41])
        s2p = self.node_former_dropout(self.activation(self.node_former(torch.cat((s2_pl1_form,s2_pl1_dep,s2_pl2_form,s2_pl2_dep,s2_p_form,s2_p_dep),dim=1))))

        s2_ql1_form = self.form_embeddings(batch[:,42])
        s2_ql1_dep = self.dep_embeddings(batch[:,43])
        s2_ql2_form = self.form_embeddings(batch[:,44])
        s2_ql2_dep = self.dep_embeddings(batch[:,45])
        s2_q_form = self.form_embeddings(batch[:,46])
        s2_q_dep = self.dep_embeddings(batch[:,47])
        s2q = self.node_former_dropout(self.activation(self.node_former(torch.cat((s2_ql1_form,s2_ql1_dep,s2_ql2_form,s2_ql2_dep,s2_q_form,s2_q_dep),dim=1))))

        
        s2 = self.spine_former_dropout(self.activation(self.spine_former(torch.cat((s2ggp, s2gp, s2p, s2q),dim=1))))


        s1_ggpl1_form = self.form_embeddings(batch[:,48])
        s1_ggpl1_dep = self.dep_embeddings(batch[:,49])
        s1_ggpl2_form = self.form_embeddings(batch[:,50])
        s1_ggpl2_dep = self.dep_embeddings(batch[:,51])
        s1_ggp_form = self.form_embeddings(batch[:,52])
        s1_ggp_dep = self.dep_embeddings(batch[:,53])
        s1ggp = self.node_former_dropout(self.activation(self.node_former(torch.cat((s1_ggpl1_form,s1_ggpl1_dep,s1_ggpl2_form,s1_ggpl2_dep,s1_ggp_form,s1_ggp_dep),dim=1))))

        s1_gpl1_form = self.form_embeddings(batch[:,54])
        s1_gpl1_dep = self.dep_embeddings(batch[:,55])
        s1_gpl2_form = self.form_embeddings(batch[:,56])
        s1_gpl2_dep = self.dep_embeddings(batch[:,57])
        s1_gp_form = self.form_embeddings(batch[:,58])
        s1_gp_dep = self.dep_embeddings(batch[:,59])
        s1gp = self.node_former_dropout(self.activation(self.node_former(torch.cat((s1_gpl1_form,s1_gpl1_dep,s1_gpl2_form,s1_gpl2_dep,s1_gp_form,s1_gp_dep),dim=1))))

        s1_pl1_form = self.form_embeddings(batch[:,60])
        s1_pl1_dep = self.dep_embeddings(batch[:,61])
        s1_pl2_form = self.form_embeddings(batch[:,62])
        s1_pl2_dep = self.dep_embeddings(batch[:,63])
        s1_p_form = self.form_embeddings(batch[:,64])
        s1_p_dep = self.dep_embeddings(batch[:,65])
        s1p = self.node_former_dropout(self.activation(self.node_former(torch.cat((s1_pl1_form,s1_pl1_dep,s1_pl2_form,s1_pl2_dep,s1_p_form,s1_p_dep),dim=1))))

        s1_ql1_form = self.form_embeddings(batch[:,66])
        s1_ql1_dep = self.dep_embeddings(batch[:,67])
        s1_ql2_form = self.form_embeddings(batch[:,68])
        s1_ql2_dep = self.dep_embeddings(batch[:,69])
        s1_q_form = self.form_embeddings(batch[:,70])
        s1_q_dep = self.dep_embeddings(batch[:,71])
        s1q = self.node_former_dropout(self.activation(self.node_former(torch.cat((s1_ql1_form,s1_ql1_dep,s1_ql2_form,s1_ql2_dep,s1_q_form,s1_q_dep),dim=1))))

        
        s1 = self.spine_former_dropout(self.activation(self.spine_former(torch.cat((s1ggp, s1gp, s1p, s1q),dim=1))))

        
        s0_ggpl1_form = self.form_embeddings(batch[:,72])
        s0_ggpl1_dep = self.dep_embeddings(batch[:,73])
        s0_ggpl2_form = self.form_embeddings(batch[:,74])
        s0_ggpl2_dep = self.dep_embeddings(batch[:,75])
        s0_ggp_form = self.form_embeddings(batch[:,76])
        s0_ggp_dep = self.dep_embeddings(batch[:,77])
        s0ggp = self.node_former_dropout(self.activation(self.node_former(torch.cat((s0_ggpl1_form,s0_ggpl1_dep,s0_ggpl2_form,s0_ggpl2_dep,s0_ggp_form,s0_ggp_dep),dim=1))))

        s0_gpl1_form = self.form_embeddings(batch[:,78])
        s0_gpl1_dep = self.dep_embeddings(batch[:,79])
        s0_gpl2_form = self.form_embeddings(batch[:,80])
        s0_gpl2_dep = self.dep_embeddings(batch[:,81])
        s0_gp_form = self.form_embeddings(batch[:,82])
        s0_gp_dep = self.dep_embeddings(batch[:,83])
        s0gp = self.node_former_dropout(self.activation(self.node_former(torch.cat((s0_gpl1_form,s0_gpl1_dep,s0_gpl2_form,s0_gpl2_dep,s0_gp_form,s0_gp_dep),dim=1))))

        s0_pl1_form = self.form_embeddings(batch[:,84])
        s0_pl1_dep = self.dep_embeddings(batch[:,85])
        s0_pl2_form = self.form_embeddings(batch[:,86])
        s0_pl2_dep = self.dep_embeddings(batch[:,87])
        s0_p_form = self.form_embeddings(batch[:,88])
        s0_p_dep = self.dep_embeddings(batch[:,89])
        s0p = self.node_former_dropout(self.activation(self.node_former(torch.cat((s0_pl1_form,s0_pl1_dep,s0_pl2_form,s0_pl2_dep,s0_p_form,s0_p_dep),dim=1))))

        s0_ql1_form = self.form_embeddings(batch[:,90])
        s0_ql1_dep = self.dep_embeddings(batch[:,91])
        s0_ql2_form = self.form_embeddings(batch[:,92])
        s0_ql2_dep = self.dep_embeddings(batch[:,93])
        s0_q_form = self.form_embeddings(batch[:,94])
        s0_q_dep = self.dep_embeddings(batch[:,95])
        s0q = self.node_former_dropout(self.activation(self.node_former(torch.cat((s0_ql1_form,s0_ql1_dep,s0_ql2_form,s0_ql2_dep,s0_q_form,s0_q_dep),dim=1))))

        
        s0 = self.spine_former_dropout(self.activation(self.spine_former(torch.cat((s0ggp, s0gp, s0p, s0q),dim=1))))

        
        next_word_form = self.form_embeddings(batch[:,96])
   

        representation = self.fc1_dropout(self.activation(self.fc1(torch.cat((s3, s2,s1,s0,next_word_form),dim=1))))

        return self.fc1_to_transitions(representation), self.fc1_to_vocabulary(representation) 







    
cpdef void train_model(_model, _train_dataloader, _dev_dataloader, int _num_epochs, bint _save_all_models, str _model_save_location):
    '''
    Trains a pytorch model for locally-normalized transition prediction and language modeling
    '''   
    
    cdef int epoch, i, dev_transition_correct, dev_transition_total
    cdef double epoch_train_transition_loss, epoch_train_vocab_loss, epoch_train_combined_loss, epoch_dev_transition_loss, epoch_dev_vocab_loss, epoch_dev_combined_loss, dev_accuracy = 0.0, best_dev_accuracy = 0.0, best_dev_transition_loss = float('inf'),  best_dev_combined_loss = float('inf'), best_dev_vocab_loss = float('inf')
    cdef bint already_saved_model_this_epoch
 

    _model.to(device)
    _model.train()

    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(_model.parameters(), lr = 0.001, weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_num_epochs)

    tic = time.perf_counter()
    for epoch in range(_num_epochs):

        epoch_train_transition_loss = 0.0
        epoch_train_vocab_loss = 0.0
        epoch_train_combined_loss = 0.0

        
        log_file = open(f'{_model_save_location}/train_log.txt','a')
    
        for i, (batch) in enumerate(_train_dataloader):

            optimizer.zero_grad()
            
            transitions = batch[:,-1].type(torch.LongTensor)
        
            #first, compute loss for TRANSITIONS    
            transition_output, vocab_output = _model(batch.to(device)) 
            transition_loss = criterion(transition_output, transitions.to(device))
            epoch_train_transition_loss += transition_loss.item()
        
            #then, compute the loss for predicting next word
            #BUT ONLY FOR SHIFT (GEN) TRANSITIONS
            
            #calculate a mask that will identify only the SHIFT transitions (transition embedding index <-> 0,1,2)
            shift_mask = np.where((batch[:,-1]==0) | (batch[:,-1]==1) | (batch[:,-1]==2))
            vocab_loss = criterion(vocab_output[shift_mask], batch[:,-2].type(torch.LongTensor)[shift_mask].to(device))
            epoch_train_vocab_loss += vocab_loss.item()
   
            
            #Sum the losses, backpropogate, and update the parameters
            loss = transition_loss + vocab_loss
            epoch_train_combined_loss += loss.item()
            loss.backward()
            optimizer.step()

            
        print (f'Training Epoch: [{epoch+1}/{_num_epochs}], Training Transition Loss: {round(epoch_train_transition_loss,3)}, Vocab Loss: {round(epoch_train_vocab_loss,3)}, Combined Loss: {round(epoch_train_combined_loss,3)}')
        log_file.write(f'Training Epoch: [{epoch+1}/{_num_epochs}], Training Transition Loss: {round(epoch_train_transition_loss,3)}, Vocab Loss: {round(epoch_train_vocab_loss,3)}, Combined Loss: {round(epoch_train_combined_loss,3)}\n\n')
        print()



        #eval model on dev set 
        dev_transition_correct = 0
        dev_transition_total = 0

        epoch_dev_transition_loss = 0.0
        epoch_dev_vocab_loss = 0.0
        epoch_dev_combined_loss = 0.0
        
        _model.eval()
        with torch.no_grad():
            for i, (batch) in enumerate(_dev_dataloader):
                
                transitions = batch[:,-1].type(torch.LongTensor)
                
                transition_output, vocab_output = _model(batch.to(device))
                dev_transition_loss = criterion(transition_output, transitions.to(device))
                epoch_dev_transition_loss += dev_transition_loss.item()
                
                _, transition_predicted = torch.max(transition_output.data, 1)
                dev_transition_total += transitions.size(0)
                dev_transition_correct += (transition_predicted == transitions.to(device)).sum()
                dev_accuracy = 100*dev_transition_correct/dev_transition_total
                
                shift_mask = np.where((batch[:,-1]==0) | (batch[:,-1]==1) | (batch[:,-1]==2))
                dev_vocab_loss = criterion(vocab_output[shift_mask], batch[:,-2].type(torch.LongTensor)[shift_mask].to(device))
                epoch_dev_vocab_loss += dev_vocab_loss.item()

                dev_loss = dev_transition_loss + dev_vocab_loss
                epoch_dev_combined_loss += dev_loss.item()
        
            print(f"Development Transition accuracy: {round(dev_accuracy,3)}")
            log_file.write(f"Development Transition accuracy: {round(dev_accuracy,3)}\n")
            print(f"Development Transition Loss: {round(epoch_dev_transition_loss,3)}, Dev Vocab Loss: {round(epoch_dev_vocab_loss,3)}, Combined Loss: {round(epoch_dev_combined_loss,3)}\n")
            log_file.write(f"Development Transition Loss: {round(epoch_dev_transition_loss,3)}, Dev Vocab Loss: {round(epoch_dev_vocab_loss,3)},, Combined Loss: {round(epoch_dev_combined_loss,3)}\n\n")

            
        _model.train()
        #scheduler.step()

        if _save_all_models:
            torch.save(_model.state_dict(), f'{_model_save_location}/{epoch+1}.pth')

        else:
            already_saved_model_this_epoch = False
            
            if (epoch_dev_transition_loss < best_dev_transition_loss):
                print(f"Saving model because epoch_dev_transition_loss {round(epoch_dev_transition_loss,3)} is less than best_dev_transition_loss: {round(best_dev_transition_loss,3)}")
                log_file.write(f"Saving model because epoch_dev_transition_loss {round(epoch_dev_transition_loss,3)} is less than best_dev_transition_loss: {round(best_dev_transition_loss,3)}\n")
                best_dev_transition_loss = epoch_dev_transition_loss
                torch.save(_model.state_dict(), f'{_model_save_location}/{epoch+1}.pth')
                already_saved_model_this_epoch = True
                      
            if (dev_accuracy > best_dev_accuracy):
                print(f"Saving model because dev_accuracy {round(dev_accuracy,3)} is better than best_dev_accuracy: {round(best_dev_accuracy,3)}")
                log_file.write(f"Saving model because dev_accuracy {round(dev_accuracy,3)} is better than best_dev_accuracy: {round(best_dev_accuracy,3)}\n")
                best_dev_accuracy = dev_accuracy
                if not already_saved_model_this_epoch:
                    torch.save(_model.state_dict(), f'{_model_save_location}/{epoch+1}.pth')
                    already_saved_model_this_epoch = True
                      
            if (epoch_dev_vocab_loss < best_dev_vocab_loss):
                print(f"Saving model because epoch_dev_vocab_loss {round(epoch_dev_vocab_loss,3)} is less than best_dev_vocab_loss: {round(best_dev_vocab_loss,3)}")
                log_file.write(f"Saving model because epoch_dev_vocab_loss {round(epoch_dev_vocab_loss,3)} is less than best_dev_vocab_loss: {round(best_dev_vocab_loss,3)}\n")
                best_dev_vocab_loss = epoch_dev_vocab_loss
                if not already_saved_model_this_epoch:
                    torch.save(_model.state_dict(), f'{_model_save_location}/{epoch+1}.pth')
                    already_saved_model_this_epoch = True


                
            if (epoch_dev_combined_loss < best_dev_combined_loss):
                print(f"Saving model because epoch_dev_combined_loss {round(epoch_dev_combined_loss,3)} is less than best_dev_combined_loss: {round(best_dev_combined_loss,3)}\n")
                log_file.write(f"Saving model because epoch_dev_combined_loss {round(epoch_dev_combined_loss,3)} is less than best_dev_combined_loss: {round(best_dev_combined_loss,3)}\n")
                best_dev_combined_loss = epoch_dev_combined_loss
                torch.save(_model.state_dict(), f'{_model_save_location}/{epoch+1}.pth')
                if not already_saved_model_this_epoch:
                    torch.save(_model.state_dict(), f'{_model_save_location}/{epoch+1}.pth')
                    already_saved_model_this_epoch = True

                
        toc = time.perf_counter()
        print(f"\nTotal training time: {toc-tic:0.4f} seconds")
        log_file.write(f"\nTotal training time: {toc-tic:0.4f} seconds\n")
        print('-----------------------------------------------------------------\n')
        log_file.write('-----------------------------------------------------------------\n\n')
        
        log_file.close()
            





cpdef void complete_tree(Sentence sentence):
    '''
    This function takes a Sentence (parse state) as input and, if the tree is not complete, the tree will
    trivially be completed by assigning unclaimed elements to ROOT.
    Also, if there are any remaining indices on the Buffer, they are assigned to ROOT.
    '''
    cdef int i
    cdef Token token
    cdef Node left_child    

    for i in range(len(sentence.stack)):

        #if this is a complete subtree, attach it to ROOT
        if sentence.stack[i].node_list[-1].is_complete_node:
            sentence.assigned_deps.append((0,'root',sentence.stack[i].node_list[0].token_idx))

        #else, we know that the terminal of this subtree is either a Labeled or Unlabeled Dummy Node
        else:
  
            #if is labeled dummy node
            if sentence.stack[i].node_list[-1].is_labeled_dummy_node:
                
                #if it has children, assign them to the head of the labeled DN
                for left_child in sentence.stack[i].node_list[-1].left_children:
                    sentence.assigned_deps.append((sentence.stack[i].node_list[-1].parent_idx,'lc', left_child.token_idx))

                #attach head of this right spine to ROOT
                sentence.assigned_deps.append((0, 'root', sentence.stack[i].node_list[0].token_idx))

            #else, is unlabeled dummy node
            #add each unclaiimed child to ROOT
            else:
                for left_child in sentence.stack[i].node_list[-1].left_children:
                    sentence.assigned_deps.append((0, 'root', left_child.token_idx))


    #handle remaining Tokens on the buffer
    if sentence.buffer:
        for token in sentence.buffer:
            sentence.assigned_deps.append((0, 'root',token.idx))




            
        


cpdef tuple get_transition_and_vocab_logprobs(Sentence sentence, classifier, Vocabulary train_vocab):
    cdef np.ndarray[np.int32_t, ndim=1] feats
    cdef dict transition_logprob_dict = {}
    cdef str transition
    cdef list valid_unlabeled_transitions, valid_transitions, valid_transition_indices
    '''
    This function takes as input a single Sentence (parse state), a trained classifier, and the training
    Vocabulary and returns a dictionary containing the log probabilities over valid transitions and the
    log probability of each vocab item
    '''
    
    feats = sentence.get_features_state(train_vocab)
    torch_feats = torch.from_numpy(feats).to(device)
    torch_feats = torch.unsqueeze(torch_feats,0)
    
    #outputs   
    transition_output, vocab_output = classifier(torch_feats) 

    #raw activations (logits) over transition labels: (shift=0, ..., see train_vocab.transition2idx)
    transition_activations = transition_output[0]
    #raw activations over vocabulary labels
    vocab_activations = vocab_output[0]

    #get the valid transitions for this partial parse state
    valid_unlabeled_transitions, valid_transitions, valid_transition_indices = sentence.get_valid_transitions(train_vocab)

    ####
    #take the log softmax only over valid transitions
    ####
    for log_prob, transition in zip(torch.nn.functional.log_softmax(transition_activations[valid_transition_indices],dim=0),valid_transitions):
        transition_logprob_dict[transition] = log_prob.item()

    
    #take the log softmax over vocab activations
    vocab_log_probs = torch.nn.functional.log_softmax(vocab_activations, dim= 0)

    return transition_logprob_dict, vocab_log_probs
        







    
cpdef list beam_step(list beam, classifier, Vocabulary train_vocab, int beam_size, bint include_emission_prob):
    '''
    This function takes as input a beam populated with Sentences (parse states), a trained classifier, the training Vocabulary,
    and a beam size, and an argument for specifying whether the word emission probability should be included or not. 
    It returns a new, log probability-sorted beam of size beam_size
    '''
    cdef list new_beam = []
    cdef Sentence sentence, new_sentence
    cdef dict transition_logprob_dict
    cdef str action, label, transition
    cdef int valid_transition = 0, i
    
    for sentence in beam:
        
        transition_logprob_dict, vocab_log_probs = get_transition_and_vocab_logprobs(sentence, classifier, train_vocab)

        if len(sentence.buffer) >1:  
            vocab_logprob = vocab_log_probs[train_vocab.form2idx[sentence.buffer[1].form]] 
        else:
            vocab_logprob = 0

        for transition, i  in zip(transition_logprob_dict, range(1,len(transition_logprob_dict)+1)):
            action = transition.split("(")[0]
            label = transition.split("(")[-1][:-1]

            #make a deep copy of the sentence
            new_sentence = sentence.copy()
            
            #add to its identifier so that we can identify its parent, later
            #new_sentence.identifier+=":"+str(i)
            
            #record this transition, for later
            new_sentence.most_recent_transition = transition
            
            #transition log prob
            new_sentence.log_probability += transition_logprob_dict[transition]

            #save vocab_logprob to the sentence item itself, for later
            new_sentence.most_recent_word_gen_log_prob = vocab_logprob

            if action =='shift':
                if include_emission_prob: new_sentence.log_probability += vocab_logprob
                valid_transition = new_sentence.shift(train_vocab)
            if action =='insert_as_head':
                if include_emission_prob: new_sentence.log_probability += vocab_logprob
                valid_transition = new_sentence.insert_as_head(train_vocab)
            if action == 'insert_into_tree':
                if include_emission_prob: new_sentence.log_probability += vocab_logprob
                valid_transition = new_sentence.insert_into_tree(train_vocab)

            if action =='right_comp':
                valid_transition = new_sentence.right_comp(label, train_vocab)
            if action =='left_comp':
                valid_transition = new_sentence.left_comp(label, train_vocab)
            if action =='right_pred':
                valid_transition = new_sentence.right_pred(label, train_vocab)
            if action =='left_pred':
                valid_transition = new_sentence.left_pred(label, train_vocab)  
                
            if valid_transition == 0:
                new_beam.append(new_sentence)


    #sort the beam
    new_beam = sorted(new_beam, key = operator.attrgetter("log_probability"),reverse=True)
    #cull the beam
    new_beam = new_beam[:beam_size]
    
    return new_beam

        
        
cpdef double logsumexp(list log_probs):
    '''
    computes the logarithm of the sum of the exponentials, without causing
    numerical underflow.
    '''
    cdef double x, m
    cdef list exps = []    
    
    m = max(log_probs)
    if m == -np.inf:
        return -np.inf

    for x in log_probs:
        exps.append(np.exp(x - m))

    return m + np.log(sum(exps))  
    #return m + np.log(sum(np.exp(x - m) for x in log_probs))

    
cpdef list normalize_log_probs(list log_probs):
    '''
    converts a list of log probabilities into a probability distribution
    '''
    cdef np.ndarray[double, ndim=1] np_log_probs, exp_shifted
    cdef double max_log_prob

    np_log_probs = np.array(log_probs)
    max_log_prob = np.max(np_log_probs)
    
    exp_shifted = np.exp(np_log_probs - max_log_prob)

    return list(exp_shifted / np.sum(exp_shifted))


        
cpdef Sentence beam_parse(Sentence sentence, classifier, Vocabulary train_vocab, int beam_size, bint verbose, str stat_file):
    '''
    This function takes a Sentence object (parse state) to parse, a trained classifier, the training Vocabulary,
    a beam size, a verbosity argument, and a file name to write statistics to,
    and performs beam search. If verbose is True, prints out state and probability metrics for parses on the beam.
    
    Returns a Sentence object corresponding to either the highest ranked completed parse, or the highest ranked incomplete parse, 
    after post-processing to complete the tree
    '''
    cdef list beam, out_words = [], out_Log_Prefix_Probs = [], out_Surprisals = [], normalized_log_probabilities = []#, out_Surprisals_underflow = [], out_Prefix_probs_underflow = []
    cdef int num_transitions = 0, i = 0
    cdef Sentence parse
    cdef double log_prefix_probability, surp, log_prefix_prob
    cdef str word
    

    if verbose: 
        print(f"Num Transitions: 0")
        sentence.print_state()
        print(f"Parse log Probability: {sentence.log_probability}")#, probability: {np.exp(sentence.log_probability)}")     
        print('-----------------------------------------------------')


    beam = [sentence]
    num_transitions = 0

    while True:

        #take a regular beam step, including generating the next word, sorting by probability, and culling back to beam_size
        beam = beam_step(beam, classifier, train_vocab, beam_size, include_emission_prob = True)
        
        num_transitions +=1
        if verbose: print(f"\n\nNum Transitions: {num_transitions}")

        #beam info
        log_prefix_probability = logsumexp([parse.log_probability for parse in beam])
        normalized_log_probabilities = normalize_log_probs([parse.log_probability for parse in beam])
        
        #beam_probs = [val.item() for val in np.exp([parse.log_probability for parse in beam])]
        #beam_prob_sum = sum(beam_probs)+ .0000000000000000000000000000000001    #prefix probability
        #beam_prob_distribution = [round(x / beam_prob_sum,4) for x in beam_probs]


        #if verbose: print(f"Beam Probabilities: {beam_probs}")
        #if verbose: print(f"Sum Beam Probabilities (Prefix Probability): {beam_prob_sum}")
        #if verbose: print(f"Beam Probability Distribution: {beam_prob_distribution}")
        #if verbose: print(f"Sum Beam Probability Distribution: {sum(beam_prob_distribution)}")

        if verbose: print(f"Log Prefix Probability from LogSumExp: {log_prefix_probability}")
        if verbose: print(f"Normalized Log Probabilities: {normalized_log_probabilities}")


        #for parse in beam:
        for i in range(len(beam)):

            parse = beam[i]
            
            if verbose: 
                parse.print_state()
                print(f"Transition: {parse.most_recent_transition}, log Probability: {round(parse.log_probability,4)}")#, normalized log probability: {round(normalized_log_probabilities[i],4)}, probability: {round(np.exp(parse.log_probability),4)}, normalized probability: {round(np.exp(parse.log_probability)/beam_prob_sum,4)}")
                print('-----------------------------------------------------')

                
            if parse.is_final_configuration():      

                if stat_file:
                    out_stream = open(stat_file,'a')
                    for word, log_prefix_prob, surp in zip(out_words, out_Log_Prefix_Probs, out_Surprisals):
                        out_stream.write(f"{word}\tLogPrefixProbability:{log_prefix_prob}\tSurp:{surp}\n")
                    #for word, log_prefix_prob, surp, PPu, Su in zip(out_words, out_Log_Prefix_Probs, out_Surprisals, out_Prefix_probs_underflow, out_Surprisals_underflow):
                        #out_stream.write(f"{word}\tLogPrefixProbability:{log_prefix_prob}\tSurp:{surp}\tPPu:{PPu}\tSu:{Su}\n")
                    out_stream.write("\n")
                    out_stream.close()
                
                return parse


        if len(beam[0].buffer) ==0:
            print("COMPLETING THE TREE FOR THE FIRST PARSE ON THE BEAM")
            complete_tree(beam[0])         

            if stat_file:
                out_stream = open(stat_file,'a')
                for word, log_prefix_prob, surp in zip(out_words, out_Log_Prefix_Probs, out_Surprisals):
                    out_stream.write(f"{word}\tLogPrefixProbability:{log_prefix_prob}\tSurp:{surp}\n")
                #for word, log_prefix_prob, surp, PPu, Su in zip(out_words, out_Log_Prefix_Probs, out_Surprisals, out_Prefix_probs_underflow, out_Surprisals_underflow):
                    #out_stream.write(f"{word}\tLogPrefixProbability:{log_prefix_prob}\tSurp:{surp}\tPPu:{PPu}\tSu:{Su}\n")
                out_stream.write("\n")
                out_stream.close()

            return beam[0]

        #we have just performed a SHIFT-type transition and generated the word at
        #the front of the buffer, so record post-word-generation statistics
        if num_transitions % 2 == 1 and stat_file:

            if num_transitions ==1:
                out_Surprisals.append(  (np.log(1.0) - log_prefix_probability) / np.log(2) ) #convert nats to bits
                #out_Surprisals_underflow.append(   np.log2(   1.0 / beam_prob_sum)     )
            else:
                out_Surprisals.append( (out_Log_Prefix_Probs[-1] - log_prefix_probability) / np.log(2) ) #convert nats to bits
                #out_Surprisals_underflow.append(    np.log2   (  out_Prefix_probs_underflow[-1] / beam_prob_sum   )    )
            
            out_Log_Prefix_Probs.append(log_prefix_probability)
            #out_Prefix_probs_underflow.append(beam_prob_sum)
            
            if len(beam[0].buffer) > 0: 
                out_words.append(beam[0].buffer[0].form)
            else: 
                out_words.append('EOS')

