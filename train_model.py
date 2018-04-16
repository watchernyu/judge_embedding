# Phu, Andrea and Watcher
# 2018 Spring

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch import FloatTensor, LongTensor

import numpy as np
import pandas as pd
import time
import os
import pickle
import string
import torch.utils.data as data_utils
import psutil

torch.manual_seed(1)

def dump_general_data(somedata,processed_data_path,save_filename):
    with open(os.path.join(processed_data_path,save_filename),"wb") as f:  
        pickle.dump(somedata, f)

def load_general_data(processed_data_path,save_filename):
    with open(os.path.join(processed_data_path,save_filename),"rb") as f:  
        return pickle.load(f)


class Judge_emb_model(nn.Module):
    def __init__(self, input_dim, hidden_layer_dim, embedding_dim, num_judges):
        super(Judge_emb_model,self).__init__()
        # input is m x D
        self.linear1 = nn.Linear(input_dim,hidden_layer_dim) # D x H 
        self.dropout1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(hidden_layer_dim,hidden_layer_dim) # H x H
        self.dropout2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(hidden_layer_dim,embedding_dim)
        self.dropout3 = nn.Dropout(p=0.5)
        
        self.judge_embedding = nn.Linear(embedding_dim,num_judges) # H x J
        # the output is m x J
        
        self.init_weights()

    def forward(self, X):
        out = F.relu(self.linear1(X))
        out = self.dropout1(out)
        out = F.relu(self.linear2(out))
        out = self.dropout2(out)
        out = F.relu(self.linear3(out))
        out = self.judge_embedding(out)
        
        # now we have m x J matrix, for m data points, we can do log softmax
        log_prob = F.log_softmax(out,dim=1)
        return log_prob # for each opinion data, this is probability of which judge writes this opinion
    
    def init_weights(self):
        linear_layers = [self.linear1,self.linear2,self.linear3,self.judge_embedding]
        for layer in linear_layers:
            layer.weight.data.normal_(0.0,0.1)

print("prgoram started!")
    
# initialize data paths, so we can read data easily
ruling_data_path = '/data/Dropbox/Projects/originalism/data/BloombergVOTELEVEL_Touse.dta'
sentences_data_path = '/data/Dropbox/judge_embedding_data_sp18/sentences_data.csv'
cite_graph_path = '/data/Dropbox/Data/corpora/chen-cases/cite-graph/graph.zip'
judge_bio_data_path = '/data/Dropbox/Data/Judge-Bios/judgebios/JudgesBioReshaped_TOUSE.dta'
topic_data_path = '/data/Dropbox/Projects/Ash_Chen/metadata/bb2topic.pkl'
processed_data_path = '/data/Dropbox/judge_embedding_data_sp18'

merged_sentence_data_path = '/data/Dropbox/judge_embedding_data_sp18/sentence_topic_judgeid.csv'

meta_data_path = '/data/Dropbox/judge_embedding_data_sp18/circuit_metadata_excerpt.dta'
table_of_cases_path = '/data/Dropbox/judge_embedding_data_sp18/tableofcases'

judge_mapping_binary_filename = 'judgemap.pkl'

# currently using 6B 300d glove, this one has 400K vocab
glove_emb_path = '/data/Dropbox/judge_embedding_data_sp18/glove_files/glove.6B.300d.txt'
glove_binary_filename = 'glove6B300d.pkl'

opinion_sum_vector_final_merged_data_filename = 'opinion_sum_vec_final.pkl'
opinion_sum_vector_split_6_data_filename = 'opinion_sum_vec_split6.pkl'

pd.options.display.max_columns = 999

# load data (in ready format)
X_train,y_train,X_val,y_val,X_test,y_test = load_general_data(processed_data_path,opinion_sum_vector_split_6_data_filename)

print("data load finished")

BATCH_SIZE = 64
train_dataset = data_utils.TensorDataset(data_tensor=X_train,target_tensor=y_train)
train_loader = data_utils.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)

INPUT_DIM = 300
HIDDEN_DIM = 500
EMBED_DIM = 500
number_judges = 2099

model = Judge_emb_model(input_dim=INPUT_DIM,hidden_layer_dim=HIDDEN_DIM,embedding_dim=EMBED_DIM,num_judges=number_judges)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

N_EPOCH = 20
TRAIN_SIZE = train_dataset.data_tensor.shape[0]
print("Training data size",TRAIN_SIZE)
train_losses = []
val_losses = []

X_val_var = Variable(X_val)
y_val_var = Variable(y_val)
model.eval()
y_pred_val = model.forward(X_val_var)
val_loss = criterion(y_pred_val,y_val_var)
print("initial val loss",val_loss.data[0])
startTime = time.time()

print("Training started!")
for i_epoch in range(N_EPOCH):
    epoch_train_loss = 0
    num_batches_per_epoch = int(TRAIN_SIZE/BATCH_SIZE)
    for i_batch,(X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        
        X_var, y_var = Variable(X_batch),Variable(y_batch)
        
        y_pred = model.forward(X_var)
        loss = criterion(y_pred,y_var)
        loss.backward()
        
        optimizer.step()
#         if i_batch % 2000 == 0:
#             print(i_epoch,i_batch,loss.data[0])
        epoch_train_loss += loss.data[0]
        
    # after each epoch
    
    X_val_var = Variable(X_val)
    y_val_var = Variable(y_val)
    model.eval()
    y_pred_val = model.forward(X_val_var)
    val_loss = criterion(y_pred_val,y_val_var)
    ave_train_loss = epoch_train_loss/num_batches_per_epoch
    print("epoch",i_epoch,"ave_train_loss",
          ave_train_loss,"validation loss:",val_loss.data[0],time.time()-startTime)
    val_losses.append(val_loss.data[0])
    train_losses.append(ave_train_loss)
    model.train()

result_embeddings = model.judge_embedding.weight.data.numpy()
emb_save_name = "trial_trained_embeddings.pkl"
dump_general_data(result_embeddings,processed_data_path,emb_save_name)
print("Training finished, trained embedding dump to path:",os.path.join(processed_data_path,emb_save_name))