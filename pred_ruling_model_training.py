# Train judge embedding using the task of predicting ruling label

# Phu, Andrea and Watcher
# 2018 Spring
import numpy as np
import pandas as pd
import time
import os
import pickle
import string
import psutil
import sklearn

%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from ggplot import *

from sklearn.manifold import TSNE
import math

import seaborn as sns


# Phu, Andrea and Watcher
# 2018 Spring
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch import FloatTensor, LongTensor
import pickle
import torch.utils.data as data_utils
from random import shuffle
from sklearn.utils import shuffle as skshuffle
import random

torch.manual_seed(44)
random.seed(44)
np.random.seed(44)


processed_data_path = '/data/Dropbox/judge_embedding_data_sp18'
all_data_save_path = os.path.join(processed_data_path,"finalized_all_data","all_data_dict.pkl")
all_data_support_save_path = os.path.join(processed_data_path,"finalized_all_data","all_data_dict_support.pkl")
all_data_df_save_path = os.path.join(processed_data_path,"finalized_all_data","all_data_df.h5")
finished_embedding_folder_path = os.path.join(processed_data_path,'finished_judge_embedding')
plot_save_path = '/data/Dropbox/judge_embedding_data_sp18/plots'

## read in data for each case

all_data_dict = pickle.load(open(all_data_save_path,"rb"))
topic_glove_emb = all_data_dict['topic_glove_emb'][0]
judgeId2Index = all_data_dict['judge_id_to_index'][0]
judgeIndex2Id = all_data_dict['judge_index_to_id'][0]
all_data_df = all_data_dict['data_df']
all_data_df[:4]

# we now read in the judge bio data
judge_bio_path = os.path.join(processed_data_path,"JudgesBioReshaped_TOUSE.dta")
judge_bio_df = pd.read_stata(judge_bio_path)

## get citation embeddings

citation_folder_path = os.path.join(processed_data_path,"citation_graph_data_node2vec")
citation_emb_path = os.path.join(citation_folder_path,"citation.emb")
citation_case2index_path = os.path.join(citation_folder_path,"case_id_to_index.pickle")
citation_name2index_path = os.path.join(citation_folder_path,"citation_name_to_index.pickle")
citation_emb_fpt = open(citation_emb_path,"r")
citation_emb_lines = citation_emb_fpt.readlines()
citation_caseid2Index = pickle.load(open(citation_case2index_path,"rb"))
citation_name2Index = pickle.load(open(citation_name2index_path,"rb"))


def citation_txt_to_numpy(lines):
    R, C = lines[0].strip().split()
    R = int(R)
    C = int(C)
    token_list = []
    for r in range(1,1+R):
        line = lines[r]
        tokens = line.strip().split()
        token_list.append(tokens[1:])
    return np.array(token_list).astype(float)

citation_emb = citation_txt_to_numpy(citation_emb_lines)

def add_citation_emb_to_df(df):
    # this function will add citation data to all_data_df
    df['citation'] = None
    n_instance = df.shape[0]
    citation_emb_dim = citation_emb.shape[1]
    case_no_cit_count = 0
    for i in range(n_instance):
        case = df.iloc[i]
        caseid = case['caseid']
        if caseid in citation_caseid2Index:
            citation_index = citation_caseid2Index[caseid]
            citation_vec = citation_emb[citation_index]
        else:
            case_no_cit_count+=1
            citation_vec = np.zeros(citation_emb_dim)
        
        df.iat[i,df.columns.get_loc('citation')] = citation_vec
    print("number of citation not found:", case_no_cit_count)


add_citation_emb_to_df(all_data_df)

## There is a number of cases where their corresponding citations are not found. We simply use all 0 citation embeddings for them


## We will do a normalization of doc vectors, using center by topic-year
## That means for each doc vector, we reduce from it the topic-year vector it is related to.
## E.g. if a doc vector is from 1981 and topic of civil procedure, then we reduce it by the average vector of all vectors that are from 1981 and of topic civil procedure.

def get_year_topic_dict(df):
    # given all data df, give the year-topic average vectors
    # in the form of a year-topic dictionary
    # key is year-topic string, and value is a 300 dim vector
    # which is the average of all the cases with that year and topic
    
    yearTopicDict = {}
    yearTopicCount = {}
    
    n_instance = df.shape[0]
    for i in range(n_instance):
        case = df.iloc[i]
        year = case['year']
        topic = case['topic']
        yearTopic = str(year)+"-"+str(topic)
        opinion_vector = case['opinion_vector']
        
        if yearTopic not in yearTopicDict:
            yearTopicDict[yearTopic] = np.zeros(300) + opinion_vector
            yearTopicCount[yearTopic] = 1
        else:
            yearTopicDict[yearTopic] += opinion_vector
            yearTopicCount[yearTopic] += 1
            
    # now we accumulated all the cases
    # we do an average for each dictionary entry
    for k in yearTopicDict.keys():
        yearTopicDict[k] /= yearTopicCount[k]
        
    return yearTopicDict

def add_centered_vec_to_df(df,year_topic_dict,verbose=0):
    starttime = time.time()
    df['centered_opinion_vec'] = None
    n_instance = df.shape[0]
    for i in range(n_instance):
        if verbose and (i==2 or i%int(n_instance/20)==0 ) :
            print(i,time.time()-starttime)
        case = df.iloc[i]
        year = case['year']
        topic = case['topic']
        yearTopic = str(year)+"-"+str(topic)
        meanTopicYear = year_topic_dict[yearTopic]
        opinion_vector = case['opinion_vector']    
        centered_vec = opinion_vector - meanTopicYear
        
        df.iat[i,df.columns.get_loc('centered_opinion_vec')] = centered_vec
        
year_topic_dict = get_year_topic_dict(all_data_df)
add_centered_vec_to_df(all_data_df,year_topic_dict,verbose=1)

## Next we train a model to do the ruling prediction task

# first split the data

def train_val_test_split(data_df,number_judges,train_ratio=0.8,val_ratio=0.1,verbose=0,toshuffle=True):
    starttime= time.time()
    sorted_all_data = data_df.sort_values(by='judge_embed_index')
    train_indexes = []
    val_indexes = []
    test_indexes = []
    currentiloc = 0
    for judge_index in range(number_judges):
        if verbose and judge_index%500 == 0:
            print(judge_index,time.time()-starttime)
        
        cases_of_this_judge = sorted_all_data.loc[sorted_all_data['judge_embed_index'] == judge_index]
        number_cases = cases_of_this_judge.shape[0]
        n_of_train = int(number_cases*train_ratio)
        n_of_val = int(number_cases*val_ratio)
        
        nextiloc = currentiloc+number_cases
        
        indexes = [i for i in range(currentiloc, nextiloc)]
        shuffle(indexes)
        
        train_indexes += indexes[:n_of_train]
        val_indexes += indexes[n_of_train:n_of_train+n_of_val]
        test_indexes += indexes[n_of_train+n_of_val:]
        
        currentiloc = nextiloc
    return skshuffle(data_df.loc[train_indexes]),skshuffle(data_df.loc[val_indexes]),skshuffle(data_df.loc[test_indexes])

data_train, data_val, data_test = train_val_test_split(all_data_df,2099,verbose=1)

# now we get data in tensor form
def df_to_Tensor(df,topic_glove_emb,verbose=0):
    # use this to convert a dataframe to torch tensor
    # this is the predict ruling label version
    
    # WE DO NOT ADD JUDGE EMBEDDING HERE, THAT WILL BE HANDLED BY THE MODEL
    
    feature_dim = 300+300+128+1 # 300 for opinion vec, 300 for topic, 128 for citation, 
    # and 1 value to indicate which judge
    
    X = np.zeros((df.shape[0],feature_dim))
    y = np.zeros(df.shape[0]) # y is the ruling label
    
    for i in range(df.shape[0]):
        if verbose and i%10000==0:
            print(i)
        
        data_entry = df.iloc[i]
        
        # add opinion vector to X
        X[i,:300] = data_entry['centered_opinion_vec']
        
        # add topic vector to X
        topic = data_entry['topic']
        topic = str.lower(str(topic)) 
        
        if topic not in topic_glove_emb: # deal with any unknown topic
            topic = "<UNK>"
            
        X[i,300:600] = topic_glove_emb[topic]
        
        # add citation vector to X
        X[i,600:600+128] = data_entry['citation']
        
        X[i,-1] = data_entry['judge_embed_index'] # in the model, the model needs to pick a judge vector
        # according to this value
        
        # set y
        decision = data_entry['judge_decision']
        y[i] = decision
        
    return FloatTensor(X),LongTensor(y)

X_train, y_train = df_to_Tensor(data_train,topic_glove_emb,1)
X_val, y_val = df_to_Tensor(data_val,topic_glove_emb,1)
X_test, y_test = df_to_Tensor(data_test,topic_glove_emb,1)

## Now we have X_train, y_train and in Tensor form, we can now create the model and start our training
BATCH_SIZE = 128
train_dataset = data_utils.TensorDataset(data_tensor=X_train,target_tensor=y_train)
train_loader = data_utils.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)


## The ruling label prediction requires the model to put judge embedding as part of the input
## The model will first concatenate the judge embedding with the input X it get from data loader and then put them through the layers.
class Judge_emb_model(nn.Module):
    def __init__(self, input_dim, hidden_layer_dim, embedding_dim, num_judges):
        super(Judge_emb_model,self).__init__()
        # since output is one of 2 classes, output dim is 2 
        
        self.judge_embedding = Variable(torch.FloatTensor(num_judges,embedding_dim),requires_grad=True)  # H x J
        
        # input is m x D
        self.linear1 = nn.Linear(input_dim-1+embedding_dim,hidden_layer_dim) # D x H  # -1 because that is for judge
        self.dropout1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(hidden_layer_dim,hidden_layer_dim) # H x 2
        self.dropout2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(hidden_layer_dim,2) # H x 2
        # the output is m x J
        
        self.init_weights()

    def forward(self, X):
        # first concatenate the X and the judge emb
        judge_indexes = X[:,-1].data.numpy().astype(int).tolist()
        
        X_combined = torch.cat([X[:,:-1],self.judge_embedding[judge_indexes]],1)
        
        out = F.relu(self.linear1(X_combined))
        out = self.dropout1(out)
        out = F.relu(self.linear2(out))
        out = self.dropout2(out)
        
        out = self.linear3(out)
        
        # now we have m x J matrix, for m data points, we can do log softmax
        log_prob = F.log_softmax(out,dim=1)
        return log_prob # for each opinion data, this is probability of which judge writes this opinion
    
    def init_weights(self):
        linear_layers = [self.linear1,self.linear2,self.linear3]
        for layer in linear_layers:
            layer.weight.data.normal_(0.0,0.1)
        self.judge_embedding.data.normal_(0.0,0.1)

INPUT_DIM = X_train.shape[1]
HIDDEN_DIM = 300
EMBED_DIM = 300
number_judges = 2099
model = Judge_emb_model(input_dim=INPUT_DIM,hidden_layer_dim=HIDDEN_DIM,
                        embedding_dim=EMBED_DIM,num_judges=number_judges)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(model.parameters())+[model.judge_embedding],lr=0.005)


N_EPOCH = 8
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
model.train()

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


judge_emb = model.judge_embedding.data.numpy()

save_pred_ruling_trained_path = os.path.join(finished_embedding_folder_path,"pred_ruling_trained_emb.pkl")
pickle.dump(judge_emb,open(save_pred_ruling_trained_path,"wb"))