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
from random import shuffle
from sklearn.utils import shuffle as skshuffle
import random
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import KFold
from sklearn import metrics
import matplotlib.pyplot as plt
rand_state = 44
torch.manual_seed(rand_state)
random.seed(rand_state)
np.random.seed(rand_state)

#read in data
processed_data_path = '/data/Dropbox/judge_embedding_data_sp18'
all_data_save_path = os.path.join(processed_data_path,"finalized_all_data","all_data_dict.pkl")
all_data_support_save_path = os.path.join(processed_data_path,"finalized_all_data","all_data_dict_support.pkl")
all_data_df_save_path = os.path.join(processed_data_path,"finalized_all_data","all_data_df.h5")
finished_embedding_folder_path = os.path.join(processed_data_path,'finished_judge_embedding')
all_data_dict = pickle.load(open(all_data_save_path,"rb"))
topic_glove_emb = all_data_dict['topic_glove_emb'][0]
judgeId2Index = all_data_dict['judge_id_to_index'][0]
judgeIndex2Id = all_data_dict['judge_index_to_id'][0]
all_data_df = all_data_dict['data_df']
judge_bio_path = os.path.join(processed_data_path,"JudgesBioReshaped_TOUSE.dta")
judge_bio_df = pd.read_stata(judge_bio_path)
naive_judge_emb_path = os.path.join(finished_embedding_folder_path,"centered_naive_emb.pkl")
naive_judge_emb = pickle.load(open(naive_judge_emb_path,"rb"))

trained_judge_emb_path = os.path.join(finished_embedding_folder_path,"centered_trained_emb_May13.pkl")
trained_judge_emb = pickle.load(open(trained_judge_emb_path,"rb"))

#create dataset contains embedding Index and Judge party
judge_party_df = judge_bio_df[['judgeidentificationnumber', 'x_dem', 'x_republican']].copy()
all_judge_df = all_data_df[['songername', 'judgeidentificationnumber']].copy()
all_judge_df = all_judge_df.drop_duplicates()
all_judge_df['emb_Index'] = all_judge_df.apply(lambda x: judgeId2Index[int(x['judgeidentificationnumber']], axis = 1)

all_judge_party_df = all_judge_df.merge(judge_party_df, on = 'judgeidentificationnumber', how = 'left')
all_judge_party_df = all_judge_party_df[all_judge_party_df['x_dem']+all_judge_party_df['x_republican'] == 1 ]
all_judge_party_df.drop_duplicates(inplace = True)
judge_party_count = all_judge_party_df.groupby('judgeidentificationnumber')['emb_Index'].count().reset_index()
unique_party_judge = judge_party_count[judge_party_count['emb_Index']==1]['judgeidentificationnumber']
all_judge_party_df = all_judge_party_df[all_judge_party_df['judgeidentificationnumber'].isin(unique_party_judge)]

#Prediction using NAIVE EMB with LR (l2 reg)

naive_judge_emb_df = pd.DataFrame.from_dict(naive_judge_emb)
naive_judge_emb_final = naive_judge_emb_df.join(all_judge_party_df.set_index('emb_Index'), how='right').drop(['songername', 'judgeidentificationnumber', 'x_republican'], axis = 1)
naive_judge_emb_final.set_index(np.arange(0,naive_judge_emb_final.shape[0]), inplace = True)
X = naive_judge_emb_final.iloc[:,:300].values
y = naive_judge_emb_final['x_dem'].values
l2_reg = 10**np.arange(-2,2,0.01)
accuracy = []
for i in l2_reg:
    model = LR(C = i)
    cv_results = cross_validate(model, X, y, return_train_score=False)
    scores = cv_results['test_score']   
    accuracy.append({'l2_reg' : i, 'mean_accuracy': np.mean(scores), 'std' : np.std(scores)})

accuracy_df = pd.DataFrame.from_dict(accuracy)
plt.semilogx(accuracy_df['l2_reg'], accuracy_df['mean_accuracy'], label = 'mean')
plt.semilogx(accuracy_df['l2_reg'], accuracy_df['mean_accuracy'] + accuracy_df['std'], 'k--', label = 'mean+1*std')
plt.semilogx(accuracy_df['l2_reg'], accuracy_df['mean_accuracy'] - accuracy_df['std'], 'r--', label = 'mean-1*std')
plt.xlabel('l2 regularization coefficient')
plt.ylabel('accuracy')
plt.legend()
plt.show()

best_accuracy = np.max(accuracy_df['mean_accuracy'])
best_l2 = accuracy_df['l2_reg'][accuracy_df['mean_accuracy'].idxmax()]
print('Naive Emb + LR: Highest accuracy of ', best_accuracy, 'is achieved with l2 regularization = ', best_l2)
## Prediction using naive judge embedding with Random Forest Classifier
model = RF(random_state = rand_state)
cv_results = cross_validate(model, X, y, return_train_score=False)
scores = cv_results['test_score']   
print('Naive Emb + default RF: mean accuracy = ',np.mean(scores), '. accuracy std = ', np.std(scores))

## Prediction using trained judge embedding with Logistic Regression

trained_judge_emb_df = pd.DataFrame.from_dict(trained_judge_emb)

trained_judge_emb_final = trained_judge_emb_df.join(all_judge_party_df.set_index('emb_Index'), how='right').drop(['songername', 'judgeidentificationnumber', 'x_republican'], axis = 1)
trained_judge_emb_final.set_index(np.arange(0,naive_judge_emb_final.shape[0]), inplace = True)
X = trained_judge_emb_final.iloc[:,:300].values
y = trained_judge_emb_final['x_dem'].values

l2_reg = 10**np.arange(-4,-2,0.01)
accuracy = []
for i in l2_reg:
    model = LR(C = i)
    cv_results = cross_validate(model, X, y, return_train_score=False)
    scores = cv_results['test_score']   
    accuracy.append({'l2_reg' : i, 'mean_accuracy': np.mean(scores), 'std' : np.std(scores)})

accuracy_df = pd.DataFrame.from_dict(accuracy)
plt.semilogx(accuracy_df['l2_reg'], accuracy_df['mean_accuracy'], label = 'mean')
plt.semilogx(accuracy_df['l2_reg'], accuracy_df['mean_accuracy'] + accuracy_df['std'], 'k--', label = 'mean+1*std')
plt.semilogx(accuracy_df['l2_reg'], accuracy_df['mean_accuracy'] - accuracy_df['std'], 'r--', label = 'mean-1*std')
plt.xlabel('l2 regularization coefficient')
plt.ylabel('accuracy')
plt.legend()
plt.show()

best_accuracy = np.max(accuracy_df['mean_accuracy'])
best_l2 = accuracy_df['l2_reg'][accuracy_df['mean_accuracy'].idxmax()]
print('Trained Emb + LR: Highest accuracy of ', best_accuracy, 'is achieved with l2 regularization = ', best_l2)


# Prediction using trained judge embedding with Random Forest Classifier

model = RF(random_state = rand_state)
cv_results = cross_validate(model, X, y, return_train_score=False)
scores = cv_results['test_score']   
print('Trained Emb + default RF: mean accuracy = ',np.mean(scores), '. accuracy std = ', np.std(scores))
