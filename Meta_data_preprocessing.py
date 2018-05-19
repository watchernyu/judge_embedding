import pandas as pd
import numpy as np
import pickle as pk
import zipfile
import json
import string
import os
#original dataset
sentence_path = '/data/Dropbox/judge_embedding_data_sp18/sentences_data.csv'
ruling_path = '/data/Dropbox/Projects/originalism/data/BloombergVOTELEVEL_Touse.dta'
district_path = '/data/Dropbox/Projects/Ash_Chen/lexis-data/6411.zip'
topic_path = '/data/Dropbox/Projects/Ash_Chen/metadata/bb2topic.pkl'
meta_path = '/data/Dropbox/judge_embedding_data_sp18/circuit_metadata_excerpt.dta'
bloomberg_lexis_path = '/data/Dropbox/judge_embedding_data_sp18/caseid_citation.dta'

#output directory
data_output_path = '/data/Dropbox/judge_embedding_data_sp18/'

#preprocessed dataset
sentence_topic_judge_path = data_output_path+ 'sentence_topic_judgeid.csv'
judge_decision_path = data_output_path+ 'judge_decision.csv'
circuit_output_path = data_output_path+ 'circuit/' 
blmberg_lexis_circuit_path = data_output_path+ 'blmberg_lexis_circuit.csv'
circuit_district_link_path = data_output_path + 'circuit_district_link.csv'

#geniss
geniss = pk.load(open('/data/Dropbox/Projects/Ash_Chen/metadata/bb2genis.pkl', 'rb') )
df_geniss = pd.DataFrame(geniss, index=[0]).transpose()
df_geniss.reset_index(level=0, inplace = True)
df_geniss.columns = ['caseid', 'geniss']
#topic
topic = pk.load(open(topic_path, 'rb') )
df_topic = pd.DataFrame(topic, index=[0]).transpose()
df_topic.reset_index(level=0, inplace = True)
df_topic.columns = ['caseid', 'topic']

#ruling
df_ruling_key = pd.read_stata(ruling_path,
                              columns = ['songername', 'judgeidentificationnumber', 
                                         'jOrigname', 'caseid'])

# Merge: create 'sentence_topic_judge.csv'

##1 Sentence + ruling to get judge id and date of case 
df_merged = df_sentence.merge(df_ruling_key, left_on=['caseid', 'judge_last_name'], 
                              right_on=['caseid', 'jOrigname'], 
                              how = 'inner')
df_merged.drop(['judge_last_name', 'jOrigname'], axis = 1, inplace = True)


##2 + Topic
df_merged = df_merged.merge(df_topic, on = 'caseid', how = 'left')

##3 + Geniss
df_merged = df_merged.merge(df_geniss, on = 'caseid', how = 'left')

##drop where songername is null (no matching in ruling data)
df_merged.dropna(axis=0, subset = ['songername'], inplace = True)
df_merged.to_csv(sentence_topic_judge_path, index = False)

# Judge Level Decision: create 'judge_decision.csv'

df_meta = pd.read_stata(meta_path)
df_ruling_affirm = pd.read_stata(ruling_path, columns = ['caseid', 'affirmdummy'])
df_ruling_affirm.drop_duplicates(inplace = True)
df_ruling_affirm.affirmdummy.fillna(-1, inplace = True)

df_merged = pd.read_csv(sentence_merge_path)
df_meta = df_meta[df_meta['caseid'].isin(df_merged.caseid)]
df_meta = df_meta.merge(df_ruling_affirm, on='caseid', how= 'left')
df_meta['dissent'] = df_meta.apply(lambda x: 1 if x['j']==x['Dissenting1'] else 0, axis = 1)
def decision(x):
    if x['affirmdummy'] == 1: #affirm
        if x['dissent'] == 0:
            return 1 #affirm
        elif x['dissent'] == 1:
            return 0 #reverse
    elif x['affirmdummy'] == 0: #reverse
        if x['dissent'] == 0:
            return 0 #reverse
        elif x['dissent'] == 1:
            return 1

df_meta['judge_decision'] = df_meta.apply(decision, axis = 1)

df_meta = df_meta[['caseid','Circuit','date','songername','affirmdummy','judge_decision']]
df_meta.to_csv(judge_decision_path, index = False)

# bloomberg_lexis link
blmberg_lexis_df = pd.read_stata(bloomberg_lexis_path)
all_blmberg_caseid = pd.read_csv(judge_decision_path, usecols = ['caseid', 'Circuit'])blmberg_lexis_circuit = blmberg_lexis_df.merge(all_blmberg_caseid, on = 'caseid', how = 'right')
blmberg_lexis_circuit.drop_duplicates(inplace = True)
blmberg_lexis_circuis_circuit.dropna(axis=0, how='any', inplace = True)
blmberg_lexis_circuit.to_csv(blmberg_lexis_circuit_path, index = False)
blmberg_lexis_circuit = pd.read_csv(data_output_path+'blmberg_lexis_circuit.csv')
col_names =  ['circuit_id', 'district_id', 'caseid', 'Circuit']
circuit_district_df = pd.DataFrame(columns = col_names)

circuit_district_file_list = ['6385', '6386', '6387', '6388', '6389', '6390', 
                             '6391', '6392', '6393', '6394', '6395', '6396', '6397']
for i in circuit_district_file_list:
    try:
        circuit_district = pk.load(open( data_output_path + "circuit_district_{0}.pk".format(i), "rb" ) )
    except:
        pass
    else:
        circuit_district = circuit_district[pd.notnull(circuit_district['circuit_id'])]
        circuit_district_sub = circuit_district[['circuit_id', 'district_id']]
        circuit_district_sub = circuit_district_sub.drop_duplicates()
        circuit_district_new = circuit_district_sub.merge(blmberg_lexis_circuit, 
                                                           left_on=['circuit_id'], right_on=['citation'], 
                                                           how = 'right')
        circuit_district_new = circuit_district_new[pd.notnull(circuit_district_new['circuit_id'])]
        circuit_district_new = circuit_district_new[['circuit_id', 'district_id', 'caseid', 'Circuit']]
        circuit_district_df = pd.concat([circuit_district_df, circuit_district_new])

circuit_district_df.to_csv(circuit_district_link_path, index = False)
