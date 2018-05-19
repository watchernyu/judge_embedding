import pandas as pd
import numpy as np
import pickle as pk
import zipfile
import xml.etree.ElementTree as ET
import xml_to_json
import json
import string
import os
#original dataset
district_path = '/data/Dropbox/Projects/Ash_Chen/lexis-data/6411.zip'
bloomberg_lexis_path = '/data/Dropbox/judge_embedding_data_sp18/caseid_citation.dta'

#output directory
data_output_path = '/data/Dropbox/judge_embedding_data_sp18/'

#preprocessed dataset
district_output_path = data_output_path+ 'district/' #district json files parsed from xml
district_opinion_path = data_output_path + 'district_opinion_list.pk'

import psutil
def show_current_memory_usage():
    pid = os.getpid()
    py = psutil.Process(pid)
    mem_use_in_GB = py.memory_info().rss/(2**30)
    print("currently using",mem_use_in_GB,"GB memory!")


zf = zipfile.ZipFile(district_path, 'r')
content_list = zf.namelist()[1:]

l = []
Error = []
for i in range(len(content_list)):
        if i % 100 ==0 :
            print ('{0} left to go'.format(len(content_list)-i))
            show_current_memory_usage()
        try:     
            file = zf.extract(content_list[i])
            filename = os.path.basename(content_list[i])
            xml_to_json.xml_to_json(file, district_output_path, filename)

            json_path = district_output_path + filename + '.json'
            data_file = open(json_path, encoding='utf-8').read()
            data = json.loads(data_file)
            opinion = data['1']['courtCaseDoc']['courtCaseDocBody']['caseOpinions']['opinion']['bodyText']['p']
            opinion_text = []
            for i in range(len(opinion)):
                if isinstance(opinion[i]['anchor'], dict):
                    if '$t1' in opinion[i]['anchor'].keys():
                            text = opinion[i]['anchor']['$t1']
                            if isinstance(text, str):
                                text = text.translate(str.maketrans('','',string.punctuation))
                                opinion_text.append(text)
            opinion_text = ' '.join(opinion_text)
            try:
                caseid = data['1']['courtCaseDoc']['courtCaseDocHead']['caseInfo']['citations']['citeForThisResource']['$t']
            except:
                caseid = None


            l.append({'source_file':filename, 'caseid': caseid, 'opinion': opinion_text})
            !rm $json_path
        except: 
            Error.append(filename)
            print(filename + ' error!')

with open(district_opinion_path,"wb") as f:  
        pk.dump([l, Error], f)

#read district_opinion_list.pk
df = pk.load(open( district_opinion_path, "rb" ) )
district_id_df = pd.DataFrame.from_dict(df[0])
