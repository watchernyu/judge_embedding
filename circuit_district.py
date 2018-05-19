import pandas as pd
import numpy as np
import pickle as pk
import zipfile
import xml.etree.ElementTree as ET
import xml_to_json
import json
import string
import os

data_output_path = '/data/Dropbox/judge_embedding_data_sp18/'
circuit_output_path = data_output_path + 'circuit/'
circuit_xml_source_path = '/data/Dropbox/Data/lexis-data/circuits/'
bad_circuit_zip_path = data_output_path + 'bad_circuit_zip_list.pk'

def extract_circuit_district_id(content_list, zip_file_name):
    l = []
    for i in range(len(content_list)):
        file = zf.extract(content_list[i])
        filename = os.path.basename(content_list[i])
        try:
            xml_to_json.xml_to_json(file, circuit_output_path, filename)
        except:
            pass
        else:
            try:
                json_path = circuit_output_path + filename + '.json'
                data_file = open(json_path, encoding='utf-8').read()
                data = json.loads(data_file)

                for i in data.keys():
                    if 'courtCaseDoc' in data[i].keys(): 
                        d = {}
                        d['source_json'] = filename
                        d['source_key'] = i
                        try:
                            circuit_id = data[i]['courtCaseDoc']['courtCaseDocHead']['caseInfo']['citations']['citeForThisResource'][0]['$t'] 
                        except:
                            try:
                                circuit_id = data[i]['courtCaseDoc']['metadata']['pagination']['paginationScheme'][0]['@pageScheme']
                            except:
                                try:
                                    circuit_id = data[i]['courtCaseDoc']['courtCaseDocBody']['caseOpinions']['opinion']['bodyText']['p'][0]['citation'][0]['content']['span'][0]['$t']
                                except:
                                    circuit_id = None
                        finally:
                            d['circuit_id'] = circuit_id

                        try:
                            district_id = data[i]['courtCaseDoc']['courtCaseDocHead']['caseHistory']['priorHistory']['appealHistory']['historyCite']['citation']['content']['span'][0]['$t']
                        except:
                            district_id = None
                        else:
                            d['district_id'] = district_id

                        l.append(d)
            except:
                pass

    circuit_district = pd.DataFrame.from_dict(l)
    with open(data_output_path + 'circuit_district_{0}.pk'.format(zip_file_name),"wb") as f:  
            pk.dump(circuit_district, f)

# clean zip files

badzip = []
zip_list = np.array([6385, 6386, 6387, 6391, 6392, 6393, 6394, 6395, 6396, 6397]).astype(str)
for i in zip_list:
    print(i)
    circuit_xml_path = circuit_xml_source_path + '{0}.zip'.format(i)
    try:
        zf = zipfile.ZipFile(circuit_xml_path, 'r')
    except:
        badzip.append(i)
    else:
        content_list = zf.namelist()[1:]
        extract_circuit_district_id(content_list, i)

with open(bad_circuit_zip_path) as f:  
    pk.dump(badzip, f)


