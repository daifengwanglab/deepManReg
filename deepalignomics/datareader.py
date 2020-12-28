import numpy as np
import pandas as pd
import pdb


def read_data(datapath, metadatapath, label_key='Schizophrenia'):
    """read_data
    :param datapath: path to data file (gene)
    :param metadatapath: path to meta data file of patients
    
    :output x: data of shape n_patient * n_features
    :output y: label of shape n_patients, label[i] == 1 means that the patient has label_key.
    :output patient_id: list of patient id
    :output feature_name: list of feature name
    
    Read data from the 2 above files.
    A few notes:
    - Some patients exist in one file but not in another, they are removed
    - Some patients in data file has more than one row in the metadata file, only one is kept (assume that the diagnosis is the same for both rows)
    """
    # Reading data file
    csv_file = pd.read_csv(datapath, u'\t', header=None)
    data = csv_file.values
    patient_id = data[0,1:]
    feature_name = data[1:,0]
    x = data[1:,1:].astype(np.float32).T
    n_patient = x.shape[0]
    
    # Reading metadata file
    csv_file = pd.read_csv(metadatapath, header=None)
    metadata = csv_file.values
    header = metadata[0,:]
    metadata = metadata[1:,:]
    
    individualID = metadata[:, np.where(header == 'individualID')].flatten()
    diagnosis = metadata[:, np.where(header == 'diagnosis')].flatten()
    y = np.zeros((n_patient, ), dtype=np.int32) - 1
    
    # Match metadata with rows in data
    mismatch = []
    for i in range(n_patient):
        patient_id_i = patient_id[i]
        if patient_id_i[0:4] == 'X201':
            patient_id_i = patient_id_i[1:5] + '-' + patient_id_i[6:]
        
        metadata_index = np.where(individualID == patient_id_i)[0]
        if (metadata_index.size == 0):
            mismatch += [patient_id_i]
            continue
        else:
            metadata_index = metadata_index[0]
        
        if (diagnosis[metadata_index] != label_key):
            y[i] = 0
        else:
            y[i] = 1
    
    x = x[y != -1, :]
    patient_id = patient_id[y != -1]
    y = y[y != -1]
    
    return x, y, patient_id, feature_name
    
    
if __name__ == '__main__':
    x, y, patient_id, feature_name = read_data('./data/DER-01_PEC_Gene_expression_matrix_normalized.txt',
        './data/PEC_Capstone_clinical_meta - PEC_Capstone_clinical_meta.csv')
    
    np.save('./data/gene_data.npy', x)
    np.save('./data/label.npy', y)
    np.save('./data/patient_id.npy', patient_id)
    np.save('./data/feature_name.npy', feature_name)