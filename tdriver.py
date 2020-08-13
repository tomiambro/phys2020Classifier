#!/usr/bin/env python

import numpy as np, os, sys
import pandas as pd
from scipy.io import loadmat
from run_12ECG_classifier import load_12ECG_model, run_12ECG_classifier
from get_12ECG_features import get_12ECG_features_labels, get_HRVs_values

import multiprocessing as mp

def load_challenge_data(filename):


    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file,'r') as f:
        header_data=f.readlines()


    return data, header_data


def save_challenge_predictions(output_directory,filename,scores,labels,classes):

    recording = os.path.splitext(filename)[0]
    new_file = filename.replace('.mat','.csv')
    output_file = os.path.join(output_directory,new_file)

    # Include the filename as the recording number
    recording_string = '#{}'.format(recording)
    class_string = ','.join(classes)
    label_string = ','.join(str(i) for i in labels)
    score_string = ','.join(str(i) for i in scores)

    with open(output_file, 'w') as f:
        f.write(recording_string + '\n' + class_string + '\n' + label_string + '\n' + score_string + '\n')


def process_signals(i, f, num_files, input_directory, df_raw):
    print('    {}/{}...'.format(i+1, num_files))
    tmp_input_file = os.path.join(input_directory,f)
    data,header_data = load_challenge_data(tmp_input_file)
    
    features = get_HRVs_values(data, header_data)
    # features = get_12ECG_features_labels(data, header_data)
    return features
    # aux = pd.DataFrame([features], columns=columns)
    # df_raw = df_raw.append(aux, ignore_index=True)
    # return df_raw

  
# Find unique number of classes  
def get_classes(input_directory,files):

    classes=set()
    for f in files:
        g = f.replace('.mat','.hea')
        input_file = os.path.join(input_directory,g)
        with open(input_file,'r') as f:
            for lines in f:
                if lines.startswith('#Dx'):
                    tmp = lines.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())

    return sorted(classes)

if __name__ == '__main__':
    # Parse arguments.
    if len(sys.argv) != 3:
        raise Exception('Include the input and output directories as arguments, e.g., python driver.py input output.')

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    # Find files.
    input_files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
            input_files.append(f)

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    classes=get_classes(input_directory,input_files)

    # Load model.
    print('Loading 12ECG model...')
    # model = load_12ECG_model()

    # Create dataset
    columns = ['age', 'sex', 'fmax', 'mean_RR', 'mean_R_Peaks', 'mean_T_Peaks', 'mean_P_Peaks', 'mean_Q_Peaks', 'mean_S_Peaks', 'median_RR', 'median_R_Peaks', 'std_RR', 'std_R_Peaks', 'var_RR', 'var_R_Peaks', 'skew_RR', 'skew_R_Peaks', 'kurt_RR', 'kurt_R_Peaks', 'mean_P_Onsets', 'mean_T_Offsets', 'HRV', 'label']
    df_raw = pd.DataFrame(columns=columns)

    # Iterate over files.
    print('Extracting 12ECG features...')
    num_files = len(input_files)



    pool = mp.Pool(mp.cpu_count())

    result_obj = [pool.apply_async(process_signals, args=(i, f, num_files, input_directory, df_raw)) for i, f in enumerate(input_files)]
    
    pool.close()
    pool.join()

    results = [r.get() for r in result_obj]
    # print(results)
    df_raw = pd.concat(results)
    df_raw = df_raw.reset_index()
    df_raw = df_raw.drop('index', axis=1)


        # current_label, current_score = run_12ECG_classifier(data,header_data,classes, model)
        # Save results.
        # save_challenge_predictions(output_directory,f,current_score,current_label,classes)


    os.makedirs('datasets/raw', exist_ok=True)
    df_raw.to_feather('datasets/raw/pyhs-raw-lead2-HRV')
    print(df_raw)
    print('Done.')
