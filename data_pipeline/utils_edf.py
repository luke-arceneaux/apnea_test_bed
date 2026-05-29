'''
Tool to extract data for HomePAP lab data

- edf2csv: convert EDF homepap data to CSV files (under raw_data)
- xml_annotations_to_csv: parse XML annotations from Homepap and convert them to expert_annotations csv. 

There are two sets of data:   lab and home

dataset should be:  homepaplab or homepaphome

Assumes the source files are named properly before processing:

EDF:
   HomePapLab_CannulaFlow_subj????_sess1.edf
   
   
#
#  Uses NSRR annotation file
#
1/21/2022  	Vivian Cheng 	moved function execution calls to the bottom 
							parameterized the paths
							added "central" and "mixed" in concept of xml_annotations_to_csv
							Add comments
1/18/2022 	Vivian Cheng	Created
'''
import numpy as np
import pandas as pd
import os
import re
import csv
import pyedflib

'''

CONVERT HOMEPAP EDF FILES TO RAW_DATA CSV 

TODO: 

Install PyEDFLib using

pip install pyEDFlib

'''

# import s3_sdk
# from s3_sdk import S3Client
# from s3_sdk import ACCESS_KEY, SECRET_KEY, BUCKET_NAME

# s3 = S3Client(ACCESS_KEY=ACCESS_KEY,
#                       SECRET_KEY=SECRET_KEY,
#                       BUCKET_NAME=BUCKET_NAME)

# Main function that calls edf2csv_helper on each file 
def edf2csv(path, new_path, signal, signals_to_process, spo2_signal, dataset):
    
    files = os.listdir(path)
    for filename in files:
        print(f'*********** Processing EDF {filename} ***********')
#         matches = re.search(r'SSC_(.*)_(.*).EDF', filename)
        matches = re.search('.*\.edf', filename)
#         subject = matches.group(1)
#         session = matches.group(2)
        subject = ""
        session = ""

        if matches:
            edf2csv_helper(os.path.join(path, filename),
                       new_path,
                       signal,
                       subject,
                       session,
                       signals_to_process,
                       spo2_signal,
                       dataset)
            
#         edf2csv_helper(os.path.join(path, filename),
#                        new_path,
#                        signal,
#                        subject,
#                        session,
#                        signals_to_process,
#                        spo2_signal,
#                        dataset)


# Edf2csv converter for one file
def edf2csv_helper(filename, new_path, signal, subject, session, signals_to_process, spo2_signal, dataset):
    if not os.path.isdir(new_path): os.makedirs(new_path)
    # Read EDF
    f = pyedflib.EdfReader(filename)
    # Number of signals
    n = f.signals_in_file

    # List of all signals
    signals = f.getSignalLabels()
    print(signals)
    # NOTE: Select which signals you'd like to process
    print(f'Signals to process: {signals_to_process}')
    signals_to_process_indices = []
    for i in range(len(signals_to_process)):
        try:
            index = signals.index(signals_to_process[i])
            signals_to_process_indices.append(index)
        except:
            print(f'{signals_to_process[i]} not in list of signals')

    if spo2_signal:
        spo2_index = signals.index(spo2_signal)

    # Loop through each signal and write to csv
    df_list = []
    for i in signals_to_process_indices:
#         print(i)
        sample_freq = f.getSampleFrequency(i)
        signal_header = f.getSignalHeader(i)

        # print(f'*****Processing signal {signals[i]}*****')
        
        # Create Time, Value columns
        values = f.readSignal(i).tolist()
        spo2_values = f.readSignal(spo2_index).tolist()

        # Downsample signal so that it's the same length as Spo2
#         print(f'len sig: {len(values)}, len spo2: {len(spo2_values)}')
#         len_sig = len(values)
#         len_spo2 = len(spo2_values)
#         factor = len_sig / len_spo2 
        # check that factor is a whole number
#         assert(factor % 1 == 0)
#         print('factor:', factor)
#         # make signal same length as spo2
#         values = values[::int(factor)]
#         assert(len(values) == len(spo2_values))
#         print(f'new len sig: {len(values)}, new len spo2: {len(spo2_values)}')

        # create timestamps
        times = np.arange(0, len(values)/sample_freq, step=1/sample_freq)

        # Round to 3 decimal places 
        values = np.around(values, decimals=3) # round
#         spo2_values = np.around(spo2_values, decimals=3)
        times = np.around(times, decimals=3) # round

#         d = {'Time':times, 'Value':values, 'SpO2':spo2_values, "Position":}
        d = {'Time':times, 'Value':values, 'SpO2':spo2_values}
#         d = {'Time':times, 'Value':values}

        # Write to csv
        df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.items() ]))
        df_list.append(df)

    df = df_list[0]
    # print(df.head())
    for i in range(1, len(df_list)):
        df = df.merge(pd.DataFrame(df_list[i]), on=['Time','SpO2'], how='outer', suffixes=('',f'_{signals_to_process[i]}'))
        # print(df.head())
#     data = {}
    df.rename(columns={'Value':"Position", "Value_HR": "HR"}, inplace=True)

#     # Loop through each requested signal
#     for i in signals_to_process_indices:
#         sig_name = signals[i]
#         values = f.readSignal(i)
#         values = np.around(values, decimals=3)
#         data[sig_name] = values
#         print(values.shape)

#     # Include SpO2 separately
#     spo2_values = f.readSignal(spo2_index)
#     spo2_values = np.around(spo2_values, decimals=3)
#     data['SpO2'] = spo2_values
#     print(spo2_values.shape)

#     # Generate time vector using SpO2 sample frequency (or choose any)
# #     sample_freq = f.getSampleFrequency(spo2_index)
# #     n_samples = len(spo2_values)
#     sample_freq = 10
#     n_samples = len(data["Flow Patient"])
#     times = np.arange(0, n_samples / sample_freq, step=1 / sample_freq)
#     times = np.around(times, decimals=3)
#     data['Time'] = times
#     for k in data.keys():
#         print(data.get(k).shape)

#     # Create dataframe and write to CSV
#     df = pd.DataFrame(data)

        # TODO: Use this to downsample to every n rows
        # if sample_freq > 32:
        #     df = df.iloc[::20]
        
#         out_filename = f'ssc_stereo_subj{subject}_sess{session}.csv'
#         out_filename = filename[16:-4] + "og" + '.csv'
    out_filename = filename[-24:-4] + "og" + '.csv'

    out = os.path.join(new_path, out_filename)
    print(out)
    df.to_csv(out, sep=',', header=True, index=False)
#         print(f'Uploading {out} to S3.....')
#         s3.upload_df(df, out,bucket_name='neurostimstore')



########################################################################################################################
'''
PARSE XML ANNOTATIONS AND CONVERTS TO CSV

#
#   Use NSRR annotation as source file for conversion 
#
TODO

Install BeautifulSoup for XML parsing
pip install beautifulsoup4

'''

from bs4 import BeautifulSoup    
  
''' Parses XML annotations from HomePAP, converts to CSV'''


def xml_annotations_to_csv(path, new_path, dataset):

    if not os.path.isdir(new_path): os.makedirs(new_path)

    # TODO: Define mapping from annotation apnea labels to our standards
    d = {'hypopnea':'hypopnea',
             'obstructive apnea':'osa',
             'central apnea':'centrala',
             'mixed apnea':'mixeda'}

    files = os.listdir(path)
    for file in files:
        # TODO: extract subject 
        subject = file.split('-')[-2]
        print(f'*****Processing subject {subject}******')

        infile = open(os.path.join(path, file),"rb")
        contents = infile.read()
        soup = BeautifulSoup(contents,'xml')
        events = soup.find_all('ScoredEvent')
    
        # Gather annotations
        annots = []
        for event in events:
        
            concept = remove_tag(event.EventConcept).split('|')[0]
            concept = concept.lower() # make lowercase
            if 'hypopnea' in concept or 'apnea' in concept or 'central' in concept or 'mixed' in concept:
        
                start = remove_tag(event.Start)
                duration = remove_tag(event.Duration)
                # eventtype = remove_tag(event.EventType)
                # time = remove_tag(event.ClockTime)
                apnea_type = d[concept]

                annots.append([start,duration,apnea_type])

        # TODO: Write Header
        annots.insert(0, ['start', 'duration', 'apnea_type'])
        # TODO: Define output file
        out_file = f'{dataset}_subj{subject}_sess1.csv'
        with open(os.path.join(new_path, out_file), 'w', newline='\n') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(annots)
        

'''Helper function to remove XML tag'''
def remove_tag(text):
    return re.sub('<[^<]+?>', '', str(text))




if __name__ == "__main__":
    

    ''' TO CONVERT EDF SIGNAL TO CSV '''
    # uncomment paths and fuction call to execute  
    old_edf_path = 'edfs_to_process/'
#     new_edf_path = 'raw_data/ssc/stereo/'
#     signals_to_process = ['Nasal Pressure', 'Cannula']
#     spo2_signal = 'SpO2' # name os spo2 header
    dataset = 'zephyr'
    signal = 'stereo'
    new_edf_path = '../../../zephyr_cannula/'
    signals_to_process = ['SpO2','Pressure']
    spo2_signal = 'SpO2' # name os spo2 header
#     spo2_signal = 0
    
    edf2csv(old_edf_path, new_edf_path, signal, signals_to_process, spo2_signal, dataset)

    ''' TO CONVERT XML ANNOTATIONS -> CSV '''
    # old_xml_path = '../expert_annotations/full'
    # new_xml_path = '../expert_annotations/homepaplab/'
    # dataset = 'homepaplab'
    # xml_annotations_to_csv(old_xml_path,new_xml_path, dataset)