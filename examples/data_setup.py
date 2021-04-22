import pandas as pd 
import numpy as np 
import os, sys
from datetime import datetime 
import re
from shutil import copyfile

'''
returns dataframe having opened given file
assumes file has already been cleaned (connections < 5 min and > 24 hrs cut out)
'''
def open_cleaned_file(file_name):
    columns = ["mac_address","username","ap_name","connect_time","disconnect_time","duration","SSID", "derived"]
    df = pd.read_csv(file_name,names=columns)
    return df

'''
calculates building aggregates from cleaned files and saves as new csv 
saved as | ap | building_name | count | 
'''
def save_building_aggregates(df):
    aps = df['ap_name']
    buildings = [re.split('-(ap|AP)-[0-9]*', ap)[0] for ap in aps] # using regex to grab building name from ap
    buildings[0] = 'building' # have to set first entry in file manually
    df['building'] = buildings
    aggregates = df.groupby(['ap_name', 'building']).size().to_frame(name='count').reset_index() # buildings are sorted alphabetically
    saved_file_name = 'examples/counts/building_aggregates.csv'
    aggregates.to_csv(saved_file_name, index=False, header=False)
    return saved_file_name

'''
pulls building counts from file and saves as .npy  
'''
def save_as_npy(file_name):
    df = pd.read_csv(file_name, names=['ap_name', 'building', 'count'])
    df = df.groupby('building').size()
    building_counts = [i for i in df]
    file_name = 'building_counts.npy'
    np.save(file_name, building_counts)
    return file_name


# execution
file_name = "examples/counts/anon_UPC_sessions_1604045746.csv"
df = open_cleaned_file(file_name)
saved_file_name = save_building_aggregates(df)
saved_npy_name = save_as_npy(saved_file_name)
copyfile(saved_npy_name, 'dpcomp_core/datafiles/1D/' + saved_npy_name) # copies file into regular dpcomp data 
# LAST STEP: in dataset.py, add entry for nickname: location 
    # location should just be '1D/saved_npy_name'