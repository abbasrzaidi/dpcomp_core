import pandas as pd 
import numpy as np 
import os, sys
from datetime import datetime 
import re
from shutil import copyfile

# '''
# returns dataframe having opened given file
# assumes file has already been cleaned (connections < 5 min and > 24 hrs cut out)
# '''
# def open_cleaned_file(file_name):
#     columns = ["mac_address","username","ap_name","connect_time","disconnect_time","duration","SSID", "derived"]
#     df = pd.read_csv(file_name,names=columns)
#     return df

# '''
# calculates building aggregates from cleaned files and saves as new csv 
# saved as | ap | building_name | count | 
# '''
# def save_building_aggregates(df):
#     aps = df['ap_name']
#     buildings = [re.split('-(ap|AP)-[0-9]*', ap)[0] for ap in aps] # using regex to grab building name from ap
#     buildings[0] = 'building' # have to set first entry in file manually
#     df['building'] = buildings
#     aggregates = df.groupby(['ap_name', 'building']).size().to_frame(name='count').reset_index() # buildings are sorted alphabetically
#     saved_file_name = 'examples/counts/building_aggregates.csv'
#     aggregates.to_csv(saved_file_name, index=False, header=False)
#     return saved_file_name, len(set(df['building'])) # need number of unique buildings for domain size 

# '''
# pulls building counts from file and saves as .npy  
# '''
# def save_as_npy(file_name):
#     df = pd.read_csv(file_name, names=['ap_name', 'building', 'count'])
#     df = df.groupby('building').size()
#     building_counts = [i for i in df]
#     print(building_counts)
#     file_name = 'building_counts.npy'
#     np.save(file_name, building_counts)
#     return file_name


# # # execution
# # biggest_old_data = "examples/counts/anon_UPC_sessions_1604045746.csv"
# # new_data = "examples/counts/anon_UPC_sessions_1619511399.csv"

# # df = open_cleaned_file(new_data)
# # saved_file_name, domain = save_building_aggregates(df)
# # saved_npy_name = save_as_npy(saved_file_name)
# # copyfile(saved_npy_name, 'dpcomp_core/datafiles/1D/' + saved_npy_name) # copies file into regular dpcomp data 
# # print('domain size:', domain) # number of unique buildings 

# # LAST STEP: in dataset.py, add entry for nickname: location 
#     # location should just be '1D/saved_npy_name'

# ##############################################################################################################
'''
Converts to minutes
'''
def convert_to_min(duration_as_str):
    if duration_as_str is None:
        return 60
    x = str(duration_as_str).split(' ')
    if(len(x) > 2) :
        return int(x[0])*60 + int(x[2])
    else :
        return int(x[0])

'''
Replaces empty disconnect times, denoted with tab delimeter, with midnight of the current day
'''
def filterDates(df, date):
    date = datetime.utcfromtimestamp(int(date)).strftime('%-m/%-d/%Y') # date is in unix timestamp, need to get in M/D/YYYY format
    df['connect_time'] =  pd.to_datetime(df['connect_time']) #convert to machine readable date_time
    df = df.replace('\t-', date + ' 11:59 PM PDT') 
    return df

'''
Removes connectons < 5 mins and > 24 hrs.
Helps to remove stationary devices that don't represent true users
'''
def filterDuration(df): # TODO: modify this to fully account for non user connections
    df['disconnect_time'] =  pd.to_datetime(df['disconnect_time'])
    df = df[df['connect_time'] > df['connect_time'].max() - pd.Timedelta(pd.offsets.Day(1))] # 24 hour filtering
    df['duration'] = df['duration'].apply(convert_to_min)
    df = df[df['duration'] > 5] # 5 minute filtering
    return df

'''
Reads in raw data (examples/counts/raw/), filters on date & connection duration, saves as csv in examples/counts/clean/ folder.
'''
def cleanFile(file_name):
    columns = ["mac_address","username","ap_name","connect_time","disconnect_time","duration","SSID"]
    date = re.split('_', file_name)[3][:-4]
    df = pd.read_csv('examples/counts/raw/' + file_name, names=columns, header=None)
    df = filterDates(df, date)
    df = filterDuration(df)
    df.to_csv('examples/counts/clean/' + file_name, header=None, index=None)

'''
Calculates aggregates from clean files and saves as .npy in examples/counts/prepped/...
'''
def aggregateCounts(file_name):
    columns = ["mac_address","username","ap_name","connect_time","disconnect_time","duration","SSID", "derived"]
    df = pd.read_csv('examples/counts/clean/' + file_name, names=columns, header=None)
    df['building'] = df['ap_name'].str.replace(r'-(.*)', '', regex=True).str.upper() # adding building field
    building_counts = df.groupby('building').size().to_frame(name='count')
    counts = building_counts['count'].values
    npy_name = file_name[:-4] + '.npy'
    np.save('examples/counts/prepped/' + npy_name, counts)
    copyfile('examples/counts/prepped/' + npy_name, 'dpcomp_core/datafiles/1D/building_data.npy') # copies file into regular dpcomp data 
    return len(counts)

# set file_name to the raw file's name (stored in examples/counts/raw/...) and run
file_name = 'anon_UPC_sessions_1619511399.csv'
cleanFile(file_name)
print('num buildings:', aggregateCounts(file_name)) # need this to set the domain size in algorithm_execution.py