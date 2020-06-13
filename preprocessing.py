import os
import csv
import copy

import argparse
import numpy as np
from download_data import download_and_extract

REMOVED_CSV = {}
REMOVED_CSV["2014"] = {"beijing_all_20141231.csv", "beijing_extra_20141231.csv"}
REMOVED_CSV["2015"] = set()
REMOVED_CSV["2016"] = set()
REMOVED_CSV["2017"] = set()
REMOVED_CSV["2018"] = set()
REMOVED_CSV["2019"] = set()
REMOVED_CSV["2020"] = set()

def normalize_data_pm2_5(data):
    data_new = copy.deepcopy(data)
    data_new[data_new > 250] = -5
    data_new[data_new > 150] = -4
    data_new[data_new > 115] = -3
    data_new[data_new > 75] = -2
    data_new[data_new > 35] = -1
    data_new[data_new > 0] = 0
    data_new = abs(data_new).astype(int)
    
    return data_new

def process_data(data, pm2_5=0):
    N, d = data.shape
    
    new_data = np.zeros((N, d))
   
    m = []
    for j, row in enumerate(data):
        row = [float(i) if i != "" else 0 for i in row]
        row = np.array(row)
       
        if np.any(row==0):
            index = np.where(row==0)
            if len(index[0]) == len(row):
                m.append(j)
                continue
                
            mu = np.sum(row)/(len(row)-len(index[0]))
            row[index[0]] = mu
            if mu <= 0:
                set_trace()
        new_data[j, :] = row
        
        
    if m:
        for ind in m:    
            new_data[ind] = np.sum(new_data, axis=0)/(len(new_data)-len(m))
    
    if pm2_5:
        new_data = normalize_data_pm2_5(new_data)
    
    return new_data



def check_data(row, item, year):
    check = [True if i=="" else False for i in row[3:]]
    check_true = True
    if np.all(np.array(check)):
        if item not in REMOVED_CSV[year]:
           
            check_true = False
    return check_true

def add_removed(item, year):
    split = item.split("_")
    split[-2] = "extra"
    item_1 = "_"
    item_1 = item_1.join(split)
    REMOVED_CSV[year].add(item_1)
    split[-2] = "all"
    item_2 = "_"
    item_2 = item_2.join(split)
    REMOVED_CSV[year].add(item_2)
    

def restructure_data(data):
    num_doc = len(data)

    N, d = data[0].shape
    
    new_data = np.zeros((num_doc, N, d))
    
    for i, dat in enumerate(data):
        new_data[i, :, :] = dat
    return new_data

def clean(folderpath, consistent=False):

    print("start reading csv in folder: ", folderpath)
    pm2_5s = []
    SO2s = []
    NO2s = []
    COs = []
    O3s = []
    year = folderpath.split("-")[1][:4]
    
    for item in os.listdir(folderpath):
        
        split = item.split("_")
        if "beijing" not in item:
            REMOVED_CSV[year].add(item)
            continue
        if consistent:
            if split[-2] == "extra":
                split_1 = split
                split_1[-2] = "all"
                item_1 = "_"
                item_1 = item_1.join(split_1)

                if item_1 not in os.listdir(folderpath):
                    REMOVED_CSV[year].add(item)
            else:
                split_2 = split
                split_2[-2] = "extra"
                item_2 = "_"
                item_2 = item_2.join(split_2)
                if item_2 not in os.listdir(folderpath):
                    REMOVED_CSV[year].add(item)
        
        
        if item in REMOVED_CSV[year]:
            continue
        
        filepath_csv = folderpath + "/" + item
        
        pm2_5 = []
        SO2 = []
        NO2 = []
        CO = []
        O3 = []
        d = 0
        
        with open(filepath_csv, encoding="utf8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            prev_header = []
            for i, row in enumerate(csv_reader):
                try:
                    row[2]=="SO2"
                        
                except:
                    if item not in REMOVED_CSV[year]:
                        add_removed(item, year)
                    break
                
                if len(row[3:]) != 35:
                    if item not in REMOVED_CSV[year]:
                        add_removed(item, year)
                    break

               
                    
                if i==0:
                    header = row[3:]
                    if prev_header:
                        if header != prev_header:
                            set_trace()
                            break
                    prev_header = header
                    continue
                else:
                    if row[2] in ["PM2.5", "SO2", "NO2", "CO", "O3"]:
                        if not check_data(row[3:], item, year):
                            add_removed(item, year)
                            break
                        else:
                            if row[2]=="PM2.5":
                                pm2_5.append(row[3:])
                            elif row[2]=="SO2":
                                SO2.append(row[3:])
                            elif row[2]=="NO2":
                                NO2.append(row[3:])
                            elif row[2]=="CO":
                                CO.append(row[3:])
                            elif row[2]=="O3":
                                O3.append(row[3:])

        split = filepath_csv.split("_")
        if split[-2] == "all":
            if len(pm2_5) != 24:
                add_removed(item, year)
                continue
        elif split[-2] == "extra":
            if len(SO2) != 24:
                add_removed(item, year)
                continue
            if len(NO2) != 24:
                add_removed(item, year)
                continue
            if len(CO) != 24:
                add_removed(item, year)
                continue
            if len(O3) != 24:
                add_removed(item, year)
                continue
    print("complete read folder")

def retrieve_data_alls_extras(folderpath):

    print("start reading csv in folder: ", folderpath)
    
    pm2_5s = []
    SO2s = []
    NO2s = []
    COs = []
    O3s = []
    
    year = folderpath.split("-")[1][:4]
    items = []
    alls = []
    extras = []
    
    for item in os.listdir(folderpath):
        if "beijing" not in item:
            continue
        if item not in REMOVED_CSV[year]:
            items.append(item)

    for item in items:
        split = item.split("_")
        if split[-2] == "all":
            split_1 = split
            split_1[-2] = "extra"
            item_1 = "_"
            item_1 = item_1.join(split_1)
            if item_1 in items:
                extras.append(item_1)
                alls.append(item)
    
    alls.sort()
    extras.sort()
    assert(len(extras)==len(alls))
    extra_all_pairs = []
    
    
    for i in range(len(alls)-1):
        split_1 = alls[i].split("_")
        num_csv_1 = (int)(split_1[-1][:8])
        split_2 = extras[i].split("_")
        num_csv_2 = (int)(split_2[-1][:8])
        split_3 = alls[i+1].split("_")
        num_csv_3 = (int)(split_3[-1][:8])
        if num_csv_1 == num_csv_2 and num_csv_2 + 1 == num_csv_3:
            extra_all_pairs.append((alls[i], extras[i], alls[i+1]))

        
        
    
    for (item_1, item_2, item_3) in extra_all_pairs:

        
        filepath_csv_1 = folderpath + "/" + item_1
        filepath_csv_2 = folderpath + "/" + item_2
        filepath_csv_3 = folderpath + "/" + item_3
        
        
        pm2_5 = []
        SO2 = []
        NO2 = []
        CO = []
        O3 = []
        d = 0
        
        with open(filepath_csv_1, encoding="utf8") as csv_file_1:
           
        
            csv_reader_1 = csv.reader(csv_file_1, delimiter=",")

            for i, row in enumerate(csv_reader_1):

                if i==0:
                    header = row[3:]
                    continue
                else:
                    if row[2] in ["PM2.5", "SO2", "NO2", "CO", "O3"]:
                        if not check_data(row[3:], item, year):
                            add_removed(item, year)
                            break
                        else:
                            if row[2]=="PM2.5":
                                pm2_5.append(row[3:])
                            
        with open(filepath_csv_2, encoding="utf8") as csv_file_2:
            csv_reader_2 = csv.reader(csv_file_2, delimiter=",")

            for i, row in enumerate(csv_reader_2):

                if i==0:
                    header = row[3:]
                    continue
                else:
                    if row[2] in ["PM2.5", "SO2", "NO2", "CO", "O3"]:
                        if not check_data(row[3:], item, year):
                            add_removed(item, year)
                            break
                        else:
                            
                            if row[2]=="SO2":
                                SO2.append(row[3:])
                            elif row[2]=="NO2":
                                NO2.append(row[3:])
                            elif row[2]=="CO":
                                CO.append(row[3:])
                            elif row[2]=="O3":
                                O3.append(row[3:])
        
        with open(filepath_csv_3, encoding="utf8") as csv_file_3:
            csv_reader_3 = csv.reader(csv_file_3, delimiter=",")

            for i, row in enumerate(csv_reader_3):

                if i==0:
                    header = row[3:]
                    continue
                else:
                    if row[2] in ["PM2.5", "SO2", "NO2", "CO", "O3"]:
                        if not check_data(row[3:], item, year):
                            add_removed(item, year)
                            break
                        else:
                            
                            if row[2]=="PM2.5":
                                pm2_5.append(row[3:])
                        

        if pm2_5:
            pm2_5 = np.array(pm2_5)
            pm2_5 = process_data(pm2_5, 1)
            pm2_5s.append(pm2_5)

  

        if SO2:
            SO2 = np.array(SO2)
            SO2 = process_data(SO2)
            SO2s.append(SO2)

        if NO2:
            NO2 = np.array(NO2)
            NO2 = process_data(NO2)
            NO2s.append(NO2)

        if CO:
            CO = np.array(CO)
            CO = process_data(CO)
            COs.append(CO)

        if O3:
            O3 = np.array(O3)
            O3 = process_data(O3)
            O3s.append(O3)
    
    pm2_5s = restructure_data(pm2_5s)
    
    SO2s = restructure_data(SO2s)
    NO2s = restructure_data(NO2s)
    COs = restructure_data(COs)
    O3s = restructure_data(O3s)
    
    print("complete read folder")
    return pm2_5s, SO2s, NO2s, COs, O3s

def normalize_concat(arr_data):
    norm_arr_data = []
    for data in arr_data:
        norm_arr_data.append(normalize(data))
    return concat(norm_arr_data)

def normalize(data):
    new_data = []
    for dat in data:
        temp = (dat-np.min(dat))/(np.max(dat)-np.min(dat))
        new_data.append(temp)
    
    return new_data

def concat(arr_data):
    years = len(arr_data[0])
    new_data = []
    for year in range(years):
        temp = []
        for data in arr_data:
            temp.append(data[year])
        
        new_data.append(np.concatenate(temp, 0))
    return new_data

def concat_years(arr_data):
    years = len(arr_data)
    sum_doc = 0
    N, h, f = arr_data[0].shape
    for year in range(years):
        sum_doc += arr_data[year].shape[0]
    new_data = np.zeros((sum_doc, h, f))
    ind = 0
    for year in range(years):
        ind_last = arr_data[year].shape[0]
        new_data[ind:ind+ind_last, :, :] = arr_data[year]
        ind = ind + ind_last
    return new_data    

def preprocess(file_name, dest):

    folderpath = download_and_extract(file_name, dest)
    for key, values in REMOVED_CSV.items():
        print("length of REMOVED_CSV in year %s: %d" %(key, len(values)))
    print("")  


    if os.path.exists(folderpath):
        for item in os.listdir(folderpath):
            if ".zip" not in item and "beijing" in item:
                clean(folderpath+"/"+item)

    for key, values in REMOVED_CSV.items():
        print("length of REMOVED_CSV in year %s: %d" %(key, len(values)))
    print("")

    pm2_5s_years = []
    
    SO2s_years = []
    NO2s_years = []
    COs_years = []
    O3s_years = []
    if os.path.exists(folderpath):
        for item in os.listdir(folderpath):
            if ".zip" not in item and "beijing" in item:
                
                pm2_5s, SO2s, NO2s, COs, O3s = retrieve_data_alls_extras(folderpath+"/"+item)
                pm2_5s_years.append(pm2_5s)
                
                SO2s_years.append(SO2s)
                NO2s_years.append(NO2s)
                COs_years.append(COs)
                O3s_years.append(O3s)     

    feature_data = normalize_concat((SO2s_years, NO2s_years, O3s_years, COs_years))
    
    return feature_data, pm2_5s_years



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PM2.5 Prediction")
    
    parser.add_argument('--dest', type=str, default="../air_quality", help='path to dataset.')
    
    args = parser.parse_args()

    file_name = "北京空气质量.zip"
    dest = args.dest
    
    preprocess(file_name, dest)
    
