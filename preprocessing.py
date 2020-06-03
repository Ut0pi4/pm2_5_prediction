# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:58:19 2020

@author: xiang
"""
import os
import sys

import numpy as np
import urllib.request
import tensorflow as tf

import zipfile


SOURCE_URL = "https://cloud.tsinghua.edu.cn/d/f519a587a6d943fa9aa0/files/?p=%2F%E5%8C%97%E4%BA%AC%E7%A9%BA%E6%B0%94%E8%B4%A8%E9%87%8F.zip&dl=1"

def maybe_download(filename, work_directory):
    """Download the data from website, unless it's already here."""
    if not tf.io.gfile.exists(work_directory):
        tf.io.gfile.makedirs(work_directory)
    filepath = os.path.join(work_directory, filename)
    # set_trace()
    if not tf.io.gfile.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL, filepath)
    with tf.io.gfile.GFile(filepath) as f:
        print('Successfully downloaded', filename)
    return filepath

def extract_files(filepath):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    # filepath = Path(filepath)
    print('Extracting', filepath)
    #if not tf.io.gfile.exists(filepath):
    #    tf.io.gfile.makedirs(filepath)
        
    with zipfile.ZipFile(filepath+".zip", 'r') as zip_ref:
        #zip_ref.extractall("../air_quality/")
        
        for member in zip_ref.infolist():
            # set_trace()
            
            member.filename = member.filename.encode("cp437").decode("utf8")
            #print(member.filename)
            #set_trace()
            
            # outputfile = open("../air_quality/asd2", "wb")
            # shutil.copyfileobj(zip_ref.open(member), outputfile)
            # outputfile.close()
            #set_trace()
            zip_ref.extract(member, "../air_quality/")
    #os.chdir(filepath) # change directory from working dir to dir with files

    for item in os.listdir(filepath): # loop through items in dir
        #set_trace()
        if item.endswith(".zip"): # check for ".zip" extension
            #file_name = os.path.abspath(item) # get full path of files
            zipfile_path = os.path.join(filepath, item)
            zip_ref = zipfile.ZipFile(zipfile_path) # create zipfile object
            zip_ref.extractall(filepath) # extract file to dir
            zip_ref.close() # close file
            #os.remove(file_name) # delete zipped file

def process():
    dest = "../air_quality"
    file_name = "北京空气质量.zip"
    filepath = maybe_download(file_name, dest)
    filepath = filepath[:-4]
    extract_files(filepath)

if __name__=="__main__":
    process()
