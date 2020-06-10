
import os
import urllib
import zipfile

from download_data import download_and_extract

import tensorflow as tf



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

            zip_ref.extract(member, "../air_quality/")
#             set_trace()
# #             if member.filename == "北京空气质量"：
# #                 os.rename(member.filename, "pm_2_5_data")
#             set_trace()
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

def download_and_extract():
    dest = "../air_quality"
    file_name = "北京空气质量.zip"
    filepath = maybe_download(file_name, dest)
    filepath = filepath[:-4]
    extract_files(filepath)
