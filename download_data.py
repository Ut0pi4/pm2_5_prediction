import os
import urllib
import zipfile
import argparse
from pdb import set_trace
import tensorflow as tf

SOURCE_URL = "https://cloud.tsinghua.edu.cn/d/f519a587a6d943fa9aa0/files/?p=%2F%E5%8C%97%E4%BA%AC%E7%A9%BA%E6%B0%94%E8%B4%A8%E9%87%8F.zip&dl=1"


def maybe_download(filename, work_directory):
    """Download the data from website, unless it's already here."""
    if not tf.io.gfile.exists(work_directory):
        tf.io.gfile.makedirs(work_directory)
    filepath = os.path.join(work_directory, filename)
    # set_trace()
    if not tf.io.gfile.exists(filepath):
        print("downloading dataset....")
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
    split = filepath.split("/")
    dest = "../" + split[1][:-7]
    # set_trace()
    with zipfile.ZipFile(filepath+".zip", 'r') as zip_ref:
        #zip_ref.extractall("../air_quality/")
        
        for member in zip_ref.infolist():
            # set_trace()
            
            member.filename = member.filename.encode("cp437").decode("utf8")

            zip_ref.extract(member, dest)
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

def download_and_extract(file_name, dest):
    filepath = maybe_download(file_name, dest)
    filepath = filepath[:-4]
    extract_files(filepath)

    return filepath


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="PM2.5 Prediction")
    # parser.add_argument('--img-mode', type=int, default=1, help='set 1 to run on image, 0 to run on video.')
    parser.add_argument('--dest', type=str, default="../air_quality", help='path to dataset.')
    # parser.add_argument('--img-path', type=str, help='path to your image.')
    # parser.add_argument('--video-path', type=str, default='0', help='path to your video, `0` means to use camera.')
    # parser.add_argument('--hdf5', type=str, help='keras hdf5 file')
    args = parser.parse_args()

    file_name = "北京空气质量.zip"
    dest = args.dest
    download_and_extract(file_name, dest)
