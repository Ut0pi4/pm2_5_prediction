import os
import urllib.request
import zipfile
import argparse
from pdb import set_trace


SOURCE_URL = "https://cloud.tsinghua.edu.cn/d/f519a587a6d943fa9aa0/files/?p=%2F%E5%8C%97%E4%BA%AC%E7%A9%BA%E6%B0%94%E8%B4%A8%E9%87%8F.zip&dl=1"


def maybe_download(filename, work_directory):
    """Download the data from website, unless it's already here."""
    if not os.path.exists(work_directory):
        os.makedirs(work_directory)
    filepath = os.path.join(work_directory, filename)
    # set_trace()
    if not os.path.exists(filepath):
        print("downloading dataset....")
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL, filepath)
    with open(filepath, 'r') as f:
        print('Successfully downloaded', filename)
    return filepath

def extract_files(filepath):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filepath)
    
    split = filepath.split("/")
    dest = os.path.join("../", split[1])
    # dest = "../" + split[1]

    with zipfile.ZipFile(filepath+".zip", 'r') as zip_ref:
        # set_trace()
        for member in zip_ref.infolist(): 
            # set_trace()
            member.filename = member.filename.encode("cp437").decode("utf8")
            zip_ref.extract(member, dest)
    # set_trace()
    filepath = filepath + "/北京空气质量"
    for item in os.listdir(filepath): 
        
        if item.endswith(".zip"): 
            
            zipfile_path = os.path.join(filepath, item)
            zip_ref = zipfile.ZipFile(zipfile_path) 
            zip_ref.extractall(filepath) 
            zip_ref.close() 
    print("extract complete")        

def download_and_extract(file_name, dest):
    filepath = maybe_download(file_name, dest)
    filepath = filepath[:-4]
    extract_files(filepath)

    return filepath


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="PM2.5 Prediction")
    parser.add_argument('--dest', type=str, default="../air_quality", help='path to dataset.')
    args = parser.parse_args()

    file_name = "北京空气质量.zip"
    dest = args.dest
    download_and_extract(file_name, dest)
