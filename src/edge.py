import os, zipfile
# from datetime import date
import datetime
from dateutil.relativedelta import relativedelta
import joblib
import pandas as pd


def getFilelist(foldername):
    filelist = []

    for dirname, _, filenames in os.walk(foldername):
        for filename in filenames:
            filelist.append(os.path.join(dirname, filename))

    return filelist

def getFolderlist(foldername):
    out = [x[0] for x in os.walk(foldername) if len(x[2]) > 0]
    out.remove(foldername)
    return out

def createfolder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)    

def unzipfolder(sourcefolder, targetfolder, req_ext='.csv'):
    zipfilelist = [filename for filename in getFilelist(sourcefolder) if filename.endswith('.zip')]

    for filename in zipfilelist: # loop through items in dir
        zip_ref = zipfile.ZipFile(filename) # create zipfile object
        zip_ref.extractall(targetfolder) # extract file to dir
        zip_ref.close() # close file

    removeList = [filename for filename in getFilelist(targetfolder) if not filename.endswith(req_ext)]
    for filename in removeList:
        os.remove(filename) # delete zipped file    

def getTimeStamp():
    # Get Timestamp as a String
    out = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return out

def save_ml_model(model, filename):
    # Save to file in the current working directory
    joblib.dump(model, filename)

def load_ml_model(filename):    
    # Load from file
    joblib_model = joblib.load(filename)
    return joblib_model

def save_to_HDFStore(data, h5file, dataname):
    h5 = pd.HDFStore(h5file, 'w')
    h5[dataname] = data
    h5.close()

def load_from_HDFStore(h5file, dataname):
    h5 = pd.HDFStore(h5file, 'r')
    data = h5[dataname]
    h5.close()
    return data
