import sys
import os
import joblib
import time
from PIL import Image

from feature_extraction import *
from pre_processing import *

MODEL=joblib.load('LPQ_BAS_FINAL.sav') #loading the model

INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]

def load_images_from_folder(folder):
    images = []
    files=[int(filename.split(".")[0]) for filename in os.listdir(folder)]
    files.sort()
    for filename in files:
        print ('reading ',filename)
        img = Image.open (os.path.join(folder,str(filename)+'.png'))
        img = np.array(img.convert ("L")) #loading the test 
        if img is not None:
            images.append(img)

        # print("opened", np.array(images).shape, os.path.join(folder,filename))
    return images


def predict(test_image):
    start=time.time()

    # preprocessing
    pre_processed_image=pre_process(test_image) 

    #perfroming the feature extraction
    lpq_image=lpq(pre_processed_image,winSize=11) 
    prediction = MODEL.predict(lpq_image.reshape(1,-1))

    end=time.time()
    execution_time = end-start
    return prediction, execution_time
    

def main():
    # read test_data
    X_test=load_images_from_folder(INPUT_DIR)

    # predict
    predictions_time=[predict(img) for img in X_test]
    # print(predictions_time)

    # output files
    results = open(OUTPUT_DIR+"/results.txt", "w")
    time_file = open(OUTPUT_DIR+"/times.txt", "w")

    for i in range(len(predictions_time)-1):
        results.write(str(int(predictions_time[i][0]))+'\n') 

        t=round(predictions_time[i][1],2) 
        if t==0:
            t=0.001
        time_file.write(str(t)+'\n') 

    # write last element seperately to avoid extra blank lines
    results.write(str(int(predictions_time[i+1][0])))

    t=round(predictions_time[i+1][1],2) 
    if t==0:
        t=0.001
 
    time_file.write(str(t)) 

    # close files
    results.close()
    time_file.close()


if __name__=="__main__":
    main()
