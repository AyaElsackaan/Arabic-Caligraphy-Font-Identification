import sys
import os

from commonfunctions import *
import joblib
import imageio as iio
import cv2
from skimage import filters
from skimage.color import rgb2gray  # only needed for incorrectly saved images
from skimage.measure import regionprops
from skimage import data, color, feature,morphology
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from feature_extraction import *
from pre_processing import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC,SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
import json

model=joblib.load('final_voting.sav')
model_pca=joblib.load('pca_model.sav')

INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def get_mean_std(filename):

    f = open(filename) 
    data = json.load(f)
    mean=data['mean']
    std=data['std']
    f.close()
    return mean, std

def predict(test_image):
    start=time.time()
    binarized_image=cv2.resize(np.array(crop_image(pre_process(test_image)), dtype='uint8'), (500,500), interpolation = cv2.INTER_AREA)
    values=HVSL(np.uint8(test_image))
    center_of_masses=center_of_mass(binarized_image)
    base_line_feature2=HPP_Skeletonize(binarized_image)
    ones=count_ones(binarized_image)
    combined_array=np.array([values,center_of_masses,base_line_feature2,ones]).reshape(1,-1)
    GLCM=np.array(GLCM_features(binarized_image)).reshape(1,-1)
    GABOR=gabor_filter(test_image.reshape(test_image.shape[0],test_image.shape[1])).reshape(1,-1)
    lbp=LBP(test_image,numPoints=24,radius=3,method="uniform",window=500).reshape(1,-1)
    patches_hog_skeleton=feature.hog(morphology.skeletonize(binarized_image).astype(int),cells_per_block=(3, 3),pixels_per_cell=(150, 150), orientations=6).reshape(1,-1)
    patches_hog=feature.hog(test_image,cells_per_block=(3, 3),pixels_per_cell=(150, 150), orientations=7).reshape(1,-1)
    feature_vector=np.concatenate((patches_hog.reshape(1,-1),patches_hog_skeleton.reshape(1,-1),combined_array.reshape(1,-1),GLCM.reshape(1,-1),lbp,GABOR),axis=-1).reshape(1,-1)
    #print(feature_vector.shape)
    mean,std=get_mean_std('mean_std.json')
    feature_vector=(feature_vector-mean)/std
    pca_output=model_pca.transform(feature_vector)
    prediction=model.predict(pca_output)
    end=time.time()
    #print(prediction)
    duration = end-start
    return prediction, duration
    

def main():
    #read test_data
    X_test=load_images_from_folder(INPUT_DIR)

    #preprocessing
    for i in range(len(X_test)):
        X_test[i]=tf.image.rgb_to_grayscale( X_test[i])
        X_test[i]=cv2.resize(np.array(X_test[i]), (500,500), interpolation = cv2.INTER_AREA)
        X_test[i]=X_test[i].reshape((500,500,1)).astype('float32')
    
        print(len(X_test), X_test[i].dtype)

    # predict
    # show_images(X_test)
    predictions_time=[predict(img) for img in X_test]
    print(predictions_time)

    # output file
    results = open(OUTPUT_DIR+"/results.txt", "w")
    for i in range(len(predictions_time)-1):
        results.write(str(int(predictions_time[i][0])+1)+'\n') 
    results.write(str(int(predictions_time[i][0])+1)) 

    results.close()

    time_file = open(OUTPUT_DIR+"/time.txt", "w")
    for i in range(len(predictions_time)-1):
        t=round(predictions_time[i][1],2) 
        if t==0:
            t=0.001
        time_file.write(str(t)+'\n') 

        t=round(predictions_time[i][1],2) 
        if t==0:
            t=0.001
        time_file.write(str(t)+'\n') 
        
    time_file.close()


if __name__=="__main__":
    main()
