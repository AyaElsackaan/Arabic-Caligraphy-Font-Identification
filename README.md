# Arabic-Caligraphy-Font-Identification

An implementation of a classifier that distinguishes between 9 different arabic fonts.

This is CMPN450 course project - Faculty of Engineering, Cairo University.

## Implementation
Local Phase Quantaization (LPQ) feature is calculated for each test image and used to train a Support Vector Machine (SVM).
## To use this code

To install needed packages use:
```sh
pip install -r requirements.txt
```
To run the trained model on testcases in /test use:
```sh
python predict.py
```

To calculate the accuracy use:
```sh
python evaluate.py
```

## Input Files:
### /test
This is the directory where the test images are located. The output is ordered ascendingly according to the file names.

### ground_truth.txt
Contains the actual labels of the test images, each in a seperate line.


## Output Files (/out):

### /out/results.txt
Contains the class of each test image in ascending order according to the image name, each in a seperate line.

### /out/time.txt
Contains the classification time of each test image in ascending order according to the image name, each in a seperate line.
