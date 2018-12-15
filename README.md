# Detecting Eye Lesions - Mircoaneursyms

This was a six month project to attempt to improve on the current models to be used for Mircoaneursym detection, using a deep cnn that was created.

This repo also contains a sample research poster, named Improvement on the task...ppt and the final presentation for the class.

To run the scripts;

install:
numpy,scipy,tensorflow/tensorflowgpu,pandas,pprint

set up file locations:

generate Train and Eval patches files:
edit lines 10,12,14,47,49,51,54 to point the script where the files are saved if they are not in the same directory.

generate test patches file:

run.py:
edit lines 176 and 232 to decide where to save the model.

testmodel.py:
edit lines 157,273,and 274 to point to the location where the model is saved.

The scripts must be ran in this order;
1. generate train,eval,and test to generate the data.
2. run run.py once all of the data and model save locations have been set up
3. run testmodel.py once the training is done and the script has been pointed to the model location.

Two txt files will be set up, one which has the result of the model, and the other which has the coordinates of the said data.

This model is not working as it should, and needs further work in order to make it better.

Special thanks to professor shan for helping me edit the matlab codes.
