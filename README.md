# Required Python packages
OpenCV , Pytorch, NumPy, Requests, Json.

# DLRealTimeParkingSystem
Code for our real-time parking system, plus a notebook with code snippets for the training and the evaluation of the DL models.

**classifier.py:** here you'll find code for classifying parking spaces based on a given deep learning model.

**imageProcessing.py:** prepare the images of parking spaces for the classifier.

**updateParkData.py:** update the database with the new parking lot state.

**parking.py:** periodically classify images (preprocessed using *imageProcessing.py* and then classified using *classifier.py*) of parking spaces and update the database using *updateParkData.py*.

**DLModelNotebook.ipynb:** a notebook with the training and evaluation code of our DL models.

# App and Dashboard repos

https://github.com/med4it/parkme-mobile

https://github.com/med4it/smartpark-admin
