# Face-Detection
Hello welcome to my face detection,
its a face detection model using the famous widerface dataset 

Note: sorry its not ready yet but it will be ready very soon :) 


# Installation

* Clone the Repo : `$ git clone https://github.com/AbdallahOmarAhmed/face-mask-detection`

* Install requirements : `$ pip install -r requirements.txt`

* Setup evaluation : `$ python3 eval/setup.py build_ext --inplace`

# Train
* Download wider face dataset and put it in the project dir :
 ``` 
$ gdown https://drive.google.com/uc?id=15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M
$ gdown https://drive.google.com/uc?id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q
$ wget shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip
 ```
* Run the train.py file : `$ python3 train.py [batch size]`

# Evaluation

* Make predictions : `$ python3 make_prediction.py [your model path]`

* Test predictions : `$ python3 eval/evaluation.py`


# Current Accurace

wider face easy : 88.8%

wider face medium : 82.6%

wider face hard : 52%

still working on it 


