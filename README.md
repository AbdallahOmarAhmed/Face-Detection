# Face-Detection
Hello welcome to my face detection,

its pytorch implementation of face detection system using yolo techniques

# Installation

* Install PyTorch : https://pytorch.org/

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
###  Pre trained models:
back bone | epoch time (rtx-2070) | easy accurace | medium accurace | hard accurace | resolution
----------|------------|----------|-----------|-----------|--------
[resnet18](https://drive.google.com/file/d/1es76BbJn1Wofc9AsdR4EC123N5dE4Fyi/view?usp=sharing)|154 sec|88.4%|83.4%|56.7%|448 * 448

###  Test your own model: 

* Make predictions : `$ python3 make_prediction.py [your model path]`

* Test predictions : `$ python3 eval/evaluation.py`

# Demo

### Test an image:
`$ python3 test_image.py [model path] [image path]`

![image_screenshot_١٧ ١١ ٢٠٢١](https://user-images.githubusercontent.com/49597655/142192854-a458cc3c-b738-45fa-9c97-1e4b70ea106d.png)

