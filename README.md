# Criminal Detection

Our system is built to detect criminals and identify them using the pretrained yolo model ```yolov8n-face.pt``` and the siamese netowrk ```torchvision.models```. 

![Results](resources/player_similarity.gif)

## Design Choices:
- The yolo algorithm used is the pretrained ```yolov8n-face.pt``` which detects faces from images or videos.
- To identify the faces detected, we built a siamese network using the pretrained resNet18. The first convolutional layer is changed, and the last layer is  removed to add linear layers instead to get a 128 output size. 


## Setting up the dataset.
The expected format for both the training and validation dataset is the same. Image containing a certain person should be placed in a folder specified for that perosn. The folders for every person are then to be placed within a common root directory (which will be passed to the trainined and evaluation scripts) and it has to be called "archive (2)". The folder structure is also explained below:
```
|--archive (2)
  |--Person1
    |-Image1
    |-Image2
    .
    .
    .
    |-ImageN
  |--Person2
  |--Person3
  .
  .
  .
  |--PersonN
```


## Setting up environment.
The provided setup instructions assume that anaconda is already installed on the system. To set up the environment for this repository, run the following commands to create and activate an environment named 'test'.:
```
conda create -n test python=3.9
conda activate test
```


## Opening the website:
To open the website, run the following commands:
```
git clone https://github.com/yazeedbadaro/grad_final.git
conda activate test
pip install -r requirements.txt
streamlit run main_page.py
```
