# Diego González Serrador TFM
Understanding badminton with computer vision is the Final Master Thesis from Diego González Serrador.

# Setup instructions

1. Install python >= 3.6
2. [Optional] Setup a new virtual environment.
3. Setup [object detection framework](https://github.com/tensorflow/models/blob/master/research/object_detection), already included in this repo:
    ```bash
    # From root dir
    pip install -r requirements.txt
    sudo apt-get install protobuf-compiler
    protoc object_detection/protos/*.proto --python_out=.
    ```
4. Coco api installation
    ```bash
    # From root dir
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    cp -r pycocotools <path_to_project>/
    ```