AudiGest Model Repository
=========================
Welcome to the repository of Emotional 3D Speech Visualization from 2D Audio Visual Data.

AudiGest is a Deep Learning model to create 3D audiovisual speech animations in 3D face models extracted with [MediaPipe](https://google.github.io/mediapipe/solutions/face_mesh#python-solution-api).

Setup
=====
PyTorch for Linux (1.8.2 LTS - cuda11)
-------------------------------------
The model was developed using this version of PyTorch installed with the instructions from the [official web](https://pytorch.org/get-started/locally/).

```
$ pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```
PSBody - Mesh
-------------------------------------
We use [PSBody - Mesh](https://github.com/MPI-IS/mesh) to creating the mesh objects for rendering task. To install run the commands on your virtual enviroment folder:
```
$ git clone https://github.com/MPI-IS/mesh.git
$ sudo apt-get install libboost-dev
$ sudo apt-get install cmake
```

Then in ```mesh/mesh/cmake/thirdparty.cmake```, fixed line 29 by writing "()" for ```print```:
```
COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
```

Finally on the main directory of the repository ```mesh/``` run the command below:
```
$ make all
```
FFMPEG
-------------------------------------
Install FFMPEG:
```
$ sudo apt install ffmpeg
```
Python Modules
--------------
The Python modules needed to run the code are in ```requirements.txt```. Run in your virtual enviroment.
```
$ pip install -r requirements.txt
```

How to Use
==========
Generate a Speech Animation from Audio
--------------------------------------
Check for the saved state of the model in ```processed_data/training```, then execute the following command:
```
$ python model_inference.py --audio_path audio/MEAD_test_audio.wav --emotion happy --base_face faces/base_face.npy --video_directory videos --video_fname output
```
You can change the command arguments according to the help command:
```
$ python model_inference.py -h
```