## File and Folder
 - bytetrack -> Tracking module
 - lib 
    - AI MODEL : best640_640.onnx 
    - ondevice inference module : lib forder in detUtils.py
    - font : lib forder in NanumGothic-Bold.ttf
    - warningsound : lib forder in warningsound.wav
 
 - animal_pred.py : main script
 - setup.py : exe build script
 - requirements.txt : requirements

anaconda download and Anaconda Prompt 

## venv

- conda create -n animal python=3.12

- conda activate animal

## requirements

 - cd animal

 - pip install -r requirements.txt

## make exe


- python setup.py build

## after build
build complete 
1. build -> exe.win-amd64-3.12 best640_640.onnx, NanumGothic-Bold, warningsound.wav file in lib  
2. animal_pred.exe start!!!

