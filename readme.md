###### 2020_Capstone_02 Repository

### 1. 프로젝트 주제.

영상 자료를 이용한 감정인식 소프트웨어.

### 2. 프로젝트 제안 방법.

영상을 이용한 감정 인식 소프트웨어를 새로이 개발하여, 

### . 프로젝트에 사용한 데이터셋.

RAVDESS

### . 프로젝트 실행 과정.
step 1: getFrames.py: 영상 파일에서 프레임을 추출하여 데이터 폴더에 저장.
step 2: data_rearrange.py: 데이터 폴더에 저장된 자료를 이용하여 'data_file.csv' 파일을 생성.
step 3: extract_features.py: 
step 4: train.py: 기본 모델을 이용하여 학습 실행. (models.py 파일을 이용하여 학습하고자 하는 모델 생성 가능
step 5: test.py: 실행. 결과값을 MSE(mean squared error-딥러닝 손실함수)로 나타냄.

### . 프로젝트에 사용한 개발 환경.

실험은 Intel Xeon Silver 4214R 프로세서, 
Nvidia Quadro RTX 8000 48GB (2Way) 그래픽 프로세서, 
96GB RAM, Ubuntu 20.04 LTS 컴퓨터를 이용. 

그래픽 프로세서를 이용한 GPGPU를 위하여 CUDA10.1 및 cuDNN8.0.4를 이용. 

실험을 위해 python 3.6.9, numpy 1.18.5, opencv-python 4.4.0.44, tensorflow 1.15.4, 
keras 2.2.4, keras-vggface 0.5, librosa 0.8.0, matplotlib 3.3.2, scikit-learn 0.23.2를 이용. 

### .
