# Mnist
Deep Learning Assignment
인공지능과 딥러닝 1차 과제

1. 제공된 Mnist 데이터를 전처리하는 파이프라인 만들어서 dataset.py 에 작성. 
2. LeNet-5 과 Custom-MLP 모델을 model.py에 작성. 
  - LeNet-5 파라미터 수 
    C1: 5*5*6+6  =156 
    C3: 5*5*6*16+16 = 2416 
    C5: (5*5*16)*120+120 = 48120 
    F6: (120*84)+84 = 10164 
    F7: (84*10)+10 = 850 
    (pooling 에서는 학습 가능한 파라미터가 없음) 
    156 + 2,416 + 48,120 + 10,164 + 850 = 61,706 
    총 파라미터 수 : 61,706 
  - Custom-MLP 파라미터 수 
    F1: 44 * (1024 + 1) = 44,660 
    F2: 120 * (44 + 1) = 5,340 
    F3: 84 * (120 + 1) = 10,164 
    F4: 10 * (84 + 1) = 850 
    44,660 + 5,340 + 10,164 + 850 = 61,014 
    총 파라미터 수 : 61,014 
    (fully connected layer은 LeNet-5와 같게 설정하고, LeNet-5의 CNN의 파라미터 수와 비슷하게 MLP의 앞단을 fully connected layer로 수정) 
3. main.py에 LeNet-5 과 Custom-MLP 모델을 학습하는 코드 작성. 학습과정을 모니터링 하기 위해서 각각의 epoch 끝에 Loss 와 Accuracy 출력. 
4. Validation data 없이 Train data, Test data에 대한 Loss와 Accuracy 플롯하여 비교. 
5. LeNet-5과 Custom-MLP의 예측성능 비교. 학습에 사용한 LeNet-5의 Accuracy와 알려진 LeNet-5의 Accuracy가 비슷한지 확인. 
6. LeNet-5의 성능을 향상시킬 수 있는 정규화 기법을 사용하고 성능이 향상된 것을 증명. 
  - Data Augmentation 사용 
    1) 이미지를 30도 이내로 회전.
    2) 이미지를 상하좌우 최대 10% 평행이동.
    3) 무작위로 잘라서 원본 이미지의 80%에서 100%로 크기 조정
  - Batch Normalization 사용 
    LeNet-5의 첫번째와 두번째 컨볼루션 레이어 뒤에서 Batch Normalization을 사용하여 변형된 분포가 나오지 않도록 조절. 
    Batch Normalization은 감마(Scale), 베타(Shift) 를 통한 변환을 통해 비선형 성질을 유지 하면서 학습 될 수 있게 해줌. 

  
