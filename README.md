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
  -result_plot 폴더에 LeNet-5, Custom-MLP, Data Augmentation Batch Normalization LeNet-5의 Loss와 Accuracy 플롯이 있음.
5. LeNet-5과 Custom-MLP의 예측성능 비교. 학습에 사용한 LeNet-5의 Accuracy와 알려진 LeNet-5의 Accuracy가 비슷한지 확인.
  -LeNet-5 최소 Loss : 0.0299, 최대 Accuracy : 0.9918 (Test data)  
  -Custom-MLP 최소 Loss : 0.0797, 최대 Accuracy : 0.9783 (Test data) 
  -LeNet-5은 CNN을 레이어로 사용했기 때문에 Custom-MLP에 비해서 성능이 더 좋음.
  -CNN은 이미지의 공간 구조와 특징을 효과적으로 학습하기 위해 설계되었으며, 이를 위해 계층적인 특징 학습과 매개 변수 공유 등의 기능을 가지고 있음.
6. LeNet-5의 성능을 향상시킬 수 있는 정규화 기법을 사용하고 성능이 향상된 것을 증명. 
  - Data Augmentation 사용 
    1) 이미지를 20도 이내로 회전.
    2) 이미지를 상하좌우 최대 10% 평행이동.
    3) 무작위로 잘라서 원본 이미지의 80%에서 100%로 크기 조정
  - Batch Normalization 사용   
    LeNet-5의 각 레이어 뒤에서 Batch Normalization을 사용하여 변형된 분포가 나오지 않도록 조절.   
    Batch Normalization은 감마(Scale), 베타(Shift) 를 통한 변환을 통해 비선형 성질을 유지 하면서 학습 될 수 있게 해줌.     
  - LeNet-5 최소 Loss : 0.0299, 최대 Accuracy : 0.9918 (Test data)
  - Data Augmentation Batch Normalization LeNet-5 최소 Loss : 0.0178, 최대 Accuracy : 0.9942 (Test data)  
  - 플롯 분석
    1) Data Augmentation Batch Normalization LeNet-5은 1 epoch 부터 50 epoch까지 test accuracy가 train accuracy 보다 높음. Augmentation으로 사용한 20도 이내로 회전, 10% 평행이동, 원본 이미지의 80~100%로 크기 조정 변환이 test dataset을 미리 본 효과를 가져다 준 것으로 추측 됨.
    2) epoch 수가 커질 수 록 train,test data의 Loss가 꾸준히 떨어지는 것으로 보아 Batch Normalization의 정규화 효과로 그레디언트 소실을 줄인 것으로 보임.
  
