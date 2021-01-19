# Email-spam-classification


# 프로그램 소개
머신러닝(Naive Bayes_MultinomialNB, SVM), 신경망(MLP), 의사결정나무(Random forest) 모형을 이용한 스팸 이메일 분류 모델입니다.
모델 적합과 튜닝 과정을 소개하고 최종적으로 어떤 모형의 정확도가 높은지 ROC Curve를 통해 확인할 수 있습니다. 

# 데이터 형태

![image](https://user-images.githubusercontent.com/29458670/105003547-20ed5e80-5a76-11eb-9206-b1a31d204bd7.png)

데이터 형식은 csv로 파일에는 각 이메일에 대한 데이터가 5172개의 행과 3002 개의 열로 정리되어 있습니다. 
첫 번째 열은 이메일 이름을 나타냅니다. 그리고 마지막 열에는 예측 데이터가 기록되어 있습니다. 
나머지 3000개의 열은 메일에서 볼 수 있는 일반적인 3000개의 단어들로 구성되어 있습니다. 

![image](https://user-images.githubusercontent.com/29458670/105004001-b557c100-5a76-11eb-9385-f32be489ec10.png)

X변수는 첫 번째 열과, 마지막 열을 제외한 3000개의 열을 할당했습니다. 그리고 Y변수는 예측 데이터가 기록된 prediction열을 할당했습니다.
분석에 사용하는 데이터는 모두 6대4 비율로 train data, test data로 나누었습니다. 

# 모델 적합

# 1. 머신러닝 Naive Bayes 모델

![image](https://user-images.githubusercontent.com/29458670/105004222-fd76e380-5a76-11eb-82cd-de91bac347f2.png)

나이브 베이즈 분류는 특성들 사이의 독립을 가정하는 베이즈 정리를 적용한 확률 분류기의 일종으로 텍스트 분류에 많이 쓰이고 있습니다. 
나이브 베이즈 모델은 가우시안, 베르누이, 멀티노미얼 분포가 있는데 스팸 데이터의 경우 3000개의 x변수가 있기 때문에 멀티노미얼 나이브베이즈를 선택했습니다.
모델을 적합한 후 94%의 정확도를 기록했습니다.

# 2. 머신러닝 SVM 모델

기계학습의 한 분야로 패턴인식, 자료 분석을 위한 지도 학습 모델입니다. 주로 분류와 회귀분석을 위해 사용되는데 이 경우 분류 목적을 위해 사용되었습니다.
SVC 모델은 비선형방식을 따르는 rbf 커널을 선택했습니다.
이 모형은 cost와 gamma의 최적의 값을 찾아야하는데 cost값과 gamma값이 클수록 과적합 위험이 있다는 특징이 있습니다. 

![image](https://user-images.githubusercontent.com/29458670/105004298-1d0e0c00-5a77-11eb-838c-1b13c7597084.png)

우선 임의로 cost값을 1로 지정한 후 모델을 적합했습니다. 그 결과 77%의 정확도를 기록했습니다. 

![image](https://user-images.githubusercontent.com/29458670/105004315-24351a00-5a77-11eb-981f-c4ae76ca569a.png)

이후 tune parameter 기능을 이용해서 최적 값을 찾았습니다. 그런데 최적의 감마 값이 지정한 감마 값들 중 가장 작은 값이어서, 최적의 값을 중앙값으로 재배치한 후 다시 진행했습니다. 
이 과정을 두 번 반복한 후 적절한 최적의 cost값과 gamma값을 얻을 수 있었습니다. 이후 모델 적합 결과 94%의 정확도를 기록했습니다. 

![image](https://user-images.githubusercontent.com/29458670/105004338-2e571880-5a77-11eb-9ce9-cb94ded33b82.png)

첫 번째 best parameter 

![image](https://user-images.githubusercontent.com/29458670/105004374-3911ad80-5a77-11eb-8698-5d4f33f14d5a.png)

![image](https://user-images.githubusercontent.com/29458670/105004389-3f078e80-5a77-11eb-866c-ceb4b1ad06ac.png)

최종 best parameter 

![image](https://user-images.githubusercontent.com/29458670/105004408-46c73300-5a77-11eb-8096-cb4d43175857.png)

최적의 parameter를 이용해 모델 적합 결과 94%의 정확도를 기록했습니다. 

# 3. 신경망 MLP 모형

 MLP모형은 신경망 모형 중 다층 신경망 모델입니다. 대량의 데이터에서 정보를 얻거나 복잡한 모델을 만들 때 유용하게 사용됩니다. 
 
 ![image](https://user-images.githubusercontent.com/29458670/105004469-5e9eb700-5a77-11eb-9cfe-81352f07d0e1.png)
 
 먼저 모델을 적합해보니 42%로 낮은 정확도를 기록했습니다.
그래서 데이터의 평균을 0, 표준편차를 1로 표준화를 진행한 뒤 다시 모델을 적합했습니다. 그 결과 97%로 정확도가 크게 상승했습니다. 
SVM모형과 비교했을 때 데이터 스케일이 크게 영향을 미치는 것 같습니다. 

![image](https://user-images.githubusercontent.com/29458670/105004497-69594c00-5a77-11eb-8fb4-751006c1a7b2.png)

이후 튜닝 작업도 진행했는데 정확도가 약 0.6% 상승했습니다. 

![image](https://user-images.githubusercontent.com/29458670/105004543-7aa25880-5a77-11eb-8147-18bb08da5344.png)

# 4. 의사결정나무 Random Forest 모델

random forest 모형은 의사결정나무 모델을 베이스 모델로 사용하는 앙상블 모델입니다.
여기서 앙상블이란 여러 Base 모델들의 예측을 다수결 법칙 또는 평균을 이용해 결과를 통합하여 예측 정확성을 향상시키는 방법을 말합니다.

앙상블 모델 예시)

 ![image](https://user-images.githubusercontent.com/29458670/105004617-9279dc80-5a77-11eb-9102-1dcd219c463a.png)
 
  예시 그림은 5개의 BASE 모델들의 결과를 종합해 잘 예측된 하나의 결과를 내는 과정을 표현했습니다. 
  
random forest 모델 적합은 의사결정 트리 수, 의사결정 트리 깊이 등이 설정 가능한데, default값으로 모델을 적합했습니다. 

 ![image](https://user-images.githubusercontent.com/29458670/105004675-a1f92580-5a77-11eb-8419-fdbab6c60439.png)

# ROC CURVE 시각화 및 결론.

![image](https://user-images.githubusercontent.com/29458670/105004710-af161480-5a77-11eb-858b-9c0a0f470b58.png)

![image](https://user-images.githubusercontent.com/29458670/105004729-b3423200-5a77-11eb-8fe8-7129f1dbab5b.png)

결론입니다. 4가지 모델 모두 뛰어난 예측률을 보여줬습니다. 그중 가장 예측률이 높은 모델은 신경망 MLP모델로 약 98%의 정확도를 기록했습니다. 

MLP 모형은 첫 모델 정확도가 42%로 매우 낮은 값을 기록했지만, 데이터 표준화와 튜닝 작업을 거쳐 정확도가 약 98%로 상승했습니다.
이 분석에서 알 수 있는 사실은 분류 모델만큼 데이터를 해당 모델에 맞게 변환하는 것이 중요하다는 것입니다. 


 





