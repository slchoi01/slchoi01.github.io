---
layout: post
title: "XGBoost"
date: 2021-08-27
excerpt: "XGBoost(eXtra Gradient Boost) 정리"
tags: [machine learning, data science]
comments: true
---

## 1. XGBoost 개요
* XGBoost는 트리 기반의 앙상블 학습에서 많이 사용하는 알고리즘 중 하나. 압도적인 수치 차이는 아니지만, 분류에 있어 일반적으로 다른 머신러닝보다 뛰어난 예측 성능을 나타냄
* XGBoost는 GBM에 기반하고 있지만, GBM의 단점인 느린 수행 시간 및 과적합 규제(Regularization) 부재 등의 문제를 해결해서 매우 각광 받고 있음. 특히 XGBoost는 병렬 CPU 환경에서 병렬 학습이 가능해 기존 GBM보다 빠르게 학습을 완료할 수 있음


* XGBoost의 장점

항목|설명
:---|:---
뛰어난 예측 성능|일반적으로 분류와 회귀 영역에서 뛰어난 예측 성능을 발휘함
GBM대비 빠른 수행 시간| 일반적인 GBM은 순차적으로 Weak learner가 가중치를 증감하는 방법으로 학습하기 때문에 전반적으로 속도가 느림.
                      | 하지만 XGBoost는 병렬 수행 및 다양한 기능으로 GBM에 비해 빠른 수행 성능을 보장. XGBoost는 일반적인 GBM에 비해
                      | 수행 시간이 빠르다는 것이지, 다른 머신러닝 알고리즘(ex. 랜덤 포레스트)에 비해 빠르다는 의미는 아님
과적합 규제| 표준 GBM의 경우 과적합 규제 기능이 없으나 XGBoost는 자체에 과적합 규제 기능으로 과적합에 좀 더 강한 내구성을 가질 수 있음
Tree pruning   | 일반적으로 GBM은 분할 시 부정 손실이 발생하면 분할을 더 이상 수행하지 않지만, 이러한 방식도 자짓 지나치게 많은 분할을 
(나무 가지치기)|발생할 수 있음. 다른 GBM과 미찬가지로 XGBoost도 max_depth 파라미터로 분할 깊이를 조정하기도 하지만, tree pruning으로
               | 더 이상 긍정 이득이 없는 분할을 가지치기 해서 분할 수를 더 줄이는 추가적인 장점을 가지고 있음
자체 내장된 교차 검증|XGBoost는 반복 수행 시마다 내부적으로 학습 데이터 세트와 평가 데이터 세트에 대한 교차 검증을 수행해 최적화된 반
                     |복 수행 횟수를 가질 수 있음.
                     | 지정된 반복 횟수가 아니라 교차 검증을 통해 평가 데이터 세트의 평가 값이 최적화 되면 반복을 중간에 멈출 수 있는 
                     | 조기 중간 기능이 있음
결손값 자체 처리|XGBoost는 결손값을 자체 처리할 수 있는 기능을 가지고 있음

* XGBoost의 핵심 라이브러리는 C/C++로 작성돼 있음. XGBoost 개발 그룹은 파이썬에서도 XGBoost를 구동할 수 있도록 파이썬 패키지를 제공, 이 파이썬 패키지의 역할은 대부분 C/C++ 핵심 라이브러리를 호출하는 것
* XGBoost의 파이썬 패키지명은 `xgboost`.`xgboost` 패키지 내에는 XGBoost 전용의 파있너 패키지와 사이킷런과 호환되는 래퍼용 XGBoost가 함께 존재
* XGBoost 고유의 프레임 워크를 파이썬 언어 기반에서 구현한 것으로 별도의 API 기반임. 사이킷런 프레임워크를 기반으로 한 것이 아니기에 사이킷런의  `fit()`, `predict()` 메서드와 같은 사이킷런 고유의 아키텍처가 적용될 수 없으며, 다양한 유틸리티(`cross_val_score`, GridSearchCV, Pipline 등)와 함께 상요될 수 없음
* XGBoost는 사이킷런과 연동할 수 있는 래퍼 클래스(Wrapper class)를 제공
    * XGBoost 패키지의 사이킷런 래퍼 클래스는 XGBClassifier와 XGPRegressor
    * 위 패키지를 사용하면 estimator가 학습을 위해 사용하는 `fit()`과 `predict()`와 같은 표준 사이킷런 개발 프로세스 및 다양한 유틸리티를 활용할 수 있음
* 사이킷런 래퍼 XGBoost 모듈은 사이킷런의 다른 Estimator와 사용법이 같은 데 반해 파이썬 네이티브 XGBoost는 고유의 API와 하이퍼 파라미터를 이용

**XGBoost 설치**
* 아나콘다 Command 창을 관리자 권한으로 연 뒤에 `conda install -c anaconda py-xgboost` 명령어 입력

**XGBoost 설치 확인**


```python
import xgboost

print(xgboost.__version__)
```

    0.90
    

## 2. 파이썬 래퍼 XGBoost 하이퍼 파라미터
* XGBoost는 GBM과 유사한 하이퍼 파라미터를 동일하게 가지고 있으며, 여기에 조기 중단, 과적합을 규제하기 위한 하이퍼 파라미터 등이 추가 됨
* 유형별 파이썬 래퍼 XGBoost 하이퍼 파라미터
    * 일반 파라미터: 일반적으로 실행 시 스레드의 개수나 silent 모드 증의 선택을 위한 파라미터로서 디폴트 파라미터 값을 바꾸는 경우는 겅의 없음
    * 부스터 파라미터: 트리 최적화, 부스팅, regularization등과 관련 파라미터 등을 지칭
    * 학습 태스크 파라미터: 학습 수행시의 객체 함수. 평가를 위한 지표 등을 설정하는 파라미터
    * 대부분의 하이퍼 파라미터는 Booster 파라미터에 속함

**주요 일반 파라미터**
* `booster`: gbtree(tree based model) 또는 gblinear(linear model) 선택. 디폴트는 gbtree
* `silent`: 디폴트는 0이며, 출력 메시지를 나타내고 싶지 않을 경우 1로 설정
* `nthread`: CPU의 실행 스레드 개수를 조정. 디폴트는 CPU의 전체 스레드를 다 사용하는 것. 멀티 코어/스레드 CPU 시스템에서 전체 CPU를 사용하지 않고 일부 CPU만 사용해 ML 애플리케이션을 구동하는 경우에 변경

**주요 부스터 파라미터**
* `eta [default=0.3, alias: learnign_rate]`: GBM의 학습률과 같은 파라미터. 0에서 1 사이의 값을 지정하며 부스팅 스텝을 반복적으로 수행할 때 업데이트되는 학습률 값, 파이썬 래퍼 기반의 xgboost를 이용할 경우 디폴트는 0.3. 사이킷런 래퍼 클래스를 이용할 경우 eat는 learning_rate 파라미터로 대체되며, 디폴트는 0.1. 보통은 0.01 ~ 0.2 사이의 값을 선호
* `num_boost_rounds`: GBM의 n_estimators와 같은 파라미터
* `min_child_weight [default = 1]`: 트리에서 추가적으로 가지를 나눌지를 결정하기 위해 필요한 데이터들의 weight 총합. min_child_weight이 클수록 분할을 자제. 과적합을 조절하기 위해 사용
* `gamma [default = 0, alias: min_split_loss]`: 트리의 리프 노드를 추가적으로 나눌지를 결정할 최소 손실 감소 값. 해당 값보다 큰 손실(loss)이 감소된 경우에 리프 노드를 분리. 값이 클수록 과적합 감소 효과가 있음
* `max_depth [default = 6]`: 트리 기반 알고리즘의 max_depth와 같음. 0을 지정하면 깊이에 제한이 없음. Max_depth가 높으면 특정 피처 조건에 특화되어 룰 조건이 만들어지므로 과적합 가능성이 높아지며 보통은 3~10 사이의 값을 적용
* `sub_sample [default = 1]`: GBM의 subsample과 동일. 트리가 켜져서 과적합되는 것을 제어하기 위해 데이터를 샘플링하는 비율을 지정. sub_sample = 0.5로 지정하면 전체 데이터의 절반을 트리를 생성하는 데 사용. 0에서 1 사이의 값이 가능하나 일반적으로 0.5 ~ 1 사이의 값을 사용
* `colsample_bytree [default = 1]`: GVM의 max_features와 유사. 트리 생성에 필요한 피처를 임의로 샘플링하는데 사용. 매우 많은 피처가 있는 경우 과적합을 조정하는데 적용
* `lambda [default = 1, alias: reg_lambda]`: L2 Regularization 적용 값. 피처 개수가 많을 경우 적용을 검토하여 값이 클수록 과적합 감소 효과가 있음
* `alpha [default = 0, alias: reg_alpha]`: L1 Regularization 적용 값. 피처 개수가 많을 경우 적용을 검토하며 값이 클수록 과적합 감소 효과가 있음
* `scale_pos_weight [default = 1]`: 특정 값으로 치우친 비대칭한 클래스로 구성된 데이터 세트의 균형을 유지하기 위한 파라미터

**학습 태스크 파라미터**
* `objective`: 최솟값을 가져야할 손실 함수를 정의. XGBoost는 많은 유형의 손실함수를 사용할 수 있음. 주로 사용되는 손실함수는 이진 분류인지 다중 분류인지에 따라 달라짐
* `binary:logistic`: 이진 분류일 대 적용
* `multi:softmax`: 다중 분류일 때 적용. 손실함수가 multi:softmax일 경우 레이블 클래스의 개수인 num_class 파라미터를 지정해야 함
* `eval_metrix`: 검증에 사용되는 함수를 정의. 기본값은 회귀의 경우 rmse, 분류일 경우 error, 

**과적합 문제가 심각한 경우**
* eat 값을 낮춤(0.01 ~ 0.1). eta 값을 낮출 경우 num_round(또는 n_estimators)는 반대로 높여줘야 함
* max_depth 값을 낮춤
* min_child_weight 값을 높임
* gamma 값을 높임
* subsample과 colsample_bytree를 조정하는 것도 트리가 너무 복잡하게 생성되는 것을 막아 과적합 문제에 도움이 될 수 있음

**XGBoost 특징**
* XGBoost는 자체적으로 교차 검증, 성능 평가, 피처 중요도 등의 시각화 기능을 가지고 있음. 또한 XGBoost는 기존 GBM에서 부족한 다른 여러 가지 성능 향상 기능이 있음. 그중 수행 속도를 향상시키기 위한 대표적인 기능으로 조기 중단(Early Stopping) 기능이 있음
    * 기본 GBM의 경우 n_estimators(또는 num_boost_rounds)에 지정된 횟수만큼 반복적으로 학습 오류를 감소시키며 학습을 진행하면서 중간에 반복을 멈출 수 없고 n_estimators에 지정된 횟수를 다 완료해야 함
    * XGBoost는 조기 중단 기능이 있어 n_estimators에 지정한 부스팅 반복 횟수에 도달하지 않더라도 예측 오류가 더 이상 개선되지 않으면 반복을 끝까지 수행하지 않고 중지해 수행 시간을 개선할 수 있음

## 3. 파이썬 Native XGBoost 적용 – 위스콘신 Breast Cancer 데이터 셋                                           
* XGBoost의 파이썬 패키지인 xgboost는 자체적으로 교차 검증, 성능 평가, 피처 중요도 등의 시각화 기능을 가지고 있음. 또한 조기 중단 기능이 있어 num_rounds로 지정한 부스팅 반복 횟수에 도달하지 않더라도 더 이상 얘측 오류가 개선되지 않으면 반복을 끝까지 수행하지 않고 중지해 수행 시간으 개선하는 기능도 가지고 있음
* 일반적으로 수행 선으 향상 XGBoost는 GBM과 는 다르게 병렬 처리와 조기 중단 등으로 빠른 수행시간 처리가 가능하지만, CPU 코어가 많지 않은 개인용 PC에서는 수행 시간 향상을 경험하기 어려울 수 있음


**데이터 설명**
* 위스콘신 유방암 데이터 세트는 종양의 크기, 모양 등의 다양한 속성값을 기반으로 악성 종양(malignant)인지 양성 종양(benign)인지를 분류한 데이터 세트
* 종양은 양성 종양과 악성 종양으로 구분할 수 있으며, 양성 종양이 비교적 성장 속도가 느리고 전이되지 않는 것에 반해, 악성 종양은 주위 조직에 침입하면서 빠르게 성장하고 신체 각 부위에 확산되거나 전이 되어 생명을 위협함
* 위스콘신 유방암 데이터 세트에 기반해 종양의 다양한 피처에 따라 악성종양인지 일반 양성 종양인지를 XGBoost을 이용해 예측


**XGBoost를 이용한 종양 종류 예측**
* xgboost 패키지는 피처의 중요도를 시각화해주는 모듈인 `plot_importance`를 함께 제공. 나중에 피처 중요도를 시각화해볼 것
* 위스콘신 유방암 데이터 세트는 사이킷런에도 내장돼 있으며 `load_breast_cancer()`를 호출하면 데이터를 불러올 수 있음


**데이터 불러오기**


```python
import xgboost as xgb
from xgboost import plot_importance
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
X_features= dataset.data
y_label = dataset.target

cancer_df = pd.DataFrame(data=X_features, columns=dataset.feature_names)
cancer_df['target']= y_label
cancer_df.head(3)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.8</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>...</td>
      <td>17.33</td>
      <td>184.6</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.9</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>...</td>
      <td>23.41</td>
      <td>158.8</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.0</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>...</td>
      <td>25.53</td>
      <td>152.5</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 31 columns</p>
</div>



* 종양의 크기와 모양에 관련된 많은 속성이 숫자형 값으로 돼 있음. 타깃 레이블 값의 종류는 악성인 'malignant'가 0 값으로, 양성인 'benign'이 1 값으로 돼 있음


* 레이블 값의 분포 확인


```python
print(dataset.target_names)
print(cancer_df['target'].value_counts())
```

    ['malignant' 'benign']
    1    357
    0    212
    Name: target, dtype: int64
    

**데이터 세트 분할**
* 전체 데이터 세트 중 80%를 학습용으로, 20%를 테스트용으로 분할


```python
# 전체 데이터 중 80%는 학습용 데이터, 20%는 테스트용 데이터 추출
X_train, X_test, y_train, y_test=train_test_split(X_features, y_label,
                                         test_size=0.2, random_state=156 )
print(X_train.shape , X_test.shape)

```

    (455, 30) (114, 30)
    

**DMatrix 변환**
* 파이썬 래퍼 XGBoost와 사이킷런의 눈에 띄는 차이는 학습용과 테스트용 데이터 세트를 위해 별도의 객체인 DMatrix를 생성한다는 점
    * DMatrix는 주로 넘파이 입력 파라미터를 받아서 만들어지는 XGBoost만의 전용 데이터 세트
    * DMatrix의 주요 입력 파라미터는 data와 label. data는 피처 데이터 세트이며, label은 분류의 경우에는 레이블 데이터 세트, 회귀의 경우 숫자형인 종속값 데이터 세트
* DMatrix는 넘파이 외에 libsvm txt 포맷 파일, xgboost 이진 버퍼 파일을 파라미터로 입력받아 변환할 수 있음
* 판다스의 DataFrame으로 데이터 인터페이스를 하기 위해서는 DataFrame.values를 이용해 넘파이로 일차 변환한 뒤 이를 이용해 DMatrix 변환을 적용


```python
dtrain = xgb.DMatrix(data=X_train , label=y_train)
dtest = xgb.DMatrix(data=X_test , label=y_test)
```

**하이퍼 파라미터 설정**
* XGBoost의 하이퍼 파라미터는 주로 딕셔너리 형태로 입력


```python
params = { 'max_depth':3, # 트리 최대 깊이는 3
           'eta': 0.1, # 학습률 0.1
           'objective':'binary:logistic', # 예제 데이터가 0 또는 1 이진 분류이므로 목적함수는 이진 로지스틱
           'eval_metric':'logloss' # 오류 함수의 성능 지표는 logloss
        }
num_rounds = 400 # 부스팅 반복 횟수는 400번
```

**XGBoost 모델 학습**
* 파이썬 래퍼 XGBoost는 하이퍼 파라미터를 xgboost 모듈의 `train()` 함수에 파라미터로 전달(사이킷런의 경우 Estimator의 생성자를 하이퍼 파라미터로 전달)
* 학습 시 XGBoost는 수행 속도를 개선하기 위해 조기 중단 기능을 제공. 조기 중단은 xgboost의 `train()` 함수에 `early_stopping_rounds` 파라미터를 입력하여 설정
    * `early_stopping_rounds` 파라미터를 설정해 조기 중단을 수행하기 위해서는 `eval_set`과 `eval_metric`이 함께 설정돼야 함 


```python
# train 데이터 셋은 ‘train’ , evaluation(test) 데이터 셋은 ‘eval’ 로 명기합니다. 
wlist = [(dtrain,'train'),(dtest,'eval') ]
# 하이퍼 파라미터와 early stopping 파라미터를 train( ) 함수의 파라미터로 전달
xgb_model = xgb.train(params = params , dtrain=dtrain , num_boost_round=num_rounds , \
                      early_stopping_rounds=100, evals=wlist )
```

    [0]	train-logloss:0.609688	eval-logloss:0.61352
    Multiple eval metrics have been passed: 'eval-logloss' will be used for early stopping.
    
    Will train until eval-logloss hasn't improved in 100 rounds.
    [1]	train-logloss:0.540803	eval-logloss:0.547842
    [2]	train-logloss:0.483753	eval-logloss:0.494247
    [3]	train-logloss:0.434457	eval-logloss:0.447986
    [4]	train-logloss:0.39055	eval-logloss:0.409109
    [5]	train-logloss:0.354145	eval-logloss:0.374977
    [6]	train-logloss:0.321222	eval-logloss:0.345714
    [7]	train-logloss:0.292593	eval-logloss:0.320529
    [8]	train-logloss:0.267467	eval-logloss:0.29721
    [9]	train-logloss:0.245153	eval-logloss:0.277991
    [10]	train-logloss:0.225694	eval-logloss:0.260302
    [11]	train-logloss:0.207937	eval-logloss:0.246037
    [12]	train-logloss:0.192184	eval-logloss:0.231556
    [13]	train-logloss:0.177916	eval-logloss:0.22005
    [14]	train-logloss:0.165222	eval-logloss:0.208572
    [15]	train-logloss:0.153622	eval-logloss:0.199993
    [16]	train-logloss:0.14333	eval-logloss:0.190118
    [17]	train-logloss:0.133985	eval-logloss:0.181818
    [18]	train-logloss:0.125599	eval-logloss:0.174729
    [19]	train-logloss:0.117286	eval-logloss:0.167657
    [20]	train-logloss:0.109688	eval-logloss:0.158202
    [21]	train-logloss:0.102975	eval-logloss:0.154725
    [22]	train-logloss:0.097067	eval-logloss:0.148947
    [23]	train-logloss:0.091428	eval-logloss:0.143308
    [24]	train-logloss:0.086335	eval-logloss:0.136344
    [25]	train-logloss:0.081311	eval-logloss:0.132778
    [26]	train-logloss:0.076857	eval-logloss:0.127912
    [27]	train-logloss:0.072836	eval-logloss:0.125263
    [28]	train-logloss:0.069248	eval-logloss:0.119978
    [29]	train-logloss:0.065549	eval-logloss:0.116412
    [30]	train-logloss:0.062414	eval-logloss:0.114502
    [31]	train-logloss:0.059591	eval-logloss:0.112572
    [32]	train-logloss:0.057096	eval-logloss:0.11154
    [33]	train-logloss:0.054407	eval-logloss:0.108681
    [34]	train-logloss:0.052036	eval-logloss:0.106681
    [35]	train-logloss:0.049751	eval-logloss:0.104207
    [36]	train-logloss:0.04775	eval-logloss:0.102962
    [37]	train-logloss:0.045853	eval-logloss:0.100576
    [38]	train-logloss:0.044015	eval-logloss:0.098683
    [39]	train-logloss:0.042263	eval-logloss:0.096444
    [40]	train-logloss:0.040649	eval-logloss:0.095869
    [41]	train-logloss:0.039126	eval-logloss:0.094242
    [42]	train-logloss:0.037377	eval-logloss:0.094715
    [43]	train-logloss:0.036106	eval-logloss:0.094272
    [44]	train-logloss:0.034941	eval-logloss:0.093894
    [45]	train-logloss:0.033654	eval-logloss:0.094184
    [46]	train-logloss:0.032528	eval-logloss:0.09402
    [47]	train-logloss:0.031485	eval-logloss:0.09236
    [48]	train-logloss:0.030389	eval-logloss:0.093012
    [49]	train-logloss:0.029467	eval-logloss:0.091273
    [50]	train-logloss:0.028545	eval-logloss:0.090051
    [51]	train-logloss:0.027525	eval-logloss:0.089605
    [52]	train-logloss:0.026555	eval-logloss:0.089577
    [53]	train-logloss:0.025682	eval-logloss:0.090703
    [54]	train-logloss:0.025004	eval-logloss:0.089579
    [55]	train-logloss:0.024297	eval-logloss:0.090357
    [56]	train-logloss:0.023574	eval-logloss:0.091587
    [57]	train-logloss:0.022965	eval-logloss:0.091527
    [58]	train-logloss:0.022488	eval-logloss:0.091986
    [59]	train-logloss:0.021854	eval-logloss:0.091951
    [60]	train-logloss:0.021316	eval-logloss:0.091939
    [61]	train-logloss:0.020794	eval-logloss:0.091461
    [62]	train-logloss:0.020218	eval-logloss:0.090311
    [63]	train-logloss:0.019701	eval-logloss:0.089407
    [64]	train-logloss:0.01918	eval-logloss:0.089719
    [65]	train-logloss:0.018724	eval-logloss:0.089743
    [66]	train-logloss:0.018325	eval-logloss:0.089622
    [67]	train-logloss:0.017867	eval-logloss:0.088734
    [68]	train-logloss:0.017598	eval-logloss:0.088621
    [69]	train-logloss:0.017243	eval-logloss:0.089739
    [70]	train-logloss:0.01688	eval-logloss:0.089981
    [71]	train-logloss:0.016641	eval-logloss:0.089782
    [72]	train-logloss:0.016287	eval-logloss:0.089584
    [73]	train-logloss:0.015983	eval-logloss:0.089533
    [74]	train-logloss:0.015658	eval-logloss:0.088748
    [75]	train-logloss:0.015393	eval-logloss:0.088597
    [76]	train-logloss:0.015151	eval-logloss:0.08812
    [77]	train-logloss:0.01488	eval-logloss:0.088396
    [78]	train-logloss:0.014637	eval-logloss:0.088736
    [79]	train-logloss:0.014491	eval-logloss:0.088153
    [80]	train-logloss:0.014185	eval-logloss:0.087577
    [81]	train-logloss:0.014005	eval-logloss:0.087412
    [82]	train-logloss:0.013772	eval-logloss:0.08849
    [83]	train-logloss:0.013568	eval-logloss:0.088575
    [84]	train-logloss:0.013414	eval-logloss:0.08807
    [85]	train-logloss:0.013253	eval-logloss:0.087641
    [86]	train-logloss:0.013109	eval-logloss:0.087416
    [87]	train-logloss:0.012926	eval-logloss:0.087611
    [88]	train-logloss:0.012714	eval-logloss:0.087065
    [89]	train-logloss:0.012544	eval-logloss:0.08727
    [90]	train-logloss:0.012353	eval-logloss:0.087161
    [91]	train-logloss:0.012226	eval-logloss:0.086962
    [92]	train-logloss:0.012065	eval-logloss:0.087166
    [93]	train-logloss:0.011927	eval-logloss:0.087067
    [94]	train-logloss:0.011821	eval-logloss:0.086592
    [95]	train-logloss:0.011649	eval-logloss:0.086116
    [96]	train-logloss:0.011482	eval-logloss:0.087139
    [97]	train-logloss:0.01136	eval-logloss:0.086768
    [98]	train-logloss:0.011239	eval-logloss:0.086694
    [99]	train-logloss:0.011132	eval-logloss:0.086547
    [100]	train-logloss:0.011002	eval-logloss:0.086498
    [101]	train-logloss:0.010852	eval-logloss:0.08641
    [102]	train-logloss:0.010755	eval-logloss:0.086288
    [103]	train-logloss:0.010636	eval-logloss:0.086258
    [104]	train-logloss:0.0105	eval-logloss:0.086835
    [105]	train-logloss:0.010395	eval-logloss:0.086767
    [106]	train-logloss:0.010305	eval-logloss:0.087321
    [107]	train-logloss:0.010197	eval-logloss:0.087304
    [108]	train-logloss:0.010072	eval-logloss:0.08728
    [109]	train-logloss:0.01	eval-logloss:0.087298
    [110]	train-logloss:0.009914	eval-logloss:0.087289
    [111]	train-logloss:0.009798	eval-logloss:0.088002
    [112]	train-logloss:0.00971	eval-logloss:0.087936
    [113]	train-logloss:0.009628	eval-logloss:0.087843
    [114]	train-logloss:0.009558	eval-logloss:0.088066
    [115]	train-logloss:0.009483	eval-logloss:0.087649
    [116]	train-logloss:0.009416	eval-logloss:0.087298
    [117]	train-logloss:0.009306	eval-logloss:0.087799
    [118]	train-logloss:0.009228	eval-logloss:0.087751
    [119]	train-logloss:0.009154	eval-logloss:0.08768
    [120]	train-logloss:0.009118	eval-logloss:0.087626
    [121]	train-logloss:0.009016	eval-logloss:0.08757
    [122]	train-logloss:0.008972	eval-logloss:0.087547
    [123]	train-logloss:0.008904	eval-logloss:0.087156
    [124]	train-logloss:0.008837	eval-logloss:0.08767
    [125]	train-logloss:0.008803	eval-logloss:0.087737
    [126]	train-logloss:0.008709	eval-logloss:0.088275
    [127]	train-logloss:0.008645	eval-logloss:0.088309
    [128]	train-logloss:0.008613	eval-logloss:0.088266
    [129]	train-logloss:0.008555	eval-logloss:0.087886
    [130]	train-logloss:0.008463	eval-logloss:0.088861
    [131]	train-logloss:0.008416	eval-logloss:0.088675
    [132]	train-logloss:0.008385	eval-logloss:0.088743
    [133]	train-logloss:0.0083	eval-logloss:0.089218
    [134]	train-logloss:0.00827	eval-logloss:0.089179
    [135]	train-logloss:0.008218	eval-logloss:0.088821
    [136]	train-logloss:0.008157	eval-logloss:0.088512
    [137]	train-logloss:0.008076	eval-logloss:0.08848
    [138]	train-logloss:0.008047	eval-logloss:0.088386
    [139]	train-logloss:0.007973	eval-logloss:0.089145
    [140]	train-logloss:0.007946	eval-logloss:0.08911
    [141]	train-logloss:0.007898	eval-logloss:0.088765
    [142]	train-logloss:0.007872	eval-logloss:0.088678
    [143]	train-logloss:0.007847	eval-logloss:0.088389
    [144]	train-logloss:0.007776	eval-logloss:0.089271
    [145]	train-logloss:0.007752	eval-logloss:0.089238
    [146]	train-logloss:0.007728	eval-logloss:0.089139
    [147]	train-logloss:0.007689	eval-logloss:0.088907
    [148]	train-logloss:0.007621	eval-logloss:0.089416
    [149]	train-logloss:0.007598	eval-logloss:0.089388
    [150]	train-logloss:0.007575	eval-logloss:0.089108
    [151]	train-logloss:0.007521	eval-logloss:0.088735
    [152]	train-logloss:0.007498	eval-logloss:0.088717
    [153]	train-logloss:0.007464	eval-logloss:0.088484
    [154]	train-logloss:0.00741	eval-logloss:0.088471
    [155]	train-logloss:0.007389	eval-logloss:0.088545
    [156]	train-logloss:0.007367	eval-logloss:0.088521
    [157]	train-logloss:0.007345	eval-logloss:0.088547
    [158]	train-logloss:0.007323	eval-logloss:0.088275
    [159]	train-logloss:0.007303	eval-logloss:0.0883
    [160]	train-logloss:0.007282	eval-logloss:0.08828
    [161]	train-logloss:0.007261	eval-logloss:0.088013
    [162]	train-logloss:0.007241	eval-logloss:0.087758
    [163]	train-logloss:0.007221	eval-logloss:0.087784
    [164]	train-logloss:0.0072	eval-logloss:0.087777
    [165]	train-logloss:0.00718	eval-logloss:0.087517
    [166]	train-logloss:0.007161	eval-logloss:0.087542
    [167]	train-logloss:0.007142	eval-logloss:0.087642
    [168]	train-logloss:0.007122	eval-logloss:0.08739
    [169]	train-logloss:0.007103	eval-logloss:0.087377
    [170]	train-logloss:0.007084	eval-logloss:0.087298
    [171]	train-logloss:0.007065	eval-logloss:0.087368
    [172]	train-logloss:0.007047	eval-logloss:0.087395
    [173]	train-logloss:0.007028	eval-logloss:0.087385
    [174]	train-logloss:0.007009	eval-logloss:0.087132
    [175]	train-logloss:0.006991	eval-logloss:0.087159
    [176]	train-logloss:0.006973	eval-logloss:0.086955
    [177]	train-logloss:0.006955	eval-logloss:0.087053
    [178]	train-logloss:0.006937	eval-logloss:0.08697
    [179]	train-logloss:0.00692	eval-logloss:0.086973
    [180]	train-logloss:0.006901	eval-logloss:0.087038
    [181]	train-logloss:0.006884	eval-logloss:0.086799
    [182]	train-logloss:0.006866	eval-logloss:0.086826
    [183]	train-logloss:0.006849	eval-logloss:0.086582
    [184]	train-logloss:0.006831	eval-logloss:0.086588
    [185]	train-logloss:0.006815	eval-logloss:0.086614
    [186]	train-logloss:0.006798	eval-logloss:0.086372
    [187]	train-logloss:0.006781	eval-logloss:0.086369
    [188]	train-logloss:0.006764	eval-logloss:0.086297
    [189]	train-logloss:0.006747	eval-logloss:0.086104
    [190]	train-logloss:0.00673	eval-logloss:0.086023
    [191]	train-logloss:0.006714	eval-logloss:0.08605
    [192]	train-logloss:0.006698	eval-logloss:0.086149
    [193]	train-logloss:0.006682	eval-logloss:0.085916
    [194]	train-logloss:0.006666	eval-logloss:0.085915
    [195]	train-logloss:0.00665	eval-logloss:0.085984
    [196]	train-logloss:0.006634	eval-logloss:0.086012
    [197]	train-logloss:0.006618	eval-logloss:0.085922
    [198]	train-logloss:0.006603	eval-logloss:0.085853
    [199]	train-logloss:0.006587	eval-logloss:0.085874
    [200]	train-logloss:0.006572	eval-logloss:0.085888
    [201]	train-logloss:0.006556	eval-logloss:0.08595
    [202]	train-logloss:0.006542	eval-logloss:0.08573
    [203]	train-logloss:0.006527	eval-logloss:0.08573
    [204]	train-logloss:0.006512	eval-logloss:0.085753
    [205]	train-logloss:0.006497	eval-logloss:0.085821
    [206]	train-logloss:0.006483	eval-logloss:0.08584
    [207]	train-logloss:0.006469	eval-logloss:0.085776
    [208]	train-logloss:0.006455	eval-logloss:0.085686
    [209]	train-logloss:0.00644	eval-logloss:0.08571
    [210]	train-logloss:0.006427	eval-logloss:0.085806
    [211]	train-logloss:0.006413	eval-logloss:0.085593
    [212]	train-logloss:0.006399	eval-logloss:0.085801
    [213]	train-logloss:0.006385	eval-logloss:0.085807
    [214]	train-logloss:0.006372	eval-logloss:0.085744
    [215]	train-logloss:0.006359	eval-logloss:0.085658
    [216]	train-logloss:0.006345	eval-logloss:0.085843
    [217]	train-logloss:0.006332	eval-logloss:0.085632
    [218]	train-logloss:0.006319	eval-logloss:0.085726
    [219]	train-logloss:0.006306	eval-logloss:0.085783
    [220]	train-logloss:0.006293	eval-logloss:0.085791
    [221]	train-logloss:0.00628	eval-logloss:0.085817
    [222]	train-logloss:0.006268	eval-logloss:0.085757
    [223]	train-logloss:0.006255	eval-logloss:0.085674
    [224]	train-logloss:0.006242	eval-logloss:0.08586
    [225]	train-logloss:0.00623	eval-logloss:0.085871
    [226]	train-logloss:0.006218	eval-logloss:0.085927
    [227]	train-logloss:0.006206	eval-logloss:0.085954
    [228]	train-logloss:0.006194	eval-logloss:0.085874
    [229]	train-logloss:0.006182	eval-logloss:0.086057
    [230]	train-logloss:0.00617	eval-logloss:0.086002
    [231]	train-logloss:0.006158	eval-logloss:0.085922
    [232]	train-logloss:0.006147	eval-logloss:0.086102
    [233]	train-logloss:0.006135	eval-logloss:0.086115
    [234]	train-logloss:0.006124	eval-logloss:0.086169
    [235]	train-logloss:0.006112	eval-logloss:0.086263
    [236]	train-logloss:0.006101	eval-logloss:0.086291
    [237]	train-logloss:0.00609	eval-logloss:0.086217
    [238]	train-logloss:0.006079	eval-logloss:0.086395
    [239]	train-logloss:0.006068	eval-logloss:0.086342
    [240]	train-logloss:0.006057	eval-logloss:0.08618
    [241]	train-logloss:0.006046	eval-logloss:0.086195
    [242]	train-logloss:0.006036	eval-logloss:0.086248
    [243]	train-logloss:0.006025	eval-logloss:0.086263
    [244]	train-logloss:0.006014	eval-logloss:0.086293
    [245]	train-logloss:0.006004	eval-logloss:0.086222
    [246]	train-logloss:0.005993	eval-logloss:0.086398
    [247]	train-logloss:0.005983	eval-logloss:0.086347
    [248]	train-logloss:0.005972	eval-logloss:0.086276
    [249]	train-logloss:0.005962	eval-logloss:0.086448
    [250]	train-logloss:0.005952	eval-logloss:0.086294
    [251]	train-logloss:0.005942	eval-logloss:0.086312
    [252]	train-logloss:0.005932	eval-logloss:0.086364
    [253]	train-logloss:0.005922	eval-logloss:0.086394
    [254]	train-logloss:0.005912	eval-logloss:0.08649
    [255]	train-logloss:0.005903	eval-logloss:0.086441
    [256]	train-logloss:0.005893	eval-logloss:0.08629
    [257]	train-logloss:0.005883	eval-logloss:0.086459
    [258]	train-logloss:0.005874	eval-logloss:0.086391
    [259]	train-logloss:0.005864	eval-logloss:0.086441
    [260]	train-logloss:0.005855	eval-logloss:0.086461
    [261]	train-logloss:0.005845	eval-logloss:0.086491
    [262]	train-logloss:0.005836	eval-logloss:0.086445
    [263]	train-logloss:0.005827	eval-logloss:0.086466
    [264]	train-logloss:0.005818	eval-logloss:0.086319
    [265]	train-logloss:0.005809	eval-logloss:0.086488
    [266]	train-logloss:0.0058	eval-logloss:0.086538
    [267]	train-logloss:0.005791	eval-logloss:0.086471
    [268]	train-logloss:0.005782	eval-logloss:0.086501
    [269]	train-logloss:0.005773	eval-logloss:0.086522
    [270]	train-logloss:0.005764	eval-logloss:0.086689
    [271]	train-logloss:0.005755	eval-logloss:0.086738
    [272]	train-logloss:0.005747	eval-logloss:0.086829
    [273]	train-logloss:0.005738	eval-logloss:0.086684
    [274]	train-logloss:0.005729	eval-logloss:0.08664
    [275]	train-logloss:0.005721	eval-logloss:0.086496
    [276]	train-logloss:0.005712	eval-logloss:0.086355
    [277]	train-logloss:0.005704	eval-logloss:0.086519
    [278]	train-logloss:0.005696	eval-logloss:0.086567
    [279]	train-logloss:0.005687	eval-logloss:0.08659
    [280]	train-logloss:0.005679	eval-logloss:0.086679
    [281]	train-logloss:0.005671	eval-logloss:0.086637
    [282]	train-logloss:0.005663	eval-logloss:0.086499
    [283]	train-logloss:0.005655	eval-logloss:0.086356
    [284]	train-logloss:0.005646	eval-logloss:0.086405
    [285]	train-logloss:0.005639	eval-logloss:0.086429
    [286]	train-logloss:0.005631	eval-logloss:0.086456
    [287]	train-logloss:0.005623	eval-logloss:0.086504
    [288]	train-logloss:0.005615	eval-logloss:0.08637
    [289]	train-logloss:0.005608	eval-logloss:0.086457
    [290]	train-logloss:0.0056	eval-logloss:0.086453
    [291]	train-logloss:0.005593	eval-logloss:0.086322
    [292]	train-logloss:0.005585	eval-logloss:0.086284
    [293]	train-logloss:0.005577	eval-logloss:0.086148
    [294]	train-logloss:0.00557	eval-logloss:0.086196
    [295]	train-logloss:0.005563	eval-logloss:0.086221
    [296]	train-logloss:0.005556	eval-logloss:0.086308
    [297]	train-logloss:0.005548	eval-logloss:0.086178
    [298]	train-logloss:0.005541	eval-logloss:0.086263
    [299]	train-logloss:0.005534	eval-logloss:0.086131
    [300]	train-logloss:0.005526	eval-logloss:0.086179
    [301]	train-logloss:0.005519	eval-logloss:0.086052
    [302]	train-logloss:0.005512	eval-logloss:0.086016
    [303]	train-logloss:0.005505	eval-logloss:0.086101
    [304]	train-logloss:0.005498	eval-logloss:0.085977
    [305]	train-logloss:0.005491	eval-logloss:0.086059
    [306]	train-logloss:0.005484	eval-logloss:0.085971
    [307]	train-logloss:0.005478	eval-logloss:0.085998
    [308]	train-logloss:0.005471	eval-logloss:0.085998
    [309]	train-logloss:0.005464	eval-logloss:0.085877
    [310]	train-logloss:0.005457	eval-logloss:0.085923
    [311]	train-logloss:0.00545	eval-logloss:0.085948
    Stopping. Best iteration:
    [211]	train-logloss:0.006413	eval-logloss:0.085593
    
    


```python
pred_probs = xgb_model.predict(dtest)
print('predict( ) 수행 결과값을 10개만 표시, 예측 확률 값으로 표시됨')
print(np.round(pred_probs[:10],3))

# 예측 확률이 0.5 보다 크면 1 , 그렇지 않으면 0 으로 예측값 결정하여 List 객체인 preds에 저장 
preds = [ 1 if x > 0.5 else 0 for x in pred_probs ]
print('예측값 10개만 표시:',preds[:10])
```

    predict( ) 수행 결과값을 10개만 표시, 예측 확률 값으로 표시됨
    [0.934 0.003 0.91  0.094 0.993 1.    1.    0.999 0.997 0.   ]
    예측값 10개만 표시: [1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
    


```python
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score

def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix( y_test, pred)
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test , pred)
    f1 = f1_score(y_test,pred)
    # ROC-AUC 추가 
    roc_auc = roc_auc_score(y_test, pred_proba)
    print('오차 행렬')
    print(confusion)
    # ROC-AUC print 추가
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},\
    F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))
```


```python
get_clf_eval(y_test , preds, pred_probs)
```

    오차 행렬
    [[35  2]
     [ 1 76]]
    정확도: 0.9737, 정밀도: 0.9744, 재현율: 0.9870,    F1: 0.9806, AUC:0.9951
    


```python
import matplotlib.pyplot as plt
%matplotlib inline

fig, ax = plt.subplots(figsize=(10, 12))
plot_importance(xgb_model, ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22a6f3aeba8>




![png](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/pymldg/ch4/output_23_1.png)


### 사이킷런 Wrapper XGBoost 개요 및 적용 


```python
# 사이킷런 래퍼 XGBoost 클래스인 XGBClassifier 임포트
from xgboost import XGBClassifier

xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
xgb_wrapper.fit(X_train, y_train)
w_preds = xgb_wrapper.predict(X_test)
w_pred_proba = xgb_wrapper.predict_proba(X_test)[:, 1]
```


```python
get_clf_eval(y_test , w_preds, w_pred_proba)
```

    오차 행렬
    [[35  2]
     [ 1 76]]
    정확도: 0.9737, 정밀도: 0.9744, 재현율: 0.9870,    F1: 0.9806, AUC:0.9951
    


```python
from xgboost import XGBClassifier

xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
evals = [(X_test, y_test)]
xgb_wrapper.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="logloss", 
                eval_set=evals, verbose=True)

ws100_preds = xgb_wrapper.predict(X_test)
ws100_pred_proba = xgb_wrapper.predict_proba(X_test)[:, 1]
```

    [0]	validation_0-logloss:0.61352
    Will train until validation_0-logloss hasn't improved in 100 rounds.
    [1]	validation_0-logloss:0.547842
    [2]	validation_0-logloss:0.494247
    [3]	validation_0-logloss:0.447986
    [4]	validation_0-logloss:0.409109
    [5]	validation_0-logloss:0.374977
    [6]	validation_0-logloss:0.345714
    [7]	validation_0-logloss:0.320529
    [8]	validation_0-logloss:0.29721
    [9]	validation_0-logloss:0.277991
    [10]	validation_0-logloss:0.260302
    [11]	validation_0-logloss:0.246037
    [12]	validation_0-logloss:0.231556
    [13]	validation_0-logloss:0.22005
    [14]	validation_0-logloss:0.208572
    [15]	validation_0-logloss:0.199993
    [16]	validation_0-logloss:0.190118
    [17]	validation_0-logloss:0.181818
    [18]	validation_0-logloss:0.174729
    [19]	validation_0-logloss:0.167657
    [20]	validation_0-logloss:0.158202
    [21]	validation_0-logloss:0.154725
    [22]	validation_0-logloss:0.148947
    [23]	validation_0-logloss:0.143308
    [24]	validation_0-logloss:0.136344
    [25]	validation_0-logloss:0.132778
    [26]	validation_0-logloss:0.127912
    [27]	validation_0-logloss:0.125263
    [28]	validation_0-logloss:0.119978
    [29]	validation_0-logloss:0.116412
    [30]	validation_0-logloss:0.114502
    [31]	validation_0-logloss:0.112572
    [32]	validation_0-logloss:0.11154
    [33]	validation_0-logloss:0.108681
    [34]	validation_0-logloss:0.106681
    [35]	validation_0-logloss:0.104207
    [36]	validation_0-logloss:0.102962
    [37]	validation_0-logloss:0.100576
    [38]	validation_0-logloss:0.098683
    [39]	validation_0-logloss:0.096444
    [40]	validation_0-logloss:0.095869
    [41]	validation_0-logloss:0.094242
    [42]	validation_0-logloss:0.094715
    [43]	validation_0-logloss:0.094272
    [44]	validation_0-logloss:0.093894
    [45]	validation_0-logloss:0.094184
    [46]	validation_0-logloss:0.09402
    [47]	validation_0-logloss:0.09236
    [48]	validation_0-logloss:0.093012
    [49]	validation_0-logloss:0.091272
    [50]	validation_0-logloss:0.090051
    [51]	validation_0-logloss:0.089605
    [52]	validation_0-logloss:0.089577
    [53]	validation_0-logloss:0.090703
    [54]	validation_0-logloss:0.089579
    [55]	validation_0-logloss:0.090357
    [56]	validation_0-logloss:0.091587
    [57]	validation_0-logloss:0.091527
    [58]	validation_0-logloss:0.091986
    [59]	validation_0-logloss:0.091951
    [60]	validation_0-logloss:0.091939
    [61]	validation_0-logloss:0.091461
    [62]	validation_0-logloss:0.090311
    [63]	validation_0-logloss:0.089407
    [64]	validation_0-logloss:0.089719
    [65]	validation_0-logloss:0.089743
    [66]	validation_0-logloss:0.089622
    [67]	validation_0-logloss:0.088734
    [68]	validation_0-logloss:0.088621
    [69]	validation_0-logloss:0.089739
    [70]	validation_0-logloss:0.089981
    [71]	validation_0-logloss:0.089782
    [72]	validation_0-logloss:0.089584
    [73]	validation_0-logloss:0.089533
    [74]	validation_0-logloss:0.088748
    [75]	validation_0-logloss:0.088597
    [76]	validation_0-logloss:0.08812
    [77]	validation_0-logloss:0.088396
    [78]	validation_0-logloss:0.088736
    [79]	validation_0-logloss:0.088153
    [80]	validation_0-logloss:0.087577
    [81]	validation_0-logloss:0.087412
    [82]	validation_0-logloss:0.08849
    [83]	validation_0-logloss:0.088575
    [84]	validation_0-logloss:0.08807
    [85]	validation_0-logloss:0.087641
    [86]	validation_0-logloss:0.087416
    [87]	validation_0-logloss:0.087611
    [88]	validation_0-logloss:0.087065
    [89]	validation_0-logloss:0.08727
    [90]	validation_0-logloss:0.087161
    [91]	validation_0-logloss:0.086962
    [92]	validation_0-logloss:0.087166
    [93]	validation_0-logloss:0.087067
    [94]	validation_0-logloss:0.086592
    [95]	validation_0-logloss:0.086116
    [96]	validation_0-logloss:0.087139
    [97]	validation_0-logloss:0.086768
    [98]	validation_0-logloss:0.086694
    [99]	validation_0-logloss:0.086547
    [100]	validation_0-logloss:0.086498
    [101]	validation_0-logloss:0.08641
    [102]	validation_0-logloss:0.086288
    [103]	validation_0-logloss:0.086258
    [104]	validation_0-logloss:0.086835
    [105]	validation_0-logloss:0.086767
    [106]	validation_0-logloss:0.087321
    [107]	validation_0-logloss:0.087304
    [108]	validation_0-logloss:0.08728
    [109]	validation_0-logloss:0.087298
    [110]	validation_0-logloss:0.087289
    [111]	validation_0-logloss:0.088002
    [112]	validation_0-logloss:0.087936
    [113]	validation_0-logloss:0.087843
    [114]	validation_0-logloss:0.088066
    [115]	validation_0-logloss:0.087649
    [116]	validation_0-logloss:0.087298
    [117]	validation_0-logloss:0.087799
    [118]	validation_0-logloss:0.087751
    [119]	validation_0-logloss:0.08768
    [120]	validation_0-logloss:0.087626
    [121]	validation_0-logloss:0.08757
    [122]	validation_0-logloss:0.087547
    [123]	validation_0-logloss:0.087156
    [124]	validation_0-logloss:0.08767
    [125]	validation_0-logloss:0.087737
    [126]	validation_0-logloss:0.088275
    [127]	validation_0-logloss:0.088309
    [128]	validation_0-logloss:0.088266
    [129]	validation_0-logloss:0.087886
    [130]	validation_0-logloss:0.088861
    [131]	validation_0-logloss:0.088675
    [132]	validation_0-logloss:0.088743
    [133]	validation_0-logloss:0.089218
    [134]	validation_0-logloss:0.089179
    [135]	validation_0-logloss:0.088821
    [136]	validation_0-logloss:0.088512
    [137]	validation_0-logloss:0.08848
    [138]	validation_0-logloss:0.088386
    [139]	validation_0-logloss:0.089145
    [140]	validation_0-logloss:0.08911
    [141]	validation_0-logloss:0.088765
    [142]	validation_0-logloss:0.088678
    [143]	validation_0-logloss:0.088389
    [144]	validation_0-logloss:0.089271
    [145]	validation_0-logloss:0.089238
    [146]	validation_0-logloss:0.089139
    [147]	validation_0-logloss:0.088907
    [148]	validation_0-logloss:0.089416
    [149]	validation_0-logloss:0.089388
    [150]	validation_0-logloss:0.089108
    [151]	validation_0-logloss:0.088735
    [152]	validation_0-logloss:0.088717
    [153]	validation_0-logloss:0.088484
    [154]	validation_0-logloss:0.088471
    [155]	validation_0-logloss:0.088545
    [156]	validation_0-logloss:0.088521
    [157]	validation_0-logloss:0.088547
    [158]	validation_0-logloss:0.088275
    [159]	validation_0-logloss:0.0883
    [160]	validation_0-logloss:0.08828
    [161]	validation_0-logloss:0.088013
    [162]	validation_0-logloss:0.087758
    [163]	validation_0-logloss:0.087784
    [164]	validation_0-logloss:0.087777
    [165]	validation_0-logloss:0.087517
    [166]	validation_0-logloss:0.087542
    [167]	validation_0-logloss:0.087642
    [168]	validation_0-logloss:0.08739
    [169]	validation_0-logloss:0.087377
    [170]	validation_0-logloss:0.087298
    [171]	validation_0-logloss:0.087368
    [172]	validation_0-logloss:0.087395
    [173]	validation_0-logloss:0.087385
    [174]	validation_0-logloss:0.087132
    [175]	validation_0-logloss:0.087159
    [176]	validation_0-logloss:0.086955
    [177]	validation_0-logloss:0.087053
    [178]	validation_0-logloss:0.08697
    [179]	validation_0-logloss:0.086973
    [180]	validation_0-logloss:0.087038
    [181]	validation_0-logloss:0.086799
    [182]	validation_0-logloss:0.086826
    [183]	validation_0-logloss:0.086582
    [184]	validation_0-logloss:0.086588
    [185]	validation_0-logloss:0.086614
    [186]	validation_0-logloss:0.086372
    [187]	validation_0-logloss:0.086369
    [188]	validation_0-logloss:0.086297
    [189]	validation_0-logloss:0.086104
    [190]	validation_0-logloss:0.086023
    [191]	validation_0-logloss:0.08605
    [192]	validation_0-logloss:0.086149
    [193]	validation_0-logloss:0.085916
    [194]	validation_0-logloss:0.085915
    [195]	validation_0-logloss:0.085984
    [196]	validation_0-logloss:0.086012
    [197]	validation_0-logloss:0.085922
    [198]	validation_0-logloss:0.085853
    [199]	validation_0-logloss:0.085874
    [200]	validation_0-logloss:0.085888
    [201]	validation_0-logloss:0.08595
    [202]	validation_0-logloss:0.08573
    [203]	validation_0-logloss:0.08573
    [204]	validation_0-logloss:0.085753
    [205]	validation_0-logloss:0.085821
    [206]	validation_0-logloss:0.08584
    [207]	validation_0-logloss:0.085776
    [208]	validation_0-logloss:0.085686
    [209]	validation_0-logloss:0.08571
    [210]	validation_0-logloss:0.085806
    [211]	validation_0-logloss:0.085593
    [212]	validation_0-logloss:0.085801
    [213]	validation_0-logloss:0.085806
    [214]	validation_0-logloss:0.085744
    [215]	validation_0-logloss:0.085658
    [216]	validation_0-logloss:0.085843
    [217]	validation_0-logloss:0.085632
    [218]	validation_0-logloss:0.085726
    [219]	validation_0-logloss:0.085783
    [220]	validation_0-logloss:0.085791
    [221]	validation_0-logloss:0.085817
    [222]	validation_0-logloss:0.085757
    [223]	validation_0-logloss:0.085674
    [224]	validation_0-logloss:0.08586
    [225]	validation_0-logloss:0.085871
    [226]	validation_0-logloss:0.085927
    [227]	validation_0-logloss:0.085954
    [228]	validation_0-logloss:0.085874
    [229]	validation_0-logloss:0.086057
    [230]	validation_0-logloss:0.086002
    [231]	validation_0-logloss:0.085922
    [232]	validation_0-logloss:0.086102
    [233]	validation_0-logloss:0.086115
    [234]	validation_0-logloss:0.086169
    [235]	validation_0-logloss:0.086263
    [236]	validation_0-logloss:0.086292
    [237]	validation_0-logloss:0.086217
    [238]	validation_0-logloss:0.086395
    [239]	validation_0-logloss:0.086342
    [240]	validation_0-logloss:0.08618
    [241]	validation_0-logloss:0.086195
    [242]	validation_0-logloss:0.086248
    [243]	validation_0-logloss:0.086263
    [244]	validation_0-logloss:0.086293
    [245]	validation_0-logloss:0.086222
    [246]	validation_0-logloss:0.086398
    [247]	validation_0-logloss:0.086347
    [248]	validation_0-logloss:0.086276
    [249]	validation_0-logloss:0.086448
    [250]	validation_0-logloss:0.086294
    [251]	validation_0-logloss:0.086312
    [252]	validation_0-logloss:0.086364
    [253]	validation_0-logloss:0.086394
    [254]	validation_0-logloss:0.08649
    [255]	validation_0-logloss:0.086441
    [256]	validation_0-logloss:0.08629
    [257]	validation_0-logloss:0.08646
    [258]	validation_0-logloss:0.086391
    [259]	validation_0-logloss:0.086441
    [260]	validation_0-logloss:0.086461
    [261]	validation_0-logloss:0.086491
    [262]	validation_0-logloss:0.086445
    [263]	validation_0-logloss:0.086466
    [264]	validation_0-logloss:0.086319
    [265]	validation_0-logloss:0.086488
    [266]	validation_0-logloss:0.086538
    [267]	validation_0-logloss:0.086471
    [268]	validation_0-logloss:0.086501
    [269]	validation_0-logloss:0.086522
    [270]	validation_0-logloss:0.086689
    [271]	validation_0-logloss:0.086738
    [272]	validation_0-logloss:0.08683
    [273]	validation_0-logloss:0.086684
    [274]	validation_0-logloss:0.08664
    [275]	validation_0-logloss:0.086496
    [276]	validation_0-logloss:0.086355
    [277]	validation_0-logloss:0.086519
    [278]	validation_0-logloss:0.086567
    [279]	validation_0-logloss:0.08659
    [280]	validation_0-logloss:0.086679
    [281]	validation_0-logloss:0.086637
    [282]	validation_0-logloss:0.086499
    [283]	validation_0-logloss:0.086356
    [284]	validation_0-logloss:0.086405
    [285]	validation_0-logloss:0.086429
    [286]	validation_0-logloss:0.086456
    [287]	validation_0-logloss:0.086504
    [288]	validation_0-logloss:0.08637
    [289]	validation_0-logloss:0.086457
    [290]	validation_0-logloss:0.086453
    [291]	validation_0-logloss:0.086322
    [292]	validation_0-logloss:0.086284
    [293]	validation_0-logloss:0.086148
    [294]	validation_0-logloss:0.086196
    [295]	validation_0-logloss:0.086221
    [296]	validation_0-logloss:0.086308
    [297]	validation_0-logloss:0.086178
    [298]	validation_0-logloss:0.086263
    [299]	validation_0-logloss:0.086131
    [300]	validation_0-logloss:0.086179
    [301]	validation_0-logloss:0.086052
    [302]	validation_0-logloss:0.086016
    [303]	validation_0-logloss:0.086101
    [304]	validation_0-logloss:0.085977
    [305]	validation_0-logloss:0.086059
    [306]	validation_0-logloss:0.085971
    [307]	validation_0-logloss:0.085998
    [308]	validation_0-logloss:0.085999
    [309]	validation_0-logloss:0.085877
    [310]	validation_0-logloss:0.085923
    [311]	validation_0-logloss:0.085948
    Stopping. Best iteration:
    [211]	validation_0-logloss:0.085593
    
    


```python
get_clf_eval(y_test , ws100_preds, ws100_pred_proba)
```

    오차 행렬
    [[34  3]
     [ 1 76]]
    정확도: 0.9649, 정밀도: 0.9620, 재현율: 0.9870,    F1: 0.9744, AUC:0.9954
    


```python
# early_stopping_rounds를 10으로 설정하고 재 학습. 
xgb_wrapper.fit(X_train, y_train, early_stopping_rounds=10, 
                eval_metric="logloss", eval_set=evals,verbose=True)

ws10_preds = xgb_wrapper.predict(X_test)
ws10_pred_proba = xgb_wrapper.predict_proba(X_test)[:, 1]
get_clf_eval(y_test , ws10_preds, ws10_pred_proba)
```

    [0]	validation_0-logloss:0.61352
    Will train until validation_0-logloss hasn't improved in 10 rounds.
    [1]	validation_0-logloss:0.547842
    [2]	validation_0-logloss:0.494247
    [3]	validation_0-logloss:0.447986
    [4]	validation_0-logloss:0.409109
    [5]	validation_0-logloss:0.374977
    [6]	validation_0-logloss:0.345714
    [7]	validation_0-logloss:0.320529
    [8]	validation_0-logloss:0.29721
    [9]	validation_0-logloss:0.277991
    [10]	validation_0-logloss:0.260302
    [11]	validation_0-logloss:0.246037
    [12]	validation_0-logloss:0.231556
    [13]	validation_0-logloss:0.22005
    [14]	validation_0-logloss:0.208572
    [15]	validation_0-logloss:0.199993
    [16]	validation_0-logloss:0.190118
    [17]	validation_0-logloss:0.181818
    [18]	validation_0-logloss:0.174729
    [19]	validation_0-logloss:0.167657
    [20]	validation_0-logloss:0.158202
    [21]	validation_0-logloss:0.154725
    [22]	validation_0-logloss:0.148947
    [23]	validation_0-logloss:0.143308
    [24]	validation_0-logloss:0.136344
    [25]	validation_0-logloss:0.132778
    [26]	validation_0-logloss:0.127912
    [27]	validation_0-logloss:0.125263
    [28]	validation_0-logloss:0.119978
    [29]	validation_0-logloss:0.116412
    [30]	validation_0-logloss:0.114502
    [31]	validation_0-logloss:0.112572
    [32]	validation_0-logloss:0.11154
    [33]	validation_0-logloss:0.108681
    [34]	validation_0-logloss:0.106681
    [35]	validation_0-logloss:0.104207
    [36]	validation_0-logloss:0.102962
    [37]	validation_0-logloss:0.100576
    [38]	validation_0-logloss:0.098683
    [39]	validation_0-logloss:0.096444
    [40]	validation_0-logloss:0.095869
    [41]	validation_0-logloss:0.094242
    [42]	validation_0-logloss:0.094715
    [43]	validation_0-logloss:0.094272
    [44]	validation_0-logloss:0.093894
    [45]	validation_0-logloss:0.094184
    [46]	validation_0-logloss:0.09402
    [47]	validation_0-logloss:0.09236
    [48]	validation_0-logloss:0.093012
    [49]	validation_0-logloss:0.091272
    [50]	validation_0-logloss:0.090051
    [51]	validation_0-logloss:0.089605
    [52]	validation_0-logloss:0.089577
    [53]	validation_0-logloss:0.090703
    [54]	validation_0-logloss:0.089579
    [55]	validation_0-logloss:0.090357
    [56]	validation_0-logloss:0.091587
    [57]	validation_0-logloss:0.091527
    [58]	validation_0-logloss:0.091986
    [59]	validation_0-logloss:0.091951
    [60]	validation_0-logloss:0.091939
    [61]	validation_0-logloss:0.091461
    [62]	validation_0-logloss:0.090311
    Stopping. Best iteration:
    [52]	validation_0-logloss:0.089577
    
    오차 행렬
    [[34  3]
     [ 2 75]]
    정확도: 0.9561, 정밀도: 0.9615, 재현율: 0.9740,    F1: 0.9677, AUC:0.9947
    


```python
from xgboost import plot_importance
import matplotlib.pyplot as plt
%matplotlib inline

fig, ax = plt.subplots(figsize=(10, 12))
# 사이킷런 래퍼 클래스를 입력해도 무방. 
plot_importance(xgb_wrapper, ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1752ceffa90>




![png](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/pymldg/ch4/output_30_1.png)


