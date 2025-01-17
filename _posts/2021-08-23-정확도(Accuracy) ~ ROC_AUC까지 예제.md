---
layout: post
title: "정확도 ~ ROC_AUC"
date: 2021-08-23
excerpt: "정확도, 오차행렬, 정밀도와 재현율, F1 Score, ROC_AUC"
tags: [machine learning, data science]
comments: true
---

* 분류에 사용되는 성능 평가 지표에 대해 알아볼 것. 0과 1로 결정값이 한정되는 이진 분류의 성능 평가 지표에 대해 집중적으로 설명
* 분류의 성능 평가 지표
    1. 정확도(Accuracy)
    2. 오차행렬(Confusion Matrix)
    3. 정밀도(Precision)
    4. 재현율(Recall)
    5. F1 스크어
    6. ROC AUC
* 분류는 결정 클래스 값 졷류의 유형에 따라 2개의 결괏값만을 가지는 이진분류와 여러 개의 결정 클래스 값을 가지는 멀티 분류로 나뉨
* 위의 성능 지표는 이진/멀티 분류에 모두 적용되는 지표지만, 이진 분류에서 더욱 중요하게 강조하는 지표

## 1. Accuracy(정확도)

* 정확도: 실제 데이터에서 예측 데이터가 얼마나 같은지를 판단하는 지표
* 이진 분류의 경우 데이터의 구성에 따라 ML 모델의 성능을 왜곡할 수 있기 때문에 정확도 수치 하나만 가지고 성능을 평가하지 않음


```python
import sklearn

print(sklearn.__version__)
```

    0.21.2
    

* 사이킷런의 BaseEstimator 클래스를 상속받아 아무런 학습을 하지 않고, 성별에 따라 생존자를 예측하는 Classifier 생성
    * BaseEstimator를 상속받으면 Customized 형태의 Estimator를 생성할 수 있음
    * 생성할 MyDummyClassifier 클래스는 학습을 수행하는 `fit()` 메서드는 아무것도 수행하지 않으며 예측을 수행하는 `predict()` 메서드는 Sex 피처가 1이면 0, 그렇지 않으면 1로 예측하는 Classifier


```python
import numpy as np
from sklearn.base import BaseEstimator

class MyDummyClassifier(BaseEstimator):
    # fit( ) 메소드는 아무것도 학습하지 않음. 
    def fit(self, X , y=None):
        pass
    
    # predict( ) 메소드는 단순히 Sex feature가 1 이면 0 , 그렇지 않으면 1 로 예측함. 
    def predict(self, X):
        pred = np.zeros( ( X.shape[0], 1 ))
        for i in range (X.shape[0]) :
            if X['Sex'].iloc[i] == 1:
                pred[i] = 0
            else :
                pred[i] = 1
        
        return pred

```

* 생성된 MyDummyClassifier를 이용해 타이타닉 생존자 예측을 수행


```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Null 처리 함수
def fillna(df):
    df['Age'].fillna(df['Age'].mean(),inplace=True)
    df['Cabin'].fillna('N',inplace=True)
    df['Embarked'].fillna('N',inplace=True)
    df['Fare'].fillna(0,inplace=True)
    return df

# 머신러닝 알고리즘에 불필요한 속성 제거
def drop_features(df):
    df.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
    return df

# 레이블 인코딩 수행. 
def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin','Sex','Embarked']
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

# 앞에서 설정한 Data Preprocessing 함수 호출
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df
```


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 원본 데이터를 재로딩, 데이터 가공, 학습데이터/테스트 데이터 분할. 
titanic_df = pd.read_csv('./titanic_train.csv')
y_titanic_df = titanic_df['Survived']
X_titanic_df= titanic_df.drop('Survived', axis=1)
X_titanic_df = transform_features(X_titanic_df)
X_train, X_test, y_train, y_test=train_test_split(X_titanic_df, y_titanic_df, \
                                                  test_size=0.2, random_state=0)

# 위에서 생성한 Dummy Classifier를 이용하여 학습/예측/평가 수행. 
myclf = MyDummyClassifier()
myclf.fit(X_train ,y_train)

mypredictions = myclf.predict(X_test)
print('Dummy Classifier의 정확도는: {0:.4f}'.format(accuracy_score(y_test , mypredictions)))
```

    Dummy Classifier의 정확도는: 0.7877
    

* 단순한 알고리즘으로 예측을 하더라도 데이터의 구성에 따라 정확도 결과가 높게 나올 수 있기 때문에 정확도를 평가 지표로 사용할 때는 매우 신중해야함
* 정확도는 불균형한 레이블 값 분포에서 ML 모델의 성능을 판단할 경우, 적합한 평가 지표가 아님
    * EX) 100개의 데이터가 있고, 이 중에서 90개의 데이터 레이블이 0, 10개의 데이터 레이블이 1이라고 한다면 무조건 0으로 예측 결과를 반환하는 ML 모델의 경우라도 정확도가 90%가 됨

**불균형한 데이터 세트에서 정확도 지표 적용시 발생하는 문제점**
* MNIST 데이터 세트를 변환해 불균형한 데이터 세트로 만든 위 정확도 지표를 적용
    * MNIST 데이터 세트는 0부터 9까지의 숫자 이미지의 픽셀 정보를 가지고 있으며, 숫자 Digit를 예측하는 데 사용됨
    * load_digits() API를 통해 MNIST 데이터 세트를 제공
* MNIST 데이터 세트에서 레이블 값이 7인 것만 True, 나머지 값은 모두 False로 변환해 이진 분류 문제로 변형
    * 전체 데이터의 10%만 True, 나머지 90%는 False인 불균형한 데이터 세트로 변형
* 불균헝한 데이터 세트에 모든 데이터를 False, 즉 0으로 예측하는 classifier를 이용해 정확도를 측정하면 약 90%에 가까운 예측 정확도를 나타냄

1. 불균형한 데이터 세트와 Dummy Classifier 생성


```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

class MyFakeClassifier(BaseEstimator):
    def fit(self,X,y):
        pass
    
    # 입력값으로 들어오는 X 데이터 셋의 크기만큼 모두 0값으로 만들어서 반환
    def predict(self,X):
        return np.zeros( (len(X), 1) , dtype=bool)

# 사이킷런의 내장 데이터 셋인 load_digits( )를 이용하여 MNIST 데이터 로딩
digits = load_digits()

print(digits.data)
print("### digits.data.shape:", digits.data.shape)
print(digits.target)
print("### digits.target.shape:", digits.target.shape)
```

    [[ 0.  0.  5. ...  0.  0.  0.]
     [ 0.  0.  0. ... 10.  0.  0.]
     [ 0.  0.  0. ... 16.  9.  0.]
     ...
     [ 0.  0.  1. ...  6.  0.  0.]
     [ 0.  0.  2. ... 12.  0.  0.]
     [ 0.  0. 10. ... 12.  1.  0.]]
    ### digits.data.shape: (1797, 64)
    [0 1 2 ... 8 9 8]
    ### digits.target.shape: (1797,)
    


```python
digits.target == 7
```




    array([False, False, False, ..., False, False, False])




```python
# digits번호가 7번이면 True이고 이를 astype(int)로 1로 변환, 7번이 아니면 False이고 0으로 변환. 
y = (digits.target == 7).astype(int)
X_train, X_test, y_train, y_test = train_test_split( digits.data, y, random_state=11)
```

2. 불균형한 데이터로 생성한 `y_test`의 데이터 분포도를 확인하고 MyFakeClassifier를 이용해 예측과 평가를 수행


```python
# 불균형한 레이블 데이터 분포도 확인. 
print('레이블 테스트 세트 크기 :', y_test.shape)
print('테스트 세트 레이블 0 과 1의 분포도')
print(pd.Series(y_test).value_counts())

# Dummy Classifier로 학습/예측/정확도 평가
fakeclf = MyFakeClassifier()
fakeclf.fit(X_train , y_train)
fakepred = fakeclf.predict(X_test)
print('모든 예측을 0으로 하여도 정확도는:{:.3f}'.format(accuracy_score(y_test , fakepred)))
```

    레이블 테스트 세트 크기 : (450,)
    테스트 세트 레이블 0 과 1의 분포도
    0    405
    1     45
    dtype: int64
    모든 예측을 0으로 하여도 정확도는:0.900
    

* 단순히 predict()의 결과를 np.zeros()로 모두 0 값으로 반환함에도 불구하고 450개의 테스트 데이터 세트에 수행한 예측 정확도는 90%


**정리**
* 정확도 평가 지표는 불균형한 레이블 데이터 세트에서는 성능 수치로 사용돼서는 안됨
* 정확도가 가지는 분류 평가 지표로서 이러한 한계점을 극복하기 위해 여러 가지 분류 지표와 함께 적됻해야 함

## 2. 오차 행렬(Confusion Matrix)
* 이진 분류에서 성능 지표로 많이 활용됨
* 오차 행렬은 학습된 분류 모델이 예측을 수행하면서 얼마나 헷갈리고 있는지도 함께 보여주는 지표
* 이진 분류의 예측 오류가 얼마인지와 더불어 어떠한 유형의 예측 오류가 발생하고 있는지를 함께 나타내는 지표

* 오차 행렬은 4분면 행렬에서 실제 리이블 클래스 값과 예측 레이블 클래스 값이 어떠한 유형을 가지고 매핑되는지를 나타냄
* 4분면의 왼쪽, 오른쪽을 예측된 클래스 값 기준으로 Negatice와 Positive로 분류하고, 4분면의 위, 아래를 실제 클래스 값 기준으로 Negative와 Positive로 분류하면 예측 클래스와 실제 클래스의 값 유형에 따라 결정되는 TN, FP, FN, TP 형태로 오차 행렬의 4분면을 채울 수 있음
* TN, FP, FN, TP 값을 다양하게 결합해 분류 모델 예측 성능의 오류가 어떠한 모습으로 발생하는지 알 수 있음

![오차 행렬](https://img1.daumcdn.net/thumb/R800x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FEojHI%2FbtqBuwXbsUN%2F1Tt2KW49Zp8ZM7tpBWHUcK%2Fimg.png)

* TN, FP, FN, TP는 예측 클래스와 실제 클래스의 Positive 결정 값(값 1)과 Negative 결정 값(값 0)의 결합에 따라 결정됨
    * EX) TN: True Negative. True는 예측 클래스 값과 실제 클래스 값이 같다는 의미고 Negative는 예측값이 Negative 값이라는 의미. 예측을 Negative 값 0으로 예측했는데, 실제 값도 Negitive 값 0이라는 의미
* TN, FP, FN, TP 기호가 의미하는 것은 앞 문자 True/False는 예측값과 실제값이 '같은가/틀린가'를 의미. 뒤 문자 Negative/Positive는 예측 결과 값이 부정(0)/긍정(1)을 의미
    * TN: 예측값을 Negative 값 0으로 예측했고 실제 값 역시 Negative 값 0
    * FP: 예측값을 Positive 값 1로 예측했는데 실제 값은 Negative 0
    * FN: 예측값을 Negative 0으로 예측했는데 실제 값은 Positive 값 1
    * TP: 예측값을 Positive 1로 예측했는데 실제 값 역시 Positive 값 1

* 사이킷런은 오차 행렬을 구하기 위해 confusion_matrix() API를 제공
* 정확도 예제에서 사용한 MyFakeClassifier의 예측 성능 지표를 오차 행렬로 표현
    * MyFakeClassifier의 예측 결과인 `fakepred`와 실제 결과인 `y_test`를 `confusion_matrix()`의 인자로 입력해 오차 행렬을 confusion_matrix()를 이용해 배열 형태로 출력


```python
from sklearn.metrics import confusion_matrix

# 앞절의 예측 결과인 fakepred와 실제 결과인 y_test의 Confusion Matrix출력
confusion_matrix(y_test , fakepred)
```




    array([[405,   0],
           [ 45,   0]], dtype=int64)



* 출력된 오차 행렬은 이진 분류의 TN, FP, FN, FP는 상단 도표와 동일한 위치를 가지고 array에서 가져올 수 있음
* `target == 7`인지 아닌지에 따라 클래스 값을 True/False 이진 분류로 변경한 데이터 세트를 사용해 무조건 Negative로 예측하는 Classifier였고 테스트 데이터 세트의 클래스 값 분포는 0이 405건, 1이 45건임
* 따라서 TN은 전체 450건 데이터 중 무조건 Negative 0으로 예측해서 True가 된 결과 405건, FP는 Positive 1로 예측한 건수가 없으므로 0건, N은 Positive 1인 건수 45건을 Negative로 예측해서 False가 된 결과 45건, TP는 Positive 1로 예측한 건수가 없으므로 0건

**정리**
* TP, TN, FP, FN 값은 Classifier 성능의 여러 면모를 판단할 수 있는 기반 정보를 제공
* 이 값을 조합해 Classifier의 성능을 측정할 수 있는 주요 지표인 정확도, 정밀도, 재현율 값을 알 수 있음
* 정확도는 예측값과 실제 값이 얼마나 동일한가에 대한 비율만으로 결정됨. 즉, 오차 행렬에서 True에 해당하는 값인 TN과 TF에 좌우됨
* 정확도 = 예측 결과와 실제 값이 동일한 건수 / 전체 데이터 수 = (TN + TP)/(TN + FP + FN+ TP) 로 정의

## 3. 정밀도(Precision) 과 재현율(Recall)
* 불균형한 데이터 세트에서 많이 사용
* Positive 데이터 세트의 예측 성능에 좀 더 초점을 맞춘 평가 지표
* 정밀도 재현율 공식
    * 정밀도 = TP / (FP + TP)
    * 재현율 = TP / (FN + TP)


* **정밀도**: 예측을 Positive로 한 대상 중에 예측과 실제 값이 Positive로 일치한 데이터의 비율
    * FP + TP는 예측을 Positive로 한 모든 데이터 건수
    * TP는 예측과 실제 값이 Positive로 일치한 데이터 건수
    * Positive 예측 성능을 정밀하게 측정하기 위한 평가 지표로 양성 예측도라고도 불림


* **재현율**: 실제 값이 Positive인 대상 중에 예측과 실제 값이 Positive로 일치한 데이터의 비율
    * FN + TP는 실제 값이 Positive인 모든 데이터 건수
    * TP는 예측과 실제 값이 Positive로 일치한 데이터 건수
    * 민감도 호근 TPR라고도 불림
    
    
* 이진 분류 모델의 업무 특성에 따라 특정 평가 지표가 더 중요한 지표로 간주될 수 있음
    * 재현율이 중요 지표인 경우: 실제 Positive 양성 데이터를 Negative로 잘못 판단하게 되면 업무상 큰 영향이 발생하는 경우
        * EX) 암 판단 모델, 금융 사기 적발 모델
    * 정밀도가 중요 지표인 경우: 실제 Negative 음성인 데이터 예측을 Postive 양성으로 잘못 판단하게 되면 업무상 큰 영향이 발생하는 경우
        * EX) 스팸메일 여부를 판단하는 모델
        
        
* 재현율과 정밀도는 모두 TP를 높이는데 초점을 맞추지만, 재현율은 FN(실제 Positive, 예측 Negative)를 낮추는데, 정밀도는 FP를 낮추는데 초점을 맞춤
* 가장 좋은 성능 평가는 재현율과 정밀도 모두 높은 수치를 얻는 것

**MyFakeClassifier의 예측 결과로 정밀도와 재현율 측정**
* 사이킷런은 정밀도 계산을 위해 `precision_score()`를, 재현율 계산을 위해 `recall_score()`를 API로 제공


```python
from sklearn.metrics import accuracy_score, precision_score , recall_score

print("정밀도:", precision_score(y_test, fakepred))
print("재현율:", recall_score(y_test, fakepred))
```

    정밀도: 0.0
    재현율: 0.0
    

    C:\Users\slc\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    

**오차행렬, 정확도, 정밀도, 재현율을 한꺼번에 계산하는 함수 생성**
* 평가를 간편하게 적용하기 위해 confusion matrix, accuracy, precision, recall 등의 평가를 한번에 호출하는 `get_clf_eval()` 함수를 생성


```python
from sklearn.metrics import accuracy_score, precision_score , recall_score , confusion_matrix

def get_clf_eval(y_test , pred):
    confusion = confusion_matrix( y_test, pred)
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test , pred)
    print('오차 행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}'.format(accuracy , precision ,recall))
```

**생존자 예측 후 평가 수행**
* 로지스틱 회귀 기반으로 타이타닉 생존자를 예측하고 confusion matrix, accuracy, precision, recall 평가를 수행
    * 정밀도에 비해 재현율이 낮게 나옴


```python
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression

# 원본 데이터를 재로딩, 데이터 가공, 학습데이터/테스트 데이터 분할. 
titanic_df = pd.read_csv('./titanic_train.csv')
y_titanic_df = titanic_df['Survived']
X_titanic_df= titanic_df.drop('Survived', axis=1)
X_titanic_df = transform_features(X_titanic_df)

X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, \
                                                    test_size=0.20, random_state=11)

lr_clf = LogisticRegression()

lr_clf.fit(X_train , y_train)
pred = lr_clf.predict(X_test)
get_clf_eval(y_test , pred)
```

    오차 행렬
    [[108  10]
     [ 14  47]]
    정확도: 0.8659, 정밀도: 0.8246, 재현율: 0.7705
    

    C:\Users\slc\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    

### 3.1 정밀도/재현율 트레이드오프(Precision/Recall Trade-off)
* 정밀도와 재현율은 상호 보완적인 평가 지표이기 때문에 어느 한 쪽을 강제로 높이면 다른 하나의 수치는 떨어지기 쉬움


* 사이킷런의 분류 알고리즘은 예측 데이터가 특정 레이블에 속하는지를 계산하기 위해 먼저 개별 레이블별로 결정 확률을 구한 후 예측 확률이 큰 레이블값으로 예측
* 일반적으로 이진 분류에서는 이 임곗값을 0.5, 즉 50%로 정하고 이 기준 값보다 확률이 크면 Positive, 작으면 Negative로 결정


* 사이킷런은 개별 데이터별로 예측 확률을 반환하는 메서드인 predict_proba()를 제공
    * `predict_proba()`는 학습이 완료된 사이킷런 Classifier 객체에서 호출이 가능하면 테스트 피처 데이터 세트를 파라미터로 입력해주면 테스트 피처 레코드의 개별 클래스 예측 확률을 반환
    * `predict()` 메서드와 유사하지만 단지 반환 결과가 예측 결과 클래스값이 아닌 예측 확률 결과임

**predict_proba( ) 메소드 확인**
* 이진 분류에서 `predict-proba()`를 수행해 반환되는 ndarray는 첫 번째 칼럼이 클래스 값 0에 대한 예측 확률, 두 번째 칼럼이 클래스 값 1에 대한 예측 확률임
* 타이타닉 생존자 데이터를 학습한 LogisiticRegression 객체에서 `predict_proba()`메서드를 수행한 뒤 반환 값을 확인하고, `predict()` 메서드의 결과와 비교


```python
pred_proba = lr_clf.predict_proba(X_test)
pred  = lr_clf.predict(X_test)
print('pred_proba()결과 Shape : {0}'.format(pred_proba.shape))
print('pred_proba array에서 앞 3개만 샘플로 추출 \n:', pred_proba[:3])

# 예측 확률 array 와 예측 결과값 array 를 concatenate 하여 예측 확률과 결과값을 한눈에 확인
pred_proba_result = np.concatenate([pred_proba , pred.reshape(-1,1)],axis=1)
print('두개의 class 중에서 더 큰 확률을 클래스 값으로 예측 \n',pred_proba_result[:3])

```

    pred_proba()결과 Shape : (179, 2)
    pred_proba array에서 앞 3개만 샘플로 추출 
    : [[0.44935228 0.55064772]
     [0.86335513 0.13664487]
     [0.86429645 0.13570355]]
    두개의 class 중에서 더 큰 확률을 클래스 값으로 예측 
     [[0.44935228 0.55064772 1.        ]
     [0.86335513 0.13664487 0.        ]
     [0.86429645 0.13570355 0.        ]]
    

* 반환 결과인 ndarray는 0과 1에 대한 확률을 나타내므로 첫 번째 칼럼 값과 두 번째 칼럼 값을 더하면 1이 됨
* 맨 마지막 줄의 `predict()` 메서드의 결과 비교에서도 확인 가능하듯, 두 개의 칼럼 중에서 더 큰 확률 값으로 `predict()` 메서드가 최종 예측하고 있음


* `predict()` 메서드는 `predict_proba()` 메서드에 기반해 생성된 API
    * `predict()`는 `predict_proba()` 호출 결과로 반환된 배열에서 분류 결정 임계값보다 큰 값이 들어 있는 칼럼의 위치를 받아 최종적으로 예측 클래스를 결정하는 API
* 사이킷런은 분류 결정 임곗값을 조절해 정밀도와 재현율의 성능 수치를 상호 보완적으로 조정할 수 있음

**Binarizer 활용**
* `threshold` 변수를 특정 값으로 설정하고 Binarizer 클래스를 객체로 생성
* 생성된 Binarizer 객체의 `fit_transform()` 메서드를 이용해 넘파이 ndarray를 입력하면 입력된 ndarray의 값을 지정된 `threshold`보다 같거나 작으면 0값으로, 크면 1 값으로 변환해 반환 


```python
from sklearn.preprocessing import Binarizer

X = [[ 1, -1,  2],
     [ 2,  0,  0],
     [ 0,  1.1, 1.2]]

# threshold 기준값보다 같거나 작으면 0을, 크면 1을 반환
binarizer = Binarizer(threshold=1.1)                     
print(binarizer.fit_transform(X))
```

    [[0. 0. 1.]
     [1. 0. 0.]
     [0. 0. 1.]]
    

**분류 결정 임계값 0.5 기반에서 Binarizer를 이용하여 예측값 변환**
* Binarizer를 이용해 사이킷런 `predict()`의 의사 코드를 작성
* LigisticRegression 객체의 `predict_proba()` 메서드로 구한 각 클래스별 예측 확률값인 `pred_proba` 객체 변수에 분류 결정 임계값(`threshold`)을 0.5로 지정한 Binarizer 클래스를 적용해 최종 예측값을 구함. 그 후 최종 예측값에 대해 `get_clf_eval()` 함수를 적용해 평가 지표 출력


```python
from sklearn.preprocessing import Binarizer

#Binarizer의 threshold 설정값. 분류 결정 임곗값임.  
custom_threshold = 0.5

# predict_proba( ) 반환값의 두번째 컬럼 , 즉 Positive 클래스 컬럼 하나만 추출하여 Binarizer를 적용
pred_proba_1 = pred_proba[:,1].reshape(-1,1)

binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_1) 
custom_predict = binarizer.transform(pred_proba_1)

get_clf_eval(y_test, custom_predict)
```

    오차 행렬
    [[108  10]
     [ 14  47]]
    정확도: 0.8659, 정밀도: 0.8246, 재현율: 0.7705
    

* 계산된 평가 지표는 앞 예제의 타이타닉 데이터로 학습된 로지스특 회귀 Classifier 객체에서 호출된 `predict()`로 계싼된 지표 값과 정확히 동일
* `predict()`가 `predict_proba()`에 기반함을 알 수 있음

**분류 결정 임계값 0.4 기반에서 Binarizer를 이용하여 예측값 변환**


```python
# Binarizer의 threshold 설정값을 0.4로 설정. 즉 분류 결정 임곗값을 0.5에서 0.4로 낮춤  
custom_threshold = 0.4
pred_proba_1 = pred_proba[:,1].reshape(-1,1)
binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_1) 
custom_predict = binarizer.transform(pred_proba_1)

get_clf_eval(y_test , custom_predict)
```

    오차 행렬
    [[97 21]
     [11 50]]
    정확도: 0.8212, 정밀도: 0.7042, 재현율: 0.8197
    

* 임계값을 낮추니 재현율 값이 올라가고 정밀도가 떨어짐
* 분류 결정 임계값은 Positive 예측값을 결정하는 확률의 기준이 되는데 확률이 0.5가 아닌 0.4부터 Positive로 예측을 하기 때문에 임계값 값을 낮출수록 True 값이 많아지게 됨
* Positive 예측값이 많아지면 상대적으로 재현율 값이 높아짐. 양성 예측을 많이 하다 보니 실제 양성을 음성으로 예측하는 횟수가 상대적으로 줄어들기 때문
* 임계값이 낮아지면서 TP가 47에서 50으로 늘었고 FN이 14에서 11로 줄어듦. 그에 따라 재현율 이 0.770에서 0.820으로 좋아짐
* FP는 10에서 21로 늘면서 정밀도가 0.825에서 0.704로 많이 나빠짐. 정확도도 0.866에서 0.821로 나빠짐

**여러개의 분류 결정 임곗값을 변경하면서  Binarizer를 이용하여 예측값 변환**
* 임계값을 0.4 ~ 0.6까지 0.05식 증가시키며 평가 지표 조사


```python
# 테스트를 수행할 모든 임곗값을 리스트 객체로 저장. 
thresholds = [0.4, 0.45, 0.50, 0.55, 0.60]

def get_eval_by_threshold(y_test , pred_proba_c1, thresholds):
    # thresholds list객체내의 값을 차례로 iteration하면서 Evaluation 수행.
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1) 
        custom_predict = binarizer.transform(pred_proba_c1)
        print('임곗값:',custom_threshold)
        get_clf_eval(y_test , custom_predict)

get_eval_by_threshold(y_test ,pred_proba[:,1].reshape(-1,1), thresholds )
```

    임곗값: 0.4
    오차 행렬
    [[97 21]
     [11 50]]
    정확도: 0.8212, 정밀도: 0.7042, 재현율: 0.8197
    임곗값: 0.45
    오차 행렬
    [[105  13]
     [ 13  48]]
    정확도: 0.8547, 정밀도: 0.7869, 재현율: 0.7869
    임곗값: 0.5
    오차 행렬
    [[108  10]
     [ 14  47]]
    정확도: 0.8659, 정밀도: 0.8246, 재현율: 0.7705
    임곗값: 0.55
    오차 행렬
    [[111   7]
     [ 16  45]]
    정확도: 0.8715, 정밀도: 0.8654, 재현율: 0.7377
    임곗값: 0.6
    오차 행렬
    [[113   5]
     [ 17  44]]
    정확도: 0.8771, 정밀도: 0.8980, 재현율: 0.7213
    

**결과 정리**

평가 지표|0.4|0.45|0.5|0.55|0.6
:-------:|:-:|:--:|:-:|:--:|:-:
정확도|0.8380|0.8492|0.8492|0.8659|0.8771
정밀도|0.7286|0.7656|0.7742|0.8364|0.8824
재현율|0.8361|0.8033|0.7869|0.7541|0.7377

* 임계값이 0.45일 경우 디폴트 0.5인 경우와 비교해 정확도는 동일하고 정밀도는 약간 떨어졌으나 재현율이 오름
* 재현율을 향상시키면서 다른 수치를 어느 정도 감소시켜야한다면 임계깞 0.45가 적당해 보임

**precision_recall_curve( ) 를 이용하여 임곗값에 따른 정밀도-재현율 값 추출**
* `precision_recall_curve()` API의 입력 파라미터와 반환 값
    * 입력 파라미터: `y_true`(실제 클래스값 배열), `probas_pred`: Positive 칼럼의 예측 확률 배열
    * 반환값: 정밀도(임계값별 정밀도 값을 배열로 반환), 재현율(임계값별 재현율 값을 배열로 반환)
    
    
* `precision_recall_curve()`의 인자로 실제 값 데이터 세트와 레이블 값이 1일 때의 예측 확률 값을 입력
* 레이블 값이 1일 때의 예측 확률 값은 `predict_proba(X_Test)[:, 1]`로 `predict_proba()`의 반환 ndarray의 두 번째 칼럼 값에 해당하는 데이터 세트
* `precision_recall_curve()`는 일반적으로 0.11 ~ 0.95 정도의 임계값을 담은 넘파이 ndarray와 임계값에 해당하는 정밀도 및 재현율 값을 담은 넘파이 ndarray를 반환


* 반환되는 임계값이 너무 작은 값 단위로 많이 구성되어 있으므로 반환된 임계값의 데이터가 147건이므로 샘플로 10건만 추출하되, 임곗값을 15 단계로 추출해 좀 더 큰 임곗값과 그때의 정밀도와 재현율 값을 같이 살펴봄


```python
from sklearn.metrics import precision_recall_curve

# 레이블 값이 1일때의 예측 확률을 추출 
pred_proba_class1 = lr_clf.predict_proba(X_test)[:, 1] 

# 실제값 데이터 셋과 레이블 값이 1일 때의 예측 확률을 precision_recall_curve 인자로 입력 
precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_class1 )
print('반환된 분류 결정 임곗값 배열의 Shape:', thresholds.shape)
print('반환된 precisions 배열의 Shape:', precisions.shape)
print('반환된 recalls 배열의 Shape:', recalls.shape)

print("thresholds 5 sample:", thresholds[:5])
print("precisions 5 sample:", precisions[:5])
print("recalls 5 sample:", recalls[:5])

#반환된 임계값 배열 로우가 147건이므로 샘플로 10건만 추출하되, 임곗값을 15 Step으로 추출. 
thr_index = np.arange(0, thresholds.shape[0], 15)
print('샘플 추출을 위한 임계값 배열의 index 10개:', thr_index)
print('샘플용 10개의 임곗값: ', np.round(thresholds[thr_index], 2))

# 15 step 단위로 추출된 임계값에 따른 정밀도와 재현율 값 
print('샘플 임계값별 정밀도: ', np.round(precisions[thr_index], 3))
print('샘플 임계값별 재현율: ', np.round(recalls[thr_index], 3))
```

    반환된 분류 결정 임곗값 배열의 Shape: (147,)
    반환된 precisions 배열의 Shape: (148,)
    반환된 recalls 배열의 Shape: (148,)
    thresholds 5 sample: [0.11573101 0.11636721 0.11819211 0.12102773 0.12349478]
    precisions 5 sample: [0.37888199 0.375      0.37735849 0.37974684 0.38216561]
    recalls 5 sample: [1.         0.98360656 0.98360656 0.98360656 0.98360656]
    샘플 추출을 위한 임계값 배열의 index 10개: [  0  15  30  45  60  75  90 105 120 135]
    샘플용 10개의 임곗값:  [0.12 0.13 0.15 0.17 0.26 0.38 0.49 0.63 0.76 0.9 ]
    샘플 임계값별 정밀도:  [0.379 0.424 0.455 0.519 0.618 0.676 0.797 0.93  0.964 1.   ]
    샘플 임계값별 재현율:  [1.    0.967 0.902 0.902 0.902 0.82  0.77  0.656 0.443 0.213]
    

* 추출된 임계값 샘플 10개에 해당하는 정밀도 값과 재현율 값을 살펴보면 임계값이 증가할수록 정밀도 값은 동시에 높아지나 재현율 값은 낮아짐을 알 수 있음
* `precision_recall_curve()` API는 정밀도와 재현율의 임계값에 따른 값 변화를 곡선 형태의 그래프로 시각화하는 데 이용 가능

**임곗값의 변경에 따른 정밀도-재현율 변화 곡선을 그림**
* 정밀도는 점선으로, 재현율은 실선으로 표현
* 임계값이 낮을수록 많은 수의 양성 예측으로 인해 재현율 값이 극도로 높아지고 정밀도 값이 극도로 낮아짐


```python
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
%matplotlib inline

def precision_recall_curve_plot(y_test , pred_proba_c1):
    # threshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray 추출. 
    precisions, recalls, thresholds = precision_recall_curve( y_test, pred_proba_c1)
    
    # X축을 threshold값으로, Y축은 정밀도, 재현율 값으로 각각 Plot 수행. 정밀도는 점선으로 표시
    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')
    plt.plot(thresholds, recalls[0:threshold_boundary],label='recall')
    
    # threshold 값 X 축의 Scale을 0.1 단위로 변경
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    
    # x축, y축 label과 legend, 그리고 grid 설정
    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')
    plt.legend(); plt.grid()
    plt.show()
    
precision_recall_curve_plot( y_test, lr_clf.predict_proba(X_test)[:, 1] )

```


![png](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/pymldg/ch3/output_53_0.png)


### 3.2 정밀도와 재현율의 맹점
**1. 정밀도가 100%가 되는 방법**
* 확실한 기준이 되는 경우만 Positive로 예측하고 나머지는 모두 Negative로 예측
* 환자가 80세 이상이고 비만이며 이전에 암 진단을 받았고 암 세포의 크기가 상위 0.1% 이상이면 무조건 Positive, 다른 경우는 Negative로 예측하는 것
* 정밀도 = TP / (TP + FP)
    * 전체 환자 1000명 중 확실한 pOSITIVE 징후만 가진 환자는 단 1명이라고 하면 이 한 명만 Positive로 예측하고 나머지는 모두 Negative로 예측하더라도 FP는 0, TP는 1이 되므로 정밀도는 100%가 됨
    
    
**2. 재현율이 100%가 되는 방법**
* 모든 환자를 Positive로 예측
* 재현율 = TP / (TP + FN)
    * 환자 1000명을 다 Positive로 예측하면 실제 양성인 사람이 30명 정도라도 TN이 수치에 포함되지 않고 FN은 아예 0이므로 100%가 됨
    
    
**정리**
* 정밀도와 재현율 성능 수치도 어느 한쪽만 참조하면 극단적인 수치 조작이 가능
* 정밀도 또는 재현율 중 하나만 스코어가 좋고 다른 하나는 스코어가 나쁜 분류는 성능이 좋지 않은 분류로 간주할 수 있음
* 정밀도 또는 재현율 중 하나에 상대적인 중요도를 부여해 각 예측 상황에 맞는 분류 알고리즘을 튜닝할 수 있지만, 정밀도/재현율 중 하나만 강조하는 상황이 돼서는 안됨

## 4. F1 Score
* F1 스코어는 정밀도와 재현율을 결합한 지표. 정밀도와 재현율이 어느 한쪽으로 치우치지 않는 수치를 나타낼 때 상대적으로 높은 값을 가짐
* A 예측 모델의 경우 정밀도가 0.9, 재현율이 0.1로 극단적인 차이가 나고, B 예측 모델은 정밀도가 0.5, 재현율이 0.5로 B 모델이 A 모델에 비해 매우 우수한 F1 스코어를 가지게 됨


* 사이킷런은 F1 스코어를 구하기 위해 f1_score()라는 API를 제공
* 정밀도와 재현율 절의 예제에서 학습/예측한 로지스틱 회귀 기반 타이타닉 생존자 모델의 F1 스코어 측정


```python
from sklearn.metrics import f1_score 
f1 = f1_score(y_test , pred)
print('F1 스코어: {0:.4f}'.format(f1))

```

    F1 스코어: 0.7966
    

* 타이타닉 생존자 예측에서 임곗값을 변화시키면서 F1 스코어를 포함한 평가 지표 측정


```python
def get_clf_eval(y_test , pred):
    confusion = confusion_matrix( y_test, pred)
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test , pred)
    # F1 스코어 추가
    f1 = f1_score(y_test,pred)
    print('오차 행렬')
    print(confusion)
    # f1 score print 추가
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1:{3:.4f}'.format(accuracy, precision, recall, f1))

thresholds = [0.4 , 0.45 , 0.50 , 0.55 , 0.60]
pred_proba = lr_clf.predict_proba(X_test)
get_eval_by_threshold(y_test, pred_proba[:,1].reshape(-1,1), thresholds)

```

    임곗값: 0.4
    오차 행렬
    [[97 21]
     [11 50]]
    정확도: 0.8212, 정밀도: 0.7042, 재현율: 0.8197, F1:0.7576
    임곗값: 0.45
    오차 행렬
    [[105  13]
     [ 13  48]]
    정확도: 0.8547, 정밀도: 0.7869, 재현율: 0.7869, F1:0.7869
    임곗값: 0.5
    오차 행렬
    [[108  10]
     [ 14  47]]
    정확도: 0.8659, 정밀도: 0.8246, 재현율: 0.7705, F1:0.7966
    임곗값: 0.55
    오차 행렬
    [[111   7]
     [ 16  45]]
    정확도: 0.8715, 정밀도: 0.8654, 재현율: 0.7377, F1:0.7965
    임곗값: 0.6
    오차 행렬
    [[113   5]
     [ 17  44]]
    정확도: 0.8771, 정밀도: 0.8980, 재현율: 0.7213, F1:0.8000
    

**결과 정리**

평가 지표|0.4|0.45|0.5|0.55|0.6
:-------:|:-:|:--:|:-:|:--:|:-:
정확도|0.8380|0.8492|0.8492|0.8659|0.8771
정밀도|0.7286|0.7656|0.7742|0.8364|0.8824
재현율|0.8361|0.8033|0.7869|0.7541|0.7377
F1| 0.7576|0.7869|0.7966|0.7965|0.800

* F1 스코어는 임계값이 0.6일 때 가장 좋은 값을 보여줌. 하지만 임계값이 0.6인 경우 재현율이 크게 감소하고 있음

## 5. ROC Curve와 AUC
* ROC 곡선과 AUC 스코어는 이진 분류의 예측 성능 측정에서 중요하게 사용되는 지표
* ROC 곡선은 FPR(False Positive Rate)이 변할 때 TPR(True Positive Rate)이 어떻게 변하는지를 나타내는 곡선
    * FPR을 X축으로, TPR을 Y축으로 잡으면 FPR의 변화에 따른 TPR의 변화가 곡선 형태로 나타남
    * TPR은 재현율을 의미. 재현율에 대응하는 지표로 TNR이라고 불리는 특이성이 있음
    * 재현율은 실제값 Positive가 정확히 예측돼야 하는 수준을 나타냄. 특이성은 실제값 Negative가 정확히 예측돼야 하는 수준을 나타냄
    * TPR = TP / (FN + TP). TNR = TN / (FP + TN)
    * ROC 곡선의 X축 기준인 FPR(False Positive Rate) = FP / (FP + TN)이므로 1-  TNR 또는 1 - 특이성으로 표현

![ROC 곡선 예시](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQITn6oWkGlCxAsF3WEdo_kBNFA9l0LrdznAw&usqp=CAU)

* 가운데 직선은 ROC 곡선의 최저 값. ROC 곡선이 가운데 직선에 가까울수록 성능이 떨어지는 것이며, 멀어질수록 성능이 뛰어난 것


* ROC 곡선은 FPR을 0 ~ 1까지 변경하면서 TPR의 변화 값을 구함
    * 분류 결정 임계값은 Positive 예측값을 결정하는 확률의 기준이기 때문에 FPR을 0으로 만들려면 임계값을 1로 지정하면 됨
        * 임계값을 1로 지정하면 Positive 예측 기준이 매우 높기 때문에 분류기가 임계값보다 높은 확률을 가진 데이터를 Positive로 예측할 수 없기 때문
        * 아예 Positive로 얘측하지 않기 때문에 FP 값이 0이 되므로 자연스럽게 FPR 0이 됨
    * FPR을 1로 만들기 위해서는 TN을 0으로 만들면 됨. TN을 0으로 만들려면 분류 결정 임계값을 0으로 지정하면 됨
        * 분류기의 Positive 확률 기준이 너무 낮아서 다 Positive로 예측
        * 아예 Negative 예측이 없기 때문에 TN이 0이 되고 FPR 값은 1이 됨
    * 임계값을 1부터 0까지 변화시키면서 FPR을 구하고 이 FPR 값의 변화에 따른 TPR 값을 구하는 것이 ROC 곡선
    * 임계값을 1부터 0까지 거꾸로 변화시키면서 구한 재현율 곡선 형태와 비슷
    
    
* 사이킷런은 ROC 곡선을 구하기 위해 `roc_curve()` API를 제공
    * 사용법은 `precision_recall_curve()` API와 유사
    * 반환값이 FPR, TPR, 임계값으로 구성되어 있음
* `roc_curve()` 주요 입력 파라미터와 반환 값
    * 입력 파라미터: `y_true`(실제 클래스 값 array), `y_score`(`predict_prova()`의 반환 값 array에서 Positive 칼럼의 예측 확률이 보통 사용됨)
    * 반환값: fpr(fpr 값을 array로 반환), tpr(tpr 값을 array로 반환), thresholds(threshold 값 array)

**`roc_curve()` API를 이용해 타이타닉 생존자 예측 모델의 FPR, TPR, 임계값 구하기**
* 정밀도와 재현율에서 학습한 LogisticRegression 객체의 `predict_proba()` 결과를 다시 이용해 `roc_curve()`의 결과를 도출


```python
from sklearn.metrics import roc_curve

# 레이블 값이 1일때의 예측 확률을 추출 
pred_proba_class1 = lr_clf.predict_proba(X_test)[:, 1] 

fprs , tprs , thresholds = roc_curve(y_test, pred_proba_class1)
# 반환된 임곗값 배열에서 샘플로 데이터를 추출하되, 임곗값을 5 Step으로 추출. 
# thresholds[0]은 max(예측확률)+1로 임의 설정됨. 이를 제외하기 위해 np.arange는 1부터 시작
thr_index = np.arange(1, thresholds.shape[0], 5)
print('샘플 추출을 위한 임곗값 배열의 index:', thr_index)
print('샘플 index로 추출한 임곗값: ', np.round(thresholds[thr_index], 2))

# 5 step 단위로 추출된 임계값에 따른 FPR, TPR 값
print('샘플 임곗값별 FPR: ', np.round(fprs[thr_index], 3))
print('샘플 임곗값별 TPR: ', np.round(tprs[thr_index], 3))

```

    샘플 추출을 위한 임곗값 배열의 index: [ 1  6 11 16 21 26 31 36 41 46]
    샘플 index로 추출한 임곗값:  [0.94 0.73 0.62 0.52 0.44 0.28 0.15 0.14 0.13 0.12]
    샘플 임곗값별 FPR:  [0.    0.008 0.025 0.076 0.127 0.254 0.576 0.61  0.746 0.847]
    샘플 임곗값별 TPR:  [0.016 0.492 0.705 0.738 0.803 0.885 0.902 0.951 0.967 1.   ]
    

* `roc_curve()` 결과를 보면 임계값이 1에 가까운 값에서 점점 작아지면서 FPR이 점점 커짐, fpr이 조금씩 커질 때 tpr은 가파르게 커짐

**FPR의 변화에 따른 TPR의 변화를 ROC 곡선으로 시각화**


```python
from sklearn.metrics import roc_curve

# 레이블 값이 1일때의 예측 확률을 추출 
pred_proba_class1 = lr_clf.predict_proba(X_test)[:, 1] 
print('max predict_proba:', np.max(pred_proba_class1))

fprs , tprs , thresholds = roc_curve(y_test, pred_proba_class1)
print('thresholds[0]:', thresholds[0])
# 반환된 임곗값 배열 로우가 47건이므로 샘플로 10건만 추출하되, 임곗값을 5 Step으로 추출. 
thr_index = np.arange(0, thresholds.shape[0], 5)
print('샘플 추출을 위한 임곗값 배열의 index 10개:', thr_index)
print('샘플용 10개의 임곗값: ', np.round(thresholds[thr_index], 2))

# 5 step 단위로 추출된 임계값에 따른 FPR, TPR 값
print('샘플 임곗값별 FPR: ', np.round(fprs[thr_index], 3))
print('샘플 임곗값별 TPR: ', np.round(tprs[thr_index], 3))
```

    max predict_proba: 0.943262794352988
    thresholds[0]: 1.943262794352988
    샘플 추출을 위한 임곗값 배열의 index 10개: [ 0  5 10 15 20 25 30 35 40 45]
    샘플용 10개의 임곗값:  [1.94 0.87 0.63 0.55 0.44 0.32 0.15 0.14 0.13 0.12]
    샘플 임곗값별 FPR:  [0.    0.008 0.025 0.059 0.127 0.203 0.559 0.602 0.695 0.847]
    샘플 임곗값별 TPR:  [0.    0.246 0.672 0.738 0.787 0.885 0.902 0.951 0.967 0.984]
    

* ROC 곡선 자체는 FPR과 TPR의 변화 값을 보는데 이용하며 분류의 성능 지표로 사용되는 것은 ROC 곡선 면적에 기반한 AUC 값으로 결정
* AUC 값은 ROC 곡선 밑의 면적을 구한 것으로서 1에 가까울수록 좋은 수치
* AUC 수치가 커지려면 FPR이 작은 상테에서 얼마나 큰 TPR을 얻을 수 있느냐가 관건
    * 가운데 직선에서 멀어지고 왼쪽 상단 모서리 쪽으로 가파르게 곡선이 이동할수록 직사각형에 가까운 곡선이 되어 면석이 1에 가까워지는 좋은 ROC AUC 선을 수치를 얻게 됨


```python
def roc_curve_plot(y_test , pred_proba_c1):
    # 임곗값에 따른 FPR, TPR 값을 반환 받음. 
    fprs , tprs , thresholds = roc_curve(y_test ,pred_proba_c1)

    # ROC Curve를 plot 곡선으로 그림. 
    plt.plot(fprs , tprs, label='ROC')
    # 가운데 대각선 직선을 그림. 
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    
    # FPR X 축의 Scale을 0.1 단위로 변경, X,Y 축명 설정등   
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    plt.xlim(0,1); plt.ylim(0,1)
    plt.xlabel('FPR( 1 - Sensitivity )'); plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.show()
    
roc_curve_plot(y_test, lr_clf.predict_proba(X_test)[:, 1] )

```


![png](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/pymldg/ch3/output_69_0.png)



```python
from sklearn.metrics import roc_auc_score

### 아래는 roc_auc_score()의 인자를 잘못 입력한 것으로, 책에서 수정이 필요한 부분입니다. 
### 책에서는 roc_auc_score(y_test, pred)로 예측 타겟값을 입력하였으나 
### roc_auc_score(y_test, y_score)로 y_score는 predict_proba()로 호출된 예측 확률 ndarray중 Positive 열에 해당하는 ndarray입니다. 

#pred = lr_clf.predict(X_test)
#roc_score = roc_auc_score(y_test, pred)

pred_proba = lr_clf.predict_proba(X_test)[:, 1]
roc_score = roc_auc_score(y_test, pred_proba)
print('ROC AUC 값: {0:.4f}'.format(roc_score))

```

    ROC AUC 값: 0.8987
    

* 타이타닉 생존자 예측 로지스틱 회귀 모델의 ROC AUV 값은 약 0.8987로 측정됨


**`get_clf_eval()` 함수에 `roc_auc_score()`를 이용해 ROC AUC 값을 측정하는 로직 추가**
* ROC AUC는 예측 확률값을 기반으로 계산되므로 `get_clf_eval()` 함수의 인자로 받을 수 있도록 `get_clf_eval(y_test, pred = None, pred_proba = None)`로 함수형을 변경
* `get_clf_eval()` 함수는 정확도, 정밀도, 재현율, F1 스코어, ROC AUC 값까지 출력 가능


```python
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
