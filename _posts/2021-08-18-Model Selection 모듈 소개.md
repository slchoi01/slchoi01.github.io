---
layout: post
title: "Model Selection"
date: 2021-08-16
excerpt: "Model Selection 소개"
tags: [machine learning, data science]
comments: true
---

* `model_selection`: 학습 데이터와 테스트 데이터 세트를 분리하거나 교차 검증 분할 및 평가, Estimator의 하이퍼 파라미터 튜닝을 위한 다양한 함수와 클래스를제공

### 1. 학습/테스트 데이터 셋 분리 – train_test_split()


```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 테스트 데이터 세트를 이용하지 않고 학습 데이터 셋으로만 학습 후 예측 수행
iris = load_iris()
dt_clf = DecisionTreeClassifier()
train_data = iris.data
train_label = iris.target
dt_clf.fit(train_data, train_label)

pred = dt_clf.predict(train_data)
print('예측 정확도:',accuracy_score(train_label,pred))
```

    예측 정확도: 1.0
    

* 정확도가 100%인 이유는 이미 학습한 학습 데이터 세트를 기반으로 예측했기 때문
* 예측을 수행하는 데이터 세트는 학습을 수행한 학습용 데이터 세ㅌ가 아닌 전용이 테스트 데이터 세트여야함


* `train_test_split()`을 활용해 원본 데이터 세트에서 학습 및 테스트 세트를 쉽게 분리 가능
    * 첫 번째 파라이미터로 피처 데이터 세트, 두 번째 파라미터로 레이블 데이터 세트를 입력
    * `test_size`: 전체 데이터에서 테스트 데이터 세트 크기를 얼마로 샘플링할 것인가를 결정. 디폴트는 0.25%
    * `shuffle`: 데이터를 분리하기 전에 데이터를 미리 섞을지 결정. 디폴트는 True. 데이터를 분산시켜 효율적인 학습 및 테스트 데이터를 생성
    * `train_test_split()`의 반환값은 튜플 형태. 학습용 데이터의 피처 데이터 세트, 테스트용 데이터의 피처 데이터 세트, 학습용 데이터의 레이블 데이터 세트, 테스트용 데이터의 레이블 데이터 세트가 반환됨


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# sklearn.model_section모듈에서 train_test_split을 로드
from sklearn.model_selection import train_test_split

dt_clf = DecisionTreeClassifier( )
iris_data = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, 
                                                    test_size=0.3, random_state=121)
```

* 학습 데이터를 기반으로 DecisionTreeClassifier를 학습하고 이 모델을 이용해 예측 정확도를 측정


```python
dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_test)
print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test,pred)))
```

    예측 정확도: 0.9556
    

### 2. 교차 검증
* 과적합: 모델이 학습 데이터에만 과도하게 최적화되어, 실제 예측을 다른 데이터로 수행할 경우 예측 성능이 과도하게 떨어지는 것
* 고정된 학습, 테스트 데이터로 평가하다보면 테스트 데이터에만 최적의 성능을 발휘할 수 있도록 편향되게 모델을 유도하는 경향이 발생하므로 해당 테스트 데이터에만 과적합되는 학습 모델이 만들어져 다른 테스트용 데이터가 들어올 경우 성능이 저하됨
* 이 문제를 해결하기 위해 교차 검증을 이용해 다양한 학습과 평가를 수행

* **교차 검증**: 별도의 여러 세트로 구성된 학습 데이터 세트와 검증 데이터 세트에서 학습과 평가를 수행. 각 세트에서 수행한 평가 결과에 따라 하이퍼 파라미터 튜닝 등의 모델 최적화를 쉽게 할 수 있음
    * ML 모델의 성능 평가는 교차 검증 기반으로 1차 평가를 한 후 최종적으로 테스트 데이터 세트에 적용해 평가하는 프로세스
    * 테스트 데이터 세트 외에 별도의 검증 데이터 세트를 둬서 최종 평가 이전에 학습된 모델을 다양하게 평가하는 데 사용

#### 1. K 폴드 교차 검증
* K개의 데이터 폴드 세트를 만들어서 K번만큼 각 폴트 세트에 학습과 검증 평가를 반복적으로 수행하는 방법
* 학습 데이터 세트와 검증 데이터 세트를 점진적으로 변경하면서 마지막 K번째까지 학습과 검증을 수행하는 것
* K개의 예측 평가를 구했으면 평균해서 K 폴드 평가 결과로 반영하면 됨
<br><br>
* 사이킷런에서는 K 폴드 교차 검증 프로세스를 구현하기 위해 KFold와 StratifiedKFold 클래스를 제공

**KFold 클래스를 이용해 붓꽃 데이터 세트를 교차 검증하고 예측 정확도 확인**


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=156)

# 5개의 폴드 세트로 분리하는 KFold 객체와 폴드 세트별 정확도를 담을 리스트 객체 생성.
kfold = KFold(n_splits=5)
cv_accuracy = []
print('붓꽃 데이터 세트 크기:',features.shape[0])

```

    붓꽃 데이터 세트 크기: 150
    


```python
n_iter = 0

# KFold객체의 split( ) 호출하면 폴드 별 학습용, 검증용 테스트의 로우 인덱스를 array로 반환  
for train_index, test_index  in kfold.split(features):
    # kfold.split( )으로 반환된 인덱스를 이용하여 학습용, 검증용 테스트 데이터 추출
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    #학습 및 예측 
    dt_clf.fit(X_train , y_train)    
    pred = dt_clf.predict(X_test)
    n_iter += 1
    # 반복 시 마다 정확도 측정 
    accuracy = np.round(accuracy_score(y_test,pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('\n#{0} 교차 검증 정확도 :{1}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}'
          .format(n_iter, accuracy, train_size, test_size))
    print('#{0} 검증 세트 인덱스:{1}'.format(n_iter,test_index))
    cv_accuracy.append(accuracy)
    
# 개별 iteration별 정확도를 합하여 평균 정확도 계산 
print('\n## 평균 검증 정확도:', np.mean(cv_accuracy)) 
```

    
    #1 교차 검증 정확도 :1.0, 학습 데이터 크기: 120, 검증 데이터 크기: 30
    #1 검증 세트 인덱스:[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
     24 25 26 27 28 29]
    
    #2 교차 검증 정확도 :0.9667, 학습 데이터 크기: 120, 검증 데이터 크기: 30
    #2 검증 세트 인덱스:[30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53
     54 55 56 57 58 59]
    
    #3 교차 검증 정확도 :0.8667, 학습 데이터 크기: 120, 검증 데이터 크기: 30
    #3 검증 세트 인덱스:[60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83
     84 85 86 87 88 89]
    
    #4 교차 검증 정확도 :0.9333, 학습 데이터 크기: 120, 검증 데이터 크기: 30
    #4 검증 세트 인덱스:[ 90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107
     108 109 110 111 112 113 114 115 116 117 118 119]
    
    #5 교차 검증 정확도 :0.7333, 학습 데이터 크기: 120, 검증 데이터 크기: 30
    #5 검증 세트 인덱스:[120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137
     138 139 140 141 142 143 144 145 146 147 148 149]
    
    ## 평균 검증 정확도: 0.9
    

#### 2. Stratified K 폴드
* 불균형한 분포도를 가진 레이블 데이터 집합을 위한 K 폴드 방식
    * 불균형한 분포도를 가진 레이블 데이터 집합은 특정 레이블 값이 특이하게 많거나 매우 적어서 값의 분포가 한쪽으로 치우치는 것을 의미
* K 폴드가 레이블 데이터 집합이 원본 데이터 집합의 레이블 분포를 학습 및 테스트 세트에 제대로 분배하지 못하는 경우의 문제를 해결
* 원본 데이터의 레이블 분포를 고려한 뒤 이 분포와 동일하게 학습과 검증 데이터 세트를 분배

**StratifiedKFold 클래스 사용**
* K 폴드의 문제점을 확인 후 StratifiedKFold 클래스를 이용해 개선


```python
import pandas as pd

iris = load_iris()

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label']=iris.target
iris_df['label'].value_counts()
```




    2    50
    1    50
    0    50
    Name: label, dtype: int64



* 3개의 폴드 세트를 KFold로 생성하고, 각 교차 검증마다 생성되는 학습/검증 레이블 데이터 값의 분포도 확인
    * 첫 번째 교차 검증의 경우 학습 레이블은 1, 2밖에 없으므로 0의 경우는 전혀 학습하지 못하고 검증 레이블은 0밖에 없으므로 학습 모델은 절대 0을 예측하지 못함. 따라서 이런 유형으로 교차 검증 데이터 세트를 분할하면 검증 예측 정확도는 0이 됨


```python
kfold = KFold(n_splits=3)
# kfold.split(X)는 폴드 세트를 3번 반복할 때마다 달라지는 학습/테스트 용 데이터 로우 인덱스 번호 반환. 
n_iter =0
for train_index, test_index  in kfold.split(iris_df):
    n_iter += 1
    label_train= iris_df['label'].iloc[train_index]
    label_test= iris_df['label'].iloc[test_index]
    print('## 교차 검증: {0}'.format(n_iter))
    print('학습 레이블 데이터 분포:\n', label_train.value_counts())
    print('검증 레이블 데이터 분포:\n', label_test.value_counts())
    
```

    ## 교차 검증: 1
    학습 레이블 데이터 분포:
     2    50
    1    50
    Name: label, dtype: int64
    검증 레이블 데이터 분포:
     0    50
    Name: label, dtype: int64
    ## 교차 검증: 2
    학습 레이블 데이터 분포:
     2    50
    0    50
    Name: label, dtype: int64
    검증 레이블 데이터 분포:
     1    50
    Name: label, dtype: int64
    ## 교차 검증: 3
    학습 레이블 데이터 분포:
     1    50
    0    50
    Name: label, dtype: int64
    검증 레이블 데이터 분포:
     2    50
    Name: label, dtype: int64
    

* StartifiedKFold는 레이블 데이터 분포도에 따라 학습/검증 데이터를 나누기 때문에 `split()` 메서드에 인자로 피처 데이터 세트뿐만 아니라 레이블 데이터 세트도 필요


```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3)
n_iter=0

for train_index, test_index in skf.split(iris_df, iris_df['label']):
    n_iter += 1
    label_train= iris_df['label'].iloc[train_index]
    label_test= iris_df['label'].iloc[test_index]
    print('## 교차 검증: {0}'.format(n_iter))
    print('학습 레이블 데이터 분포:\n', label_train.value_counts())
    print('검증 레이블 데이터 분포:\n', label_test.value_counts())
```

    ## 교차 검증: 1
    학습 레이블 데이터 분포:
     2    33
    1    33
    0    33
    Name: label, dtype: int64
    검증 레이블 데이터 분포:
     2    17
    1    17
    0    17
    Name: label, dtype: int64
    ## 교차 검증: 2
    학습 레이블 데이터 분포:
     2    33
    1    33
    0    33
    Name: label, dtype: int64
    검증 레이블 데이터 분포:
     2    17
    1    17
    0    17
    Name: label, dtype: int64
    ## 교차 검증: 3
    학습 레이블 데이터 분포:
     2    34
    1    34
    0    34
    Name: label, dtype: int64
    검증 레이블 데이터 분포:
     2    16
    1    16
    0    16
    Name: label, dtype: int64
    

**StartifiedKFold를 이용한 붓꽃 데이터 교차 검증**


```python
dt_clf = DecisionTreeClassifier(random_state=156)

skfold = StratifiedKFold(n_splits=3)
n_iter=0
cv_accuracy=[]

# StratifiedKFold의 split( ) 호출시 반드시 레이블 데이터 셋도 추가 입력 필요  
for train_index, test_index  in skfold.split(features, label):
    # split( )으로 반환된 인덱스를 이용하여 학습용, 검증용 테스트 데이터 추출
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    #학습 및 예측 
    dt_clf.fit(X_train , y_train)    
    pred = dt_clf.predict(X_test)

    # 반복 시 마다 정확도 측정 
    n_iter += 1
    accuracy = np.round(accuracy_score(y_test,pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('\n#{0} 교차 검증 정확도 :{1}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}'
          .format(n_iter, accuracy, train_size, test_size))
    print('#{0} 검증 세트 인덱스:{1}'.format(n_iter,test_index))
    cv_accuracy.append(accuracy)
    
# 교차 검증별 정확도 및 평균 정확도 계산 
print('\n## 교차 검증별 정확도:', np.round(cv_accuracy, 4))
print('## 평균 검증 정확도:', np.mean(cv_accuracy)) 
```

    
    #1 교차 검증 정확도 :0.9804, 학습 데이터 크기: 99, 검증 데이터 크기: 51
    #1 검증 세트 인덱스:[  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  50
      51  52  53  54  55  56  57  58  59  60  61  62  63  64  65  66 100 101
     102 103 104 105 106 107 108 109 110 111 112 113 114 115 116]
    
    #2 교차 검증 정확도 :0.9216, 학습 데이터 크기: 99, 검증 데이터 크기: 51
    #2 검증 세트 인덱스:[ 17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  67
      68  69  70  71  72  73  74  75  76  77  78  79  80  81  82  83 117 118
     119 120 121 122 123 124 125 126 127 128 129 130 131 132 133]
    
    #3 교차 검증 정확도 :0.9792, 학습 데이터 크기: 102, 검증 데이터 크기: 48
    #3 검증 세트 인덱스:[ 34  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  84  85
      86  87  88  89  90  91  92  93  94  95  96  97  98  99 134 135 136 137
     138 139 140 141 142 143 144 145 146 147 148 149]
    
    ## 교차 검증별 정확도: [0.9804 0.9216 0.9792]
    ## 평균 검증 정확도: 0.9604
    

* Stratified K 폴드의 경우 원본 데이터의 레이블 분포도 특성을 반영한 학습 및 검증 데이터 세트를 만들 수 있으므로 왜곡된 레이블 데이터 세터에서는 반드시 Strarified K 폴드를 이용해 교차 검증을 해야함
* 분류에서의 교차 검증은 Stratified K 폴드로 분할돼야하며 회귀에서는 Stratified K 폴드가 지원되지 않음
    * 이유: 회귀의 결정값은 이산값 형태의 레이블이 아니라 연속된 숫자값이기 때문에 결정값별로 분포를 정하는 의미가 없음

#### 3. cross_val_score( )
* KFold로 데이터를 학습하고 예측하는 코드의 순서
    1. 폴드 세트를 설정
    2. for 루프에서 반복으로 학습 및 테스트 데이터의 인덱스를 추출
    3. 반복적으로 학습과 예측을 수행하고 예측 성능을 반환


* `cross_val_score()`: 위의 과정을 한꺼번에 수행해줌
* `cross_val_score()`의 주요 파라미터
    * `estimator`: 분류 알고리즘 클래스인 Classfier(Stratified K 폴드 사용) 또는 회귀 알고리즘 클래스인 Regressor(K 폴드 사용)를 의미
    * `X`: 피처 데이터 세트, `y`: 레이블 데이터 세트
    * `scoring`: 예측 성능 평가 지표, `cv`: 교차 검증 폴드 수
    

**`cross_val_score` 사용**


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score , cross_validate
from sklearn.datasets import load_iris

iris_data = load_iris()
dt_clf = DecisionTreeClassifier(random_state=156)

data = iris_data.data
label = iris_data.target

# 성능 지표는 정확도(accuracy) , 교차 검증 세트는 3개 
scores = cross_val_score(dt_clf , data , label , scoring='accuracy',cv=3)
print('교차 검증별 정확도:',np.round(scores, 4))
print('평균 검증 정확도:', np.round(np.mean(scores), 4))
```

    교차 검증별 정확도: [0.9804 0.9216 0.9792]
    평균 검증 정확도: 0.9604
    

### 3. GridSearchCV
* GridSearchCV를 이용해 Classifier나 Regressor 같은 알고리즘에 사용되는 하이퍼 파라미터를 순차적으로 입력하면 편리하게 최적의 파라미터를 도출할 수 있는 방안을 제공

* GridSearchCV는 교차 검증을 기반으로 하이퍼 파라미터의 최적 값을 찾게 해줌. 순차적으로 파라미터를 테스트하므로 수행시간이 상대적으로 오래 걸림

* GridSearchCv 주요 파라미터
    * `estimator`: classifier, regressor, pipline이 사용
    * `param_grid`: key + 리스트 값을 가지는 딕셔너리가 주어짐. estimator 튜닝을 위해 파라미터명과 사용될 여러 파라미터 값을 지정
    * `scoring`: 예측 성능을 측정할 평가 방법을 지정
    * `cv`: 교차 검증을 위해 분할되는 학습/테스트 세트의 개수를 지정
    * `refit`: 디폴트는 True, True로 생성 시 가장 최적의 하이퍼 파라미터를 찾은 뒤 입력된 estimator 객체를 해당 하이퍼 파라미터로 재학습시킴

**GridSearchCV 활용**


```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# 데이터를 로딩하고 학습데이타와 테스트 데이터 분리
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, 
                                                    test_size=0.2, random_state=121)
dtree = DecisionTreeClassifier()

### parameter 들을 dictionary 형태로 설정 (key는 문자열로, 하이퍼 파라미터의 값은 리스트 형으로 설정)
parameters = {'max_depth':[1,2,3], 'min_samples_split':[2,3]}
```

* 학습 데이터 세트를 GridSearchCV 객체의 fit 메서드에 인자로 입력
* 주요 칼럼 의미
    * params: 수행할 때마다 적용된 개발 하이퍼 파라미터 값
    * rank_test_score: 하이퍼 파라미터별로 성능이 좋은 score 순위. 1이 가장 뛰어난 순위이며 이때의 파라미터가 최적의 하이퍼 파라미터
    * mean_test_score: 개별 하이퍼 파라미터별로 CV의 폴딩 테스트 세트에 대해 총 수행한 평가 평균값


```python
import pandas as pd

# param_grid의 하이퍼 파라미터들을 3개의 train, test set fold 로 나누어서 테스트 수행 설정.  
### refit=True 가 default 임. True이면 가장 좋은 파라미터 설정으로 재 학습 시킴.  
grid_dtree = GridSearchCV(dtree, param_grid=parameters, cv=3, refit=True)

# 붓꽃 Train 데이터로 param_grid의 하이퍼 파라미터들을 순차적으로 학습/평가 .
grid_dtree.fit(X_train, y_train)

# GridSearchCV 결과 추출하여 DataFrame으로 변환
scores_df = pd.DataFrame(grid_dtree.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', \
           'split0_test_score', 'split1_test_score', 'split2_test_score']]
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
      <th>params</th>
      <th>mean_test_score</th>
      <th>rank_test_score</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{'max_depth': 1, 'min_samples_split': 2}</td>
      <td>0.700000</td>
      <td>5</td>
      <td>0.700</td>
      <td>0.7</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{'max_depth': 1, 'min_samples_split': 3}</td>
      <td>0.700000</td>
      <td>5</td>
      <td>0.700</td>
      <td>0.7</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>2</th>
      <td>{'max_depth': 2, 'min_samples_split': 2}</td>
      <td>0.958333</td>
      <td>3</td>
      <td>0.925</td>
      <td>1.0</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>3</th>
      <td>{'max_depth': 2, 'min_samples_split': 3}</td>
      <td>0.958333</td>
      <td>3</td>
      <td>0.925</td>
      <td>1.0</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>{'max_depth': 3, 'min_samples_split': 2}</td>
      <td>0.975000</td>
      <td>1</td>
      <td>0.975</td>
      <td>1.0</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>5</th>
      <td>{'max_depth': 3, 'min_samples_split': 3}</td>
      <td>0.975000</td>
      <td>1</td>
      <td>0.975</td>
      <td>1.0</td>
      <td>0.95</td>
    </tr>
  </tbody>
</table>
</div>



* GridSearchCV 객체의 `fit()`을 수행하면 최고 성능을 나타낸 하이퍼 파라미터의 값과 그때의 평가 결과 값이 각각 `best_params_`, `best_score_` 속성에 기록됨


```python
print('GridSearchCV 최적 파라미터:', grid_dtree.best_params_)
print('GridSearchCV 최고 정확도: {0:.4f}'.format(grid_dtree.best_score_))
```

    GridSearchCV 최적 파라미터: {'max_depth': 3, 'min_samples_split': 2}
    GridSearchCV 최고 정확도: 0.9750
    

* `refit = True`일 경우 GridSearchCV가 최적 성능을 나타내는 하이퍼 파라미터로 Estimator를 학습해 `best_estimator_`로 저장함 <br> <br>
* `best_estimator_`를 이용해 `train_test_split()`으로 분리한 테스트 데이터 세트에 대해 예측하고 성능 평가


```python
# GridSearchCV의 refit으로 이미 학습이 된 estimator 반환
estimator = grid_dtree.best_estimator_

# GridSearchCV의 best_estimator_는 이미 최적 하이퍼 파라미터로 학습이 됨
pred = estimator.predict(X_test)
print('테스트 데이터 세트 정확도: {0:.4f}'.format(accuracy_score(y_test,pred)))
```

    테스트 데이터 세트 정확도: 0.9667
    

* 학습 데이터를 GridSearchCV를 이용해 최적 하이퍼 파라미터 튜닝을 수행한 뒤 별도의 테스트 세트에서 평가하는 것이 일반적인 머신러닝 모델 적용 방법


```python

```
