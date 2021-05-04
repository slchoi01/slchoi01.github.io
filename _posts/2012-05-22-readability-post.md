---
layout: post
title: "Numpy"
date: 2021-05-03
excerpt: "Numpy"
tags: [codeit_study, data science]
comments: true
---

# Numpy array 생성
---

### 파이썬 리스트를 통해 생성

```python
import numpy
# numpy array 생성
array1 = numpy.array([2, 3, 5, 7, 11, 13, 17,19, 23, 29, 31])
array1
```

```python
array([2, 3, 5, 7, 11, 13, 17,19, 23, 29, 31])
```

### 균일한 값으로 생성

- `numpy` 모듈의 `full` 메소드를 사용하면, 모든 값이 같은 numpy array 생성 가능

```python
array1 = numpy.full(6, 7)
    
print(array1)
```

```python
[7 7 7 7 7 7]
```

- 모든 값이 0인 array 생성

```python
array2 = numpy.zeros(6, dtype=int)
    
print(array2)
```

```python
[0 0 0 0 0 0]
```

- 모든 값이 1인  array 생성

```python
array2 = numpy.ones(6, dtype=int)
    
print(array2)
```

```python
[1 1 1 1 1 1]
```

### 랜덤한 값들로 생성

- 임의의 값들로 배열을 생성하고 싶을 때, `numpy`의 `random` 모듈의 `random` 함수 사용
- `numpy` 모듈 안에 `random`이라는 모듈이  있고, 그 안에 `random`이라는 함수가 있음

```python
array1 = numpy.random.random(6)
array2 = numpy.random.random(6)
    
print(array1)
print()
print(array2)
```

```python
[0.42214929 0.45275673 0.57978413 0.61417065 0.39448558 0.03347601]

[0.42521953 0.65091589 0.94045742 0.18138103 0.27150749 0.8450694 ]
```

### 연속된 값들이 담긴 numpy array 생성

- `numpy` 모듈의 `arange` 함수 사용
- `arange` 함수는 파이썬의 기본 함수인 `range` 함수와 비슷하게 동작
- 파라미터가 1개인 경우: `arange(m)`을 하면 `0`부터 `m-1`까지의 값들이 담긴 numpy array가 리턴

```python
array1 = numpy.arange(6)
print(array1)
```

```python
[0 1 2 3 4 5]
```

- 파라미터가 2개인 경우:  `arange(n, m)`을 하면 `n`부터 `m-1`까지의 값들이 담긴 numpy array가 리턴

```python
array1 = numpy.arange(2, 7)
print(array1)
```

```python
[2 3 4 5 6]
```

- 파라미터가 3개인 경우: `arange(n, m, s)`를 하면 `n`부터 `m-1`까지의 값들 중 간격이 `s`인 값들이 담긴 numpy array가 리턴

```python
array1 = numpy.arange(3, 17, 3)
print(array1)
```

```python
[ 3  6  9 12 15]
```

# Numpy 불린 연산
---

```python
import numpy
array1 = np.array([1, 2, 3, 4, 5])
```

```python
array1 > 4
```

```python
array([False, False, False, False, True])
```

```python
booleans = np.array([False, False, False, False, True])

```

```python
np.where(booleans)
```

```python
(array([4]),)
```

```python
filter = np.where(booleans)
array1[filter]
```

```python
array([5])
```

# Numpy 기본 통계
---

### 최댓값, 최솟값

- `max` 메소드와 `min` 메소드를 사용하면 numpy array의 최댓값과 최솟값을 구할 수 있음

```python
import numpy as np

array1 = np.array([14, 6, 13, 21, 23, 31, 9, 5])

print(array1.max()) # 최댓값
print(array1.min()) # 최솟값
```

```
31
5
```

### 평균값

- `mean` 메소드를 사용하면 numpy array의 평균값을 구할 수 있음

```python
import numpy as np

array1 = np.array([14, 6, 13, 21, 23, 31, 9, 5])

print(array1.mean()) # 평균값
```

```python
15.25
```

### 중앙값

- `median` 메소드를 사용하면 중간값을 구할 수 있음
- `median`은 numpy array의 메소드가 아니라 numpy의 메소드

```python
import numpy as np

array1 = np.array([8, 12, 9, 15, 16])
array2 = np.array([14, 6, 13, 21, 23, 31, 9, 5])

print(np.median(array1)) # 중앙값
print(np.median(array2)) # 중앙값
```

```python
12.0
13.5
```

- `array1`을 정렬하면 중앙값이 $12$
- `array2`에는 짝수개의 요소가 있기 때문에 중앙값이 $13$과 $14$ 두 개. 둘의 평균값을 내면 $13.5$

### 표준 편차, 분산

- 표준 편차와 분산은 값들이 평균에서 얼마나 떨어져 있는지 나타내는 지표

```python
import numpy as np

array1 = np.array([14, 6, 13, 21, 23, 31, 9, 5])

print(array1.std()) # 표준 편차
print(array1.var()) # 분산
```

```python
8.496322733983215
72.1875
```
