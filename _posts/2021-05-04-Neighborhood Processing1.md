---
layout: post
title: "Neighborhood Processing 1"
date: 2021-05-04
excerpt: "mask, spatial filtering, spatial convolution, frequencies"
tags: [matlab, 영상처리]
feature: http://i.imgur.com/Ds6S7lJ.png
comments: true
---

# Neighborhood processing 1
---

### 1. 서론

- 영역 단위 처리 방식: 주어진 화소의 이웃 화소들에 대해서도 함수를 적용
- 영상 처리에는 여러 개의 입력 화소가 사용되고 결과값으로는 하나의 화소값이 나옴
- mask를 주어진 영상 위로 이동하면서 처리

![Neighborhood processing](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled.png)

- Mask: 직사각형 또는 다른 형태
    - 마스크의 크기는 임의로 정할 수 있지만 일반적으로 홀수 크기를 사용. 짝수 길이는 중심을 정하기 힘들기 때문
    - 새로운 영상의 화소 값은 마스크 내의 그레이 값들로부터 계산됨
- Filter: 마스크와 함수를 결합한 것
    - 선형 필터: 새로운 그레이 값을 계산하는 함수가 마스크 내에 있는 모든 그레이 값들의 선형함수이면 그 필터를 선형 필터라 함

**공간 필터링(Spatial filtering)**

- 입력: 마스크, 입력 영상
- 결과: 출력 영상
- 마스크와 입력 영상의 연산을 수행하고 출력 영상(마스크가 덮혀있던 가운데 부분)에 저장한 후 마스크를 이동시킴

**공간 필터링 단계**

![performing spatial filtering](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%201.png)

1. 마스크를 현재 화소 위에 위치시킴
2. 필터의 값과 이웃 화소들의 값을 대용하는 원소끼리 서로 곱함
3. 곱의 항들을 모두 더함

**Spatial convolution**

- 모든 과정은 공간 필터링과 동일
- 유일한 차이점:  곱하고 더하기 전에 필터를 180도 회전시킴
- Spatial convolution 식
    - 위의 식은 필터를 회전, 아래 식은 영상을 회전
    - 오타: 아래 수식에서 p(i-s, j-t)가 올바른 수식

![수식](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%202.png)

- 대부분의 필터 마스크는 회전에 대해서 대칭이기 때문에 공간 필터링과 spatial convolution은 같은 출력을 얻음

**선형 필터 예시: 평균 필터(averaging)**

- 3X3 마스크를 사용. 마스크 내의 총 9개의 값들의 평균을 구하는 것
- 평균값은 새로운 영상에서 현재 화소의 그레이 값이 됨
    - e는 원 영상에서 현재 화소의 그레이 값이고, 그 평균값은 새로운 영상에서 현재 화소의 그레이 값이 됨

![averaging filter](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%203.png)

- 5X5 크기의 영상에서
    - 3X3 마스크를 완전히 중첩할 수 있는 화소들은 총 9개. 따라서 출력은 9개의 값들로 구성

    ![예시](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%204.png)

    - 빨간색 부분에 대한 평균값

    ![예시](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%205.png)

    - 영상 x를 3X3 평균 필터로 필터링한 결과

    ![결과](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%206.png)

    - 평균 필터링을 사용하게 되면 영상이 부드러워짐. 원본 영상에 비해 주변 값들과의 차이가 크지 않음
    - 영상이 부드러워져 초점이 맞춰지지 않은 것처럼 보이며 윤곽선이 흐릿하게 보임
    - 평균 필터링은 저주파 통과 필터의 한 예

### 2. 표기법

- 선형필터를 마스크 내에 있는 모든 화소들의 그레이 값, 즉 계수로 간단히 표현하는 것이 편리. 그러면 이것을 매트릭스로 표현 가능
- 위에서 언급한 평균 필터의 출력 식

![표기법](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%207.png)
- 평균 필터에서는 계수의 값을 모두 더하면 1이 됨


**예제**
- 필터가 아래처럼 주어졌을 때

![example_filter](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%208.png)
    
- 주어진 영상의 화소 e에 대한 그레이 레벨 값은 아래와 같이 계산됨

![gray_level](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%209.png)

- 결과값은 e 위치에 저장
- 가운데 위치에 대해 상하좌우는 차이값을 계산하고 대각선에 대해서는 그대로 값을 사용 ⇒ 전형적인 고주파 통과 필터
- 고주파 필터의 마스크의 계수값을 모두 더해보면 0이 나옴
- 이웃한 화소들의 크기값 차이가 크지 않으면 저주파 영상, 크면 고주파 영상

**영상의 에지 처리(Edges of the Image)**

![edges of the image](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%2010.png)

- 마스크의 일부분이 영상의 바깥 부분에 있을 경우, 영상의 에지 부분에서 어떻게 처리해야 할까?
- 그림과 같은 경우 필터 함수에서 사용할 영상의 화소 값이 없음

**영상의 에지 처리 방법**
1. Ignore the edges(에지의 무시)
    - 마스크가 영상에 완전히 포개지는 화소들에 대해서만 마스크를 적용
    - 에지를 제외한 모든 화소에 대해서만 마스크를 적용하기 때문에 출력 영상은 원 영상보다 작아짐
    - 단점: 마스크의 크기가 매우 큰 경우 상당한 양의 정보를 잃게 될 수 있음

    ![Ignore the edges](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%2011.png)

2.  Pad with zeros(0으로 채움)
    - 영상의 외부에 있는 영역에서 필요한 모든 값들이 0이라고 가정
    - 영상의 모든 화소에 대해 처리할 수 있기 때문에 원 영상의 크기와 같은 출력 영상을 얻을 수 있음
    - 단점: 영상의 주위부분에 원하지 않는 결과가 나타나는 효과가 발생할 수 있음

    ![Pad with zeros](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%2012.png)

3. Mirroring(미러링)
    - 영상의 오부에 있는 영역에서 필요한 모든 값들은 해당 에지에 대해서 미러링을 얻음
    - 영상의 모든 화소에서 처리할 수 있어서 원 영상의 크기와 같은 출력 영상을 얻을 수 있음
    - 바운더리를 미러링으로 수행해 임의로 생성해 놓는 것은 합리적. 주변에 있는 것들은 비슷할 확률이 높기 때문
    - 장점: Pad with zeros에서 주위 부분에 원하지 않는 결과가 나타나는 효과도 방지할 수 있음
    
    ![Mirroring](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%2013.png)

### 3. Mathlab에서 필터링

- `filter2` 함수를 사용하면 선형 필터링 작업을 할 수 있음

```matlab
filter2(filter, image, shape)
```
- 함수의 결과는 `double` 타입의 행렬
- `shape` 매개변수는 옵션 사항이며, 에지를 처리하는 방법을 지정
    1. `filter2(filter, image, 'same')` : default
        - 출력 영상은 원 영상의 크기와 동일한 크기의 매트릭스
        - 에지 처리는 pad with zeros

        ![same](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%2014.png)

    2. `filter2(filter, image, 'valid')`
        - 마스크를 영상 내부에 완전치 중첩되게 하는 화소만 처리. 출력은 항상 원 영상보다 작은 크기
        - 에지 처리는 ignore the edges
        - `'same'`으로 처리한 결과는 원 영상의 외부에 미리 0으로 채운 후에 `'valid'`로 처리하면 얻을 수 있음

        ![valid](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%2015.png)

    3. `filter2(filter, image, 'full')`
        - 영상의 화소 값이 없는 경우에는 0으로 채우고, 마스크와 영상 행렬이 서로 겹치는 부분이 있으면 영상과 영상 주위의 모든 곳에서 필터를 적용해 출력 매트릭스를 만듦
        - 출력은 원 영상보다 더 큰 크기

        ![full](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%2016.png)

    4. Mirroring 옵션
        - `filter2` 함수는 미러링 옵션을 제공하지 않음
        - `filter2(filter, image, 'full')` 를 수행하기 전 아래 명령을 사용하면 미러링 기법을 구현할 수 있음

        ![mirroring](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%2017.png)

        - 매트릭스 `x`를 `m_x`로 확장하며, `wr`/`wc` 는 마스크 크기의 행/열 길이의 절반인 값을 가짐(소수점 이하는 버림)
        - 예를 들어, 마스크 `a`가 3X3 매트릭스이면 `wr`와 `wc`의 값은 1
        - 미러링 방법으로 필터링한 결과

        ![mirroring](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%2018.png)


**`fspecial` 함수**

```matlab
h = fspecial(type, parameters)
```
- 사용할 필터를 만들 수 있음
- 다양한 종류의 필터를 쉽게 만들 수 있도록 많은 옵션을 가지고 있음
- `parameters`에 숫자나 벡터를 지정하지 않으면 디폴트로 3X3 크기의 필터를 만듦
- average 옵션 파라미터를 사용하면, 지정된 크기의 평균 필터가 만들어짐

```matlab
% 5X7 크기의 평균 필터를 만듦
fspecial('average', [5, 7])

% 11x11 크기의 평균 필터를 만듦
fspecial('average', 11)
```

**예시**: 영상에 3X3 평균 필터를 적용

![example](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%2019.png)

- 결과: `double` 타입의 매트릭스
- 이 매트릭스를 디스플레이하기 위해서는 3가지 방법이 있음
    1. `double` 타입 매트릭스를 `uint8` 타입의 매트릭스로 변환 후 `imshow`
    2. 매트릭스 값을 255로 나누어서 0.1~1.0 범위의 값을 가진 매트릭스로 만든 후 `imshow` 사용
    3. 디스플레이를 하기 위해 결과 매트릭스를 `mat2gray` 함수를 사용해 스케일링
- 1, 2번을 사용한 결과 영상을 보여줌

![result](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%2020.png)

![result](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%2021.png)

![result](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%2022.png)

**평균 필터를 사용한 결과**

- 평균 필터는 영상을 흐리게 만듦. 에지들은 원 영상보다 더 분명하지 않음
- 에지에서 pad with zeros 방법을 사용했기 때문에 영상의 가장자리에 우두운 부분이 나타남. 크기가 큰 필터를 사용할 때 현저하게 나타남
    - 필터링을 해서 이런 것을 원치 않는다면, `'valid'` 옵션을 사용하는 것이 좋을 수 있음
- 이런 필터링의 결과 영상은 원 영상보다 나쁘게 보일 수 있지만, 블러링 필터를 사용해 영상에서 상세한 부분을 제거하는 것도 좋은 연산이 될 수 있음
- 미러링을 활용하면 어두운 부분은 나타나지 않음

### 4. 주파수: 저주파 통과와 고주파 통과 필터

- 주파수: 거리에 따라 그레이 값이 변화하는 양을 측정한 것
- 고주파 성분: 짧은 거리 내에서 그레이 값의 변화가 매우 큰 특징을 가짐. ex) 에지와 잡음
- 저주파 성분: 영상에서 그레이 값이 거의 변화하지 않는 특성을 가진 부분. ex) 배경 부분, 피부의 질감
- 고주파 통과 필터: 고주파 성분들을 통과시키고, 저주파 성분들을 줄이거나 제거하는 필터
- 저주파 통과 필터: 저주파 성분들을 통과시키고, 고주파 성분들을 줄이거나 제거

**예시:** 3X3 평균 필터

- 영상의 에지를 흐리게 하므로 저주파 통과 필터

**예시:** 고주파 통과 필터

![high-filter](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%2023.png)

- 고주파 통과 필터의 계수들의 총합은 0
- 영상에서 그레이 값들이 유사한 저주파 부분에서 이 필터를 사용하면 새로운 영상에서 대응되는 그레이 값들이 0에 가깝게 된다는 것을 의미
- 고주파 통과 필터는 에지 검출과 에지 강조를 할 때 많이 사용

**예시:** cameraman 영상을 사용해 에지 검출

![edge](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%2024.png)

- `'laplacian'`: 고주파 통과 필터. 영상의 엣지 부분만 출력됨
- `'log'`: laplacian of Gaussian. 가우시안 필터는 저주파 통과 필터
    - `'log'`는 가우시안 필터를 통과한 후 `'laplacian'` 필터를 통과
    - 즉, 저주파 통과 필터 통과 후 고주파 통과 필터를 통과
    - 마지막 특성은 고주파 성분을 출여내는 것
    - `'log'`를 수행하면 디테일한 정보를 제거한 후 더 적은 엣지만 남음
    - 엣지를 추출하고 싶은 영상에서 잡음이 많을 때 저주파 통과 필터를 사용해 잡음을 제거하고 고주파 통과 필터를 통과시켜 원래 영상의 진짜 엣지 부분만 출력할 때 사용

    ![result](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%2025.png)

**0~255 범위 밖의 값 처리**

- 영상으로 디스플레이 하기 위해서는 화소들의 그레이 값이 0~255 사이에 있어야 함.
- 선형 필터를 적용하면 그 결과의 값은 이 범위를 벗어날 수 있음
- 디스플레이를 할 수 있는 범위 밖의 값을 처리하는 방법
    1. 음수를 양수로 만들기
        - 음수를 양수로 만드는 것은 음수 문제를 해결할 수 있지만, 255보다 큰 값은 해결할 수 없음
        - 아주 특수한 경우(ex. 음수가 몇 개만 있고 이 값들이 0에 가까운 경우)에만 사용
    2. clip values(값의 제한)
        - 디스플레이할 수 있는 $y$ 값을 구하기 위해 필터로 처리해 생성된 그레이값 $x$에 대해 아래의 식과 같이 연산을 수행

        ![clip values](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%2026.png)

        - 모든 화소 값들이 원하는 범위 내에 있도록 하지만, 많은 화소들이 0~255를 벗어나는 경우, 특히 그레이 값들이 넓은 범위에 골고루 퍼져있는 경우 적합하지 않음
        - 이런 경우 위 연산을 수행하면 필터링 결과의 값들이 파괴될 수 있음
    3. scailing transformation(스케일링 변환)
        - 필터링에 의해 생성된 가장 작은 그레이 값을 $g_L$이라 하고, 가장 큰 값을 $g_H$라고 가정
        - 선형 변환으로 $g_L~$~ $g_H$ 범위 내의 모든 값들을 0~255로 변환할 수 있음

        ![scaling transformation](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%2027.png)

        - 필터링으로 생성된 모든 그레이 값 x에 대해서 이 선형 변환을 적용하면 디스플레이할 수 있는 결과(필요한 경우 rounding 처리 후)를 얻을 수 있음

**예시:** 고주파 통과 필터를 cameraman 영상에 적용하는 과정

![1](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%2028.png)

- 매트릭스 cf2의 최댓값, 최솟값은 593, -541

![2](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%2029.png)

- `mat2gray` 함수: 매트릭스 원소들을 디스플레이 가능한 값으로 자동으로 스케일링
- 어떤 임의의 매트릭스 $M$에 대해, `mat2gray` 함수는 매트릭스 $M$ 원소의 작은 값을 0.0으로, 가장 큰 값을 1.0으로 션형 변환을 수행
- `mat2gray` 함수의 출력은 항상 `double` 타입이라는 것을 의미
- `mat2gray` 함수의 입력도 `double` 타입이어야 함

![3](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%2030.png)

- 위에서 설명한 선형 변환을 직접 구현
- 결과는 double 타입의 매트릭스. 원소들의 범위는 0.0~1.0
- imshow 명령으로 결과 매트릭스를 볼 수 있음
- 영상에 255를 먼저 곱한 후 uint8 타입의 영상으로도 만들 수 있음

![4](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%2031.png)

- 디스플레이 하기 전, 필터링 결과를 상수 값으로 나누면 더 좋은 결과를 얻을 수 있음

![5](https://github.com/slchoi01/slchoi01.github.io/blob/master/image/matlab/neighborhood/Untitled%2032.png)

- (a): mat2gray 사용한 결과
- (b): 상수로 나눈 후의 결과
