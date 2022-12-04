# -*- coding: utf-8 -*-
"""순환신경망2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UIQtEsuerjwHN09HUrdQUfz7C5OT4-ep

# 10.11 단순 RNN과 LSTM, GPU 모델의 비교 - 시퀀스 데이터 준비
***
"""

# 연속된 숫자 시퀀스 데이터와 레이블을 활용하여 순환 신경망의 모델이 잘 작동하는지 확인
import numpy as np

# 데이터를 생성하기 위한 sequence_gen() 함수 사용
# 0.0, 0.1, .. 증가하는 시퀀스 데이터를 생성
# seq_len 길이를 가지는 시퀀스 데이터를 size 갯수만큼 생성
def sequence_gen(size, seq_len):
  # 비어있는 넘파이 배열 생성
  seq_X = np.empty(shape=(size, seq_len, 1))
  Y = np.empty(shape=(size,))

  for i in range(size):
    # [0, 0.1, 0.2, .. ] 같은 시퀀스와 Y 값을 size 갯수만큼 생성
    c = np.linspace(i/10, (i+seq_len-1)/10, seq_len)
    # 새로운 축을 하나 더 추가
    seq_X[i] = c[:, np.newaxis]
    # 목표값 생성
    Y[i] = (i+seq_len) / 10
  
  return seq_X, Y

# 길이가 16인 시퀀스 8개를 훈련용으로 만든다
n, seq_len = 8, 16
train_seq_X, train_Y = sequence_gen(n, seq_len)

# 이전에 만든 훈련용 데이터를 flatten()함수를 활용하여 1줄씩 출력
print('훈련용 데이터')
for i in range(n):
  print(train_seq_X[i].flatten(), train_Y[i])

half_n, offset = int(n/2), 1.0
# 1.0만큼의 offset을 더해 테스트 셋 구성
test_seq_X = train_seq_X[:half_n] + offset
# 테스트 셋의 레이블 구성
test_Y = train_Y[:half_n] + offset

# 검증용 데이터도 비슷하게 출력
print('검증용 데이터')
for i in range(half_n):
  print(test_seq_X[i].flatten(), test_Y[i])

"""# 10.11 단순 RNN과 LSTM, GPU 모델의 비교 - 성능 비교
***
"""

# SimpleRNN의 모델 성능
import tensorflow as tf
# 유닛의 개수 256개
n_units = 256
simpleRNN_model = tf.keras.Sequential([
    # 레이어를 구성하는 유닛개수 256개 지정
    tf.keras.layers.SimpleRNN(units = n_units, return_sequences=False,
                              input_shape=[seq_len, 1]),
    tf.keras.layers.Dense(1)
])

simpleRNN_model.compile(optimizer = 'adam', loss = 'mse')
# 100에폭으로 학습 진행
simpleRNN_model.fit(train_seq_X, train_Y, epochs = 100)

# 정답을 잘 예측하는지 확인
result = simpleRNN_model.predict(test_seq_X)
result = result.flatten()
print('예측값 :', result)
print('실제값 :', test_Y)

# LSTM모델 성능
LSTM_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units = n_units, return_sequences=False,
                         input_shape=[seq_len, 1]),
    tf.keras.layers.Dense(1)
])

LSTM_model.compile(optimizer = 'adam', loss = 'mse')
LSTM_model.fit(train_seq_X, train_Y, epochs=100)

reulst = LSTM_model.predict(test_seq_X)
result = result.flatten()
print('예측값 :', result)
print('실제값', test_Y)

# GRU모델 성능
GRU_model = tf.keras.Sequential([
    tf.keras.layers.GRU(units = n_units, return_sequences=False,
                        input_shape=[seq_len, 1]),
    tf.keras.layers.Dense(1)
])

GRU_model.compile(optimizer = 'adam', loss = 'mse')
GRU_model.fit(train_seq_X, train_Y, epochs=100)

result = GRU_model.predict(test_seq_X)
result = result.flatten()
print('예측값 :', result)
print('실제값 :', test_Y)

"""# LAB 10-2 기억이 필요한 시퀀스 예측
***
## 실습 목표
> - 사인 곡선에서 일부분을 잘라 만든 시퀀스의 각 요소 각각에 임의의 난수 인덱스를 부여하자. 이번에는 이 시퀀스의 다음 값을 예측하는 것이 아니라, 시퀀스의 각 요소들 가운데 짝수 인덱스를 가진 요소들의 평균 값을 계산하는 모델을 만들어 보자.

***

"""

import numpy as np
import matplotlib.pyplot as plt

# 시퀀스의 개수 200개 시퀀스의 길이 30으로 설정
size, seq_len = 200, 30
# 비어있는 넘파이 배열 생성
# 각 시퀀스에 인덱스가 존재
seq_X = np.empty(shape=(size, seq_len, 2))
# 각 시퀀스의 정답을 담은 변수생성
Y = np.empty(shape=(size,))

# sine 곡선에서 잘라낼 구간 설정
interval = np.linspace(0.0, 2.5, seq_len+1)

shift = np.random.randn(size)
# 시퀀스 내의 각 원소에 대해 인덱스와 값을 설정
for i in range(size):
  # 인덱스
  seq_X[i,:,0] = np.random.randint(0, 6, size=(seq_len))
  # 값
  seq_X[i,:,1] = np.sin(shift[i] + interval[:-1])
  # 정답 레이블은 시퀀스 내에서 짝수 인덱스를 가진 원소의 값을 모두 더한 값
  even_idx = seq_X[i, seq_X[i,:,0]%2 == 0]
  Y[i] = even_idx[:,1].sum()

for i in [1, 3, 5, 9]:
  # 인덱스 정보
  plt.scatter(interval[:-1], seq_X[i, :, 0], color='k')
  # 값: 사인 시퀀스 파란색 선으로 나타난 점으로 표시
  plt.scatter(interval[:-1], seq_X[i, :, 1], color='b')
  # 레이블 붉은 점으로 표시
  plt.scatter(interval[-1], Y[i], color='r')
  plt.show()

# 훈련용과 테스트용으로 나누기
train_X = seq_X[:180]
train_y = Y[:180]
test_X = seq_X[180:]
test_y = Y[180:]

import tensorflow as tf
simpleRNN_model = tf.keras.Sequential([
    # RNN 유닛의 수를 앞의 예제와 같이 10개로 설정, 20개의 연결
    tf.keras.layers.SimpleRNN(units = 10, return_sequences=False,
                              input_shape = [seq_len, 2]),
    tf.keras.layers.Dense(1)
])

simpleRNN_model.compile(optimizer = 'adam', loss = 'mse')
simpleRNN_model.summary()

history = simpleRNN_model.fit(train_X, train_y, epochs=150)
plt.plot(history.history['loss'])

# 훈련 데이터에 대한 예측 결과와 실제 정답을 비교
train_y_hat = simpleRNN_model.predict(train_X)
plt.scatter(train_y, train_y_hat)
plt.show()
test_y_hat = simpleRNN_model.predict(test_X)
plt.scatter(test_y, test_y_hat)
plt.show()

# 훈련용 데이터의 정답과 예측 값, 검증용 데이터의 정답과 예측값을 데이터에 들어있는 순서대고 그리기
# 정답값 검정색
plt.plot(train_y, c='k', linewidth=2)
# 예측값 빨간색
plt.plot(train_y_hat, c='r', linewidth=1)
plt.show()
# 정답값 검정색
plt.plot(test_y, c='k', linewidth=2)
# 예측값 빨간색
plt.plot(test_y_hat, c='r', linewidth=1)

LSTM_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units = 10, return_sequences=False,
                         input_shape=[seq_len, 2]),
    tf.keras.layers.Dense(1)
])

LSTM_model.compile(optimizer = 'adam', loss = 'mse')
LSTM_model.summary()

# 훈련과 같은 방식으로 에폭의 수 150개로 지정
history = LSTM_model.fit(train_X, train_y, epochs=150)
plt.plot(history.history['loss'])

# LSTM 모델이 훈련 데이터와 검증용 데이터에 대해 얼마나 예측을 잘 하는지 같은 방식으로 살펴보기
train_y_hat = LSTM_model.predict(train_X)
plt.scatter(train_y, train_y_hat)
plt.show()
test_y_hat = LSTM_model.predict(test_X)
plt.scatter(test_y, test_y_hat)
plt.show()

# 훈련 데이터의 정답과 예측값 검증용 데이터의 정답과 예측값을 데이터에 들어있는 순서대로 그려서 확인
plt.plot(train_y, c='k', linewidth=2)
plt.plot(train_y_hat, c='r', linewidth=1)
plt.show()
plt.plot(test_y, c='k', linewidth=2)
plt.plot(test_y_hat, c='r', linewidth=1)

# GRU 모델을 만들기
GRU_model = tf.keras.Sequential([
    # 텐서플로우 모델 업데이트로 입력값을 조정하였습니다.
    tf.keras.layers.GRU(units = 10, return_sequences=False,
                        input_shape=[seq_len, 2]),
    tf.keras.layers.Dense(1)
])

GRU_model.compile(optimizer = 'adam', loss = 'mse')
GRU_model.fit(train_X, train_y, epochs=150)
train_y_hat = LSTM_model.predict(train_X)
plt.scatter(train_y, train_y_hat)
plt.show()
test_y_hat = LSTM_model.predict(test_X)
plt.scatter(test_y, test_y_hat)
plt.show()
plt.plot(train_y, c='k', linewidth=2)
plt.plot(train_y_hat, c='r', linewidth=1)
plt.show()
plt.plot(test_y, c='k', linewidth=2)
plt.plot(test_y_hat, c='r', linewidth=1)