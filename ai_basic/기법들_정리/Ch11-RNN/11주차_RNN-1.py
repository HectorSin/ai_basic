#!/usr/bin/env python
# coding: utf-8

# # 10.4 텐서플로우를 이용하여 단순 RNN 모델 만들기
# ***

# In[1]:


import numpy as np

size, seq_len = 100, 3

# 비어있는 넘파이 배열 생성
X = np.empty(shape=(size, seq_len, 1))
Y = np.empty(shape=(size,))

for i in range(size):
    # [0, 0.1, 0.2], [0.1, 0.2, 0.3] 같은 시퀀스 생성
    c = np.linspace(i/10., (i+seq_len-1)/10., seq_len)
    # 새로운 축 한개 추가
    X[i] = c[:, np.newaxis]
    # 목표값 생성
    Y[i] = (i+seq_len) / 10
    
for i in range(len(X)):
    print(X[i], Y[i])


# In[2]:


get_ipython().system('pip install tensorflow')
get_ipython().system('pip install numpy --upgrade')


# In[3]:


import tensorflow as tf
# units는 SimpleRNN  레이어에 있는 뉴런의 수
# return_sequences는 출력으로 시퀀스 전체를 출력할지 묻는 옵션
# input_shape [3, 1]에서 3는 timesteps, 1은 입력차원
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(units = 20, return_sequences=False,
                             input_shape=[3, 1]),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer = 'adam', loss = 'mse')
model.summary()


# # 10.5 RNN을 학습시켜 예측을 해 보자.
# ***

# In[4]:


import matplotlib.pyplot as plt
history = model.fit(X, Y, epochs=300)
plt.plot(history.history['loss'])
plt.show()
y_hat = model.predict(X)
plt.scatter(Y, y_hat)
plt.show()


# In[5]:


print(model.predict(np.array([[[10.2], [10.3], [10.4]]])))
print(model.predict(np.array([[[10.4], [10.5], [10.6]]])))


# In[6]:


model1256 = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(units = 256, return_sequences = False,
                             input_shape = [3, 1]),
    tf.keras.layers.Dense(1)
])

model1256.compile(optimizer = 'adam', loss = 'mse')
model1256.summary()
history = model1256.fit(X, Y, epochs=300)

plt.plot(history.history['loss'])

y_hat = model1256.predict(X)
plt.scatter(Y, y_hat)


# In[7]:


print(model1256.predict(np.array([[[10.2], [10.3], [10.4]]])))
print(model1256.predict(np.array([[[10.4], [10.5], [10.6]]])))


# # 10.6 RNN을 다층구조로 만들어 적은 수의 파라미터로 좋은 성능을 내자
# ***

# In[8]:


model_multilayer = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(units = 34, input_shape=[3,1],
                             return_sequences=True),
    tf.keras.layers.SimpleRNN(units = 34, return_sequences=True),
    tf.keras.layers.SimpleRNN(units = 34, return_sequences=True),
    tf.keras.layers.SimpleRNN(units = 34),
    tf.keras.layers.Dense(1) 
])

model_multilayer.summary()


# In[9]:


model_multilayer.compile(optimizer = 'adam', loss = 'mse')
history = model_multilayer.fit(X, Y, epochs=300)

plt.plot(history.history['loss'])

y_hat = model_multilayer.predict(X)
plt.scatter(Y, y_hat)


# In[10]:


print(model_multilayer.predict(np.array([[[10.2], [10.3], [10.4]]])))
print(model_multilayer.predict(np.array([[[10.4], [10.5], [10.6]]])))

