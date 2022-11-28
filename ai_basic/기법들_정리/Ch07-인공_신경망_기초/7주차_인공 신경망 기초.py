#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


# LAB7-1


# In[3]:


# 두 입력에 곱해질 가중치 W와 편향 b를 준비하자.
W, b = np.array([0.5, 0.5]), -0.7


# In[4]:


# 가중치와 값들을 곱한 후 편향을 더한 값을 구해 그 값이 0보다 큰지 아니면 작은지 여부를 통해 출력값을 결정하는 함수 생성
def perceptron(x1, x2):
    x = np.array([x1, x2])
    tmp = np.sum(W*x)+b
    # 값이 0 이하면 -1 출력
    if tmp <= 0: return -1
    # 값이 0 초과이면 -1 출력
    else: return 1


# In[5]:


print('--- 퍼셉트론으로 구현한 AND 게이트 ---')
for xs in [(-1, -1), (-1, 1), (1, -1), (1,1)]:
    # 위의 데이터들을 활용하여 이전에 생성한 퍼셉트론 함수를 활용하여 결과갑 출력
    y = perceptron(xs[0], xs[1])
    print(xs, ': ', y)


# In[6]:


# OR 연산을 수행할 수 있도록 가중치와 편향을 조정
W, b = np.array([0.7, 0.7]), .5


# In[7]:


print('--- 퍼셉트론으로 구현한 OR 게이트 ---')
for xs in [(-1, -1), (-1, 1), (1, -1), (1,1)]:
    y = perceptron(xs[0], xs[1])
    print(xs, ': ', y)


# In[8]:


# LAB7-2


# In[9]:


# 이전과 반대로 가중치와 편향을 0으로 초기화
W,b = np.array([0, 0]), 0.0
# 합습률도 준비
learning_rate = 0.01
# 미리 가중치와 편향들을 입력하는 방식이 아닌 학습률만큼 학습시켜 가중치를 찾아가는 방식


# In[10]:


# 계단함수인 activation() 함수 생성
def activation(s):
    # 양수이면 1 출력
    if s > 0: return 1
    # 음수이면 -1 출력
    elif s < 0: return -1
    # 0이면 0 출력
    return 0


# In[11]:


def out(x):
    return activation (W.dot(x) + b)


# In[12]:


def train(x0, x1, target):
    global W,b
    X = np.array([x0, x1])
    y = out(X)
    
    # 예측이 맞으면 아무것도 하지 않음
    if target == y: return False
    # 가중치가 변경되지 않았음을 반환
    # 예측이 틀리면 학습 실시
    # 새로 학습 실시 전 기존 값들 출력
    print('가중치 수중전 target :{} y:{} b:{} W:{}'.format(target, y, b, W))
    
    # 입력 * 출력 비례하여 가중치 변경
    W = W + learning_rate * X * target
    # 편향: 입력이 1이라고 볼 수 있음
    b = b + learning_rate * 1 * target
    print('가중치 수정후 target :{} y:{} b:{} W:{}'.format(target, y, b, W))
    return True


# In[13]:


# 퍼셉트론의 현재 가중치에 따라 입력에 대한 예측을 수행하는 함수
def predict(inputs):
    outputs = []
    for x in inputs:
        outputs.append (out(x))
    return outputs


# In[14]:


adjusted = 0
for i in range(100):
    # 훈련 데이터 1
    adjusted += train(-1,-1,  1)
    # 훈련 데이터 2
    adjusted += train(-1, 1,  1)
    # 훈련 데이터 3
    adjusted += train( 1,-1,  1)
    # 훈련 데이터 4
    adjusted += train( 1, 1,  1)
    print("iteration ------------", i)
    # 모든 훈련에 대해 가중치 변화 없으면 학습종료
    if not adjusted: break
    adjusted = 0


# In[15]:


X = [[-1, -1], [-1, 1], [1, -1], [1,1]]
yhat = predict(X)
print('x0 x1 y')
for i in range(len(X)):
    print('{0:2d} {1:2d} {2:2d}'.format(X[i][0], X[i][1], yhat[i]))


# In[16]:


#LAB 7-3


# In[17]:


adjusted = 0
for i in range(100):
    # 훈련 데이터 1
    adjusted += train(-1,-1, -1)
    # 훈련 데이터 2
    adjusted += train(-1, 1, -1)
    # 훈련 데이터 3
    adjusted += train( 1,-1, -1)
    # 훈련 데이터 4
    adjusted += train( 1, 1,  1)
    print("iteration --------------", i)
    # 모든 훈련에 대해 가중치 변화 없으면 학습종료
    if not adjusted: break
    adjusted = 0


# In[18]:


X = [[-1, -1], [-1, 1], [1, -1], [1,1]]
yhat = predict(X)
print('x0 x1 y')
for i in range(len(X)):
    print('{0:2d} {1:2d} {2:2d}'.format(X[i][0], X[i][1], yhat[i]))


# In[19]:


adjusted = 0
for i in range(100):
    # 훈련 데이터 1
    adjusted += train(-1,-1,  1)
    # 훈련 데이터 2
    adjusted += train(-1, 1,  1)
    # 훈련 데이터 3
    adjusted += train( 1,-1,  1)
    # 훈련 데이터 4
    adjusted += train( 1, 1, -1)
    print("iteration --------------", i)
    # 모든 훈련에 대해 가중치 변화 없으면 학습종료
    if not adjusted: break
    adjusted = 0


# In[20]:


X = [[-1, -1], [-1, 1], [1, -1], [1,1]]
yhat = predict(X)
print('x0 x1 y')
for i in range(len(X)):
    print('{0:2d} {1:2d} {2:2d}'.format(X[i][0], X[i][1], yhat[i]))


# In[21]:


#LAB 7-4


# In[22]:


# 가중치 벡터빛 학습률 지정
W = np.array([0, 0, 0, 0])
learning_rate = 0.01


# In[23]:


# 계단 함수 생성 0초과면 1, 0이면 0, 0미만이면 -1 출력
def activation(s):
    if s > 0: return 1
    elif s < 0: return -1
    return 0


# In[24]:


def out(polyX) :
    return activation(W.dot(polyX))


# In[25]:


def train(x0, x1, target):
    global W
    polyX = np.array([x0, x1, x0*x1, 1])
    y = out(polyX)
    
    # 예측이 맞으면 아무것도 하지 않음
    if target ==y: return 0
    # 예측이 틀리면 학습 실시
    print('가중치 수정전 target :{} y:{} W:{}'.format(target, y, W))
    # 입력 * 목표값에 비례하여 변경
    W = W + learning_rate * polyX * target
    print('가중치 수정후 target :{} y:{} W:{}'.format(target, y, W))
    # 가중치가 변경되었음을 반환
    return 1


# In[26]:


def predict(inputs):
    outputs = []
    for x in inputs:
        polyX = np.array([x[0], x[1], x[0]*x[1], 1])
        outputs.append (out(polyX))
    return outputs


# In[27]:


adjusted = 0
for i in range(100):
    # 훈련 데이터 1
    adjusted += train(-1,-1, -1)
    # 훈련 데이터 2
    adjusted += train(-1, 1,  1)
    # 훈련 데이터 3
    adjusted += train( 1,-1,  1)
    # 훈련 데이터 4
    adjusted += train( 1, 1, -1)
    print("iteration ------------", i)
    # 모든 훈련에 대해 가중치 변화 없으면 학습종료
    if not adjusted: break
    adjusted = 0
X = [[-1, -1], [-1,1], [1,-1], [1,1]]
yhat = predict(X)
print('x0 x1 y')
for i in range(len(X)):
    print('{0:2d} {1:2d} {2:2d}'.format(X[i][0], X[i][1], yhat[i]))

