#!/usr/bin/env python
# coding: utf-8

# # 2차 프로그래밍 과제
# ***
# > #### 과목: 인공지능프로그래밍언어기초
# > #### 학번: 201821498
# > #### 이름: 신재현
# > #### 학과: e-비즈니스학과
# ***

# ## 문제 3
# <조건>
# * 다음의 데이터를 활용하여 아래 그래프와 동일하게 작성하라.
# > * 1번 데어터: x = 77, 78, 85, 83, 73, 77, 73, 80, y = 25, 28, 19, 30, 21, 22, 17, 35
# > * 2번 데이터: x = 75, 77, 86, 86, 79, 83, 83, 88, y = 56, 57, 50, 53, 60, 53, 49, 61
# > * 3번 데이터: x = 34, 38, 38, 41, 30, 37, 41, 35, y = 22, 25, 19, 30, 21, 24, 28, 18

# In[3]:


import matplotlib.pyplot as plt
import numpy as np


# In[4]:


# 데이터 만들기
xData_1 = np.array([77, 78, 85, 83, 73, 77, 73, 80])
yData_1 = np.array([25, 28, 19, 30, 21, 22, 17, 35])

xData_2 = np.array([75, 77, 86, 86, 79, 83, 83, 88])
yData_2 = np.array([56, 57, 50, 53, 60, 53, 49, 61])

xData_3 = np.array([34, 38, 38, 41, 30, 37, 41, 35])
yData_3 = np.array([22, 25, 19, 30, 21, 24, 28, 18])


# In[16]:


# 1번 데이터로 그래프 그리기
plt.scatter(xData_1, yData_1, color="red", marker='o')
plt.title('Dachshund size')
plt.xlabel('Length')
plt.ylabel('Height')

plt.show()

# 2번 데이터로 그래프 그리기
plt.scatter(xData_2, yData_2, color="blue", marker='s')
plt.title('Samoyad size')
plt.xlabel('Length')
plt.ylabel('Height')

plt.show()

# 3번 데이터로 그래프 그리기
plt.scatter(xData_3, yData_3, color="green", marker="^")
plt.title('Maltese size')
plt.xlabel('Length')
plt.ylabel('Height')

# 총합 데이터로 그래프 그리기
plt.scatter(xData_1, yData_1, color="red", marker='o', label='Dachshund')
plt.scatter(xData_2, yData_2, color="blue", marker='^', label='Samoyad')
plt.scatter(xData_3, yData_3, color="green", marker="s", label='Maltese')
plt.xlabel('Length')
plt.ylabel('Height')

plt.legend()
plt.show()

plt.show()

