#!/usr/bin/env python
# coding: utf-8

# # 2차 프로그래밍 과제
# ***
# > #### 과목: 인공지능프로그래밍언어기초
# > #### 학번: 201821498
# > #### 이름: 신재현
# > #### 학과: e-비즈니스학과
# ***

# ## 문제 1
# ***
# * 다음 조건을 만족하는 넌센스 퀴즈 프로그램을 작성하시오.
# <조건>
# 1. 넌센스 퀴즈 3개를 랜덤하게 출제
# 2. 딕셔너리 자료구조 사용 (key: 문제번호)
# 3. 문제당 1점 처리하여 최종 점수를 출력
# 4. 총 2번 실행한 결과 제출

# ***
# 
# 백가지 과일이 죽기 직전을 다른 말로?
# 
# 정답(또는 quit): 백과사전
# 
# 정답
# 
# ***
# 
# 깨뜨리고 칭찬 받는 것은?
# 
# 정답(또는 quit): 접시
# 
# 오답
# 
# ***
# 
# 못 사온다고 해놓고 사온 것은?
# 
# 정답(또는 quit): 못
# 
# 정답
# 
# ***
# 
# 최종 점수는 2

# In[2]:


# 랜덤함수
import random


# In[15]:


# 딕셔너리로 넌센스 퀴즈 생성
prob1 = {"key": 0, "prob": "백가지 과일이 죽기 직전을 다른 말로?", "ans":"백과사전"}

prob2 = {"key": 1, "prob": "깨뜨리고 칭찬 받는 것은?", "ans":"신기록"}

prob3 = {"key": 2, "prob": "못 사온다고 해놓고 사온 것은?", "ans":"못"}

prob4 = {"key": 3, "prob": "병아리가 제일 잘 먹는 약은?", "ans":"삐약"}

prob5 = {"key":4, "prob":"개 중에 가장 아름다운 개는?", "ans":"무지개"}

prob6 = {"key":5, "prob":"걱정이 많은 사람이 오르는 산은?", "ans":"태산"}

prob7 = {"key":6, "prob":"다리 중 아무도 보지 못한 다리는?", "ans":"헛다리"}

problem = [prob1, prob2, prob3, prob4, prob5, prob6, prob7]


# In[21]:


# 넌센스 문제 출력 함수
# (문제번호)

def nonsense():
    global score
    key = random.randrange(0,7)
    prob = problem[key]
    print(prob["prob"])
    ans = input("정답(또는 quit): ")
    if ans == "quit":
        print("quit")
    else:
        if ans == prob["ans"]:
            print("정답")
            score += 1
        else:
            print("오답, 정답은 {}입니다.".format(prob["ans"]))


# In[23]:


score = 0

for _ in range(3):
    nonsense()

print(f"최종점수는: {score}")


# In[ ]:




