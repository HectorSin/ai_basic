#!/usr/bin/env python
# coding: utf-8

# # 2차 프로그래밍 과제
# ***
# > #### 과목: 인공지능프로그래밍언어기초
# > #### 학번: 201821498
# > #### 이름: 신재현
# > #### 학과: e-비즈니스학과
# ***

# ## 문제2
# ***
# * 다음 조건을 만족하는 프로그램을 작성하시오.
# <조건>
# 1. 파일(wlist.txt)로부터 줄 단위로 저장된 단어 읽기
# > - 단어는 영문자 소문자로만 구성되어야 함
# > - 정규식을 이용하여 대문자는 소문자로 변경, 특수문자[_, @, $, !]나 숫자는 제거
# > - 새 파일(newwlist.txt)에 줄 단위로 저장
# 
# 2. 새 파일에 저장된 단어들 중에 단어의 개수가 7이하인 단어의 수를 출력

# In[13]:


import re
import sys


# In[16]:


# 파일 불러읽기
with open("data\\wlist.txt", "r") as f:
    example = f.readlines()
    lines = []
    for line in example:
        ch_line = line.strip()
        ch_line = ch_line.lower()
        ch_line = re.sub('[0-9]+', '', ch_line)
        ch_line = re.sub('_', '', ch_line)
        ch_line = re.sub('@', '', ch_line)
        ch_line = re.sub('\$', '', ch_line)
        ch_line = re.sub('!', '', ch_line)
        lines.append(ch_line)

f = open('data\\newwlist.txt', 'w')

for line in lines:
    print(line, file= f)

f.close()

