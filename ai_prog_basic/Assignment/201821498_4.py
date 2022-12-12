#!/usr/bin/env python
# coding: utf-8

# # 2차 프로그래밍 과제
# ***
# > #### 과목: 인공지능프로그래밍언어기초
# > #### 학번: 201821498
# > #### 이름: 신재현
# > #### 학과: e-비즈니스학과
# ***

# ## 문제 4
# * 학생 관리 시스템을 클래스로 정의하고 아래 조건을 만족하는 프로그램을 작성하시오.
# <조건>
# * 클래스에 필요한 속성고 메소드는 다음과 같다.

# In[13]:


class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
# Person 클래스를 상속받음
class Student(Person):
    def __init__(self, name, age, major):
        super().__init__(name,age)
        self.major = major

class StudentManageSystem(Student):
    def __init__(self):
        # studentList에 학생들의 정보 저장
        self.studentList = []
    # 학생 등록
    def addStudent(self, name, age, major):
        self.studentList.append([name, age, major])

    # 학생 삭제
    def removeStudent(self, name):
        if name in self.studentList[:][0]:
            removeList = []
            for list in self.studentList:
                if name not in list:
                    removeList.append(list)

            self.studentList = removeList
        
        else:
            print("그런 이름의 학생은 없습니다.")
    # 전체 학생 정보 출력
    def printAllStuInfo(self):
        print(self.studentList)


# In[23]:


func = StudentManageSystem()


# In[24]:


func.addStudent("Jack", 23, "business")
func.addStudent("Bumjo", 25, "e-business")
func.addStudent("Minsu", 22, "AI")


# In[25]:


func.printAllStuInfo()


# In[26]:


func.removeStudent("Jack")


# In[27]:


func.printAllStuInfo()


# In[28]:


func.removeStudent("Jack")


# In[ ]:




