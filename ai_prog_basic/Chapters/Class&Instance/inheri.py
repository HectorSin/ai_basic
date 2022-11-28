#!/usr/bin/env python
# coding: utf-8

# In[1]:
class Person():
    def __init__(self, name):
        self.name = name
    def getName(self):
        return self.name
    def isStudent(self):
        return False

class Student(Person):
    def __init__(self, name, gpa):
        super().__init__(name)
        self.gpa = gpa

    def isStudent(self):
        return True

# In[2]:
obj1 = Person("Kim")
print(obj1.getName(), obj1.isStudent())

obj2 = Student("Park", 4.3)
print(obj2.getName(), obj2.isStudent())
# %%
