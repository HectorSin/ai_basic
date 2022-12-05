#!/usr/bin/env python
# coding: utf-8

# # 객체 지향 프로그래밍과 객체
# - 파이썬의 리스트는 'lion', 'tiger',....등의 항목을 원소(속성)로 가질 수 있음
# - sort(), append(), remove(), reverse(), pop() 라는 함수를 가지고 있음
# * 객체 - 컴퓨터 시스템에서 다양한 기능을 수행하도록 속성과 메소드를 가진 요소를 객체라고 함
# - 현리한 메소드를 쉽게 이용할 수 있는 것이 객체 지향 프로그래밍의 큰 장점

# In[5]:


animals = ['lion', 'tiger', 'cat', 'dog']
animals.sort()
animals.append('rabbit')
animals.reverse()
animals


# In[6]:


s = animals.pop()
s


# In[7]:


s.upper()


# In[8]:


s.find('a')


# ### 예: str 클래스의 다양한 메소드
# * upper(), lower(), capitalize(), startswith().....
# 
# - 수많은 클래스는 객체를 생성하기 위한 틀(타입)이다.
# - 실제로 프로그램을 하기 위해서는 객체를 생성해야 하고 객체간의 상호작용이 필요하다.

# * type()은 객체의 자료형을 알려주는 함수
# * 객체들은 생성될 때 서로 다른 고유의 아이디 값을 가짐

# In[9]:


animals = ['lion', 'tiger', 'cat', 'dog']
type(animals)


# In[10]:


id(animals)


# In[12]:


s = 'tiger'
type(s)


# In[13]:


id(s)


# In[14]:


n = 200
type(n)


# In[16]:


n.__add__(100)


# In[17]:


n


# # 객체 지향 프로그래밍과 절차적 프로그래밍
# * 객체 지향 프로그래밍(OOP: object oriented programming)
# ** 프로그램을 짤 때, 프로그램을 실제 세상에 가깝게 모델링 하는 기법
# ** 컴퓨터가 수행하는 작업을 객체들 사이의 상호작용으로 표현
# 
# * 절차적 프로그래밍 언어
# ** 함수나 모듈을 만들어 두고 이것들을 문제해결 순서ㅔ 맞게 호출하여 수행하는 방식

# ### 절차적 프로그래밍 방식
# - 데이터들이 많아지고 함수가 많아진다면 매우 많은 화살표와 함수 호출이 필요
# - 대규모 프로젝트에서는 큰 어려움
# 
# ### 객체 지향 프로그래밍
# - 잘 설계된 클래스를 이용하여 객체를 생성
# - 클래스는 속성과 행위를 가지도록 설계하고 이 클래스를 이용하여 실제로 상호작용하는 객체를 만들어서 프로그램에 적용시키는 방법을 사용
# 
# ** 객체 지향 프로그래밍 방식이 유지보수 비용이 매우 적게 들기에 최근 프로그래밍 경향은 대부분 객체 지향 방식을 선호

# # 클래스와 객체, 인스턴스
# ### 클래스
#     * 프로그램 상에서 사용되는 속성과 행위를 모아놓은 집합체
#     * 객체에 설계도 혹은 템플릿, 청사진
# ### 인스턴스
#     * 클래스로부터 만들어지는 각각의 개별적인 객체
#     * 서로 다른 인스턴스는 서로 다른 속성 값을 가질 수 있음

# In[3]:


# 도전문제 1

# Dog 클래스와 객체를 생성하라
class Dog:
    def __init__(self, name, color = '흰색', size = '40'):
        self.name = name
        self.color = color
        self.size = size
    def bark(self):
        print("멍멍~~")
    def grow(self, num):
        self.size = self.size.__add__(num)
        print(self.size)
        
my_dog = Dog('james', 'blue', 100)

my_dog.bark()
my_dog.grow(20)


# In[5]:


# 도전문제 2.1

# 도전문제 1

# Dog 클래스와 객체를 생성하라
class Dog:
    def __init__(self, name, color = '흰색', size = '40'):
        self.name = name
        self.color = color
        self.size = size
    def bark(self):
        print("멍멍~~")
    def grow(self, num):
        self.size = self.size.__add__(num)
        print(self.size)
        
my_dog = Dog('james', 'blue', 100)
my_dog2 = Dog('Jindo')

my_dog.bark()
my_dog.grow(20)
my_dog2.bark()


# In[6]:


import math


# In[8]:


# 도전문제 2.2

# 원을 클래스로 표현하기 / 이름, 속성(반지름), 초기화 생성자, 필요 기능(넓이, 둘레 구하기)

class Circle:
    def __init__(self, radius):
        self.radius = radius
    def area(self):
        pi = math.pi
        print((self.radius ** 2) * pi)
    def around(self):
        pi = math.pi
        print((self.radius * 2) * pi)
        
circle = Circle(5)
circle.area()
circle.around()


# In[19]:


# 도전문제 3
class Dog:
    def __init__(self, name):
        self.__name = name
    def __str__(self):
        return '내 강아지의 정보 : {}'.format(self.__name)
    def set_name(self, name):
        self.__name = name

my_dog = Dog('Jindo')
my_dog.__name = 'haha'
print(my_dog)
my_dog.set_name("jajanga")
print(my_dog)


# In[23]:


# 도전문제 4.1
class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)
    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)
    def __mul__(self, other):
        return Vector2D(self.x * other.x, self.y * other.y)
    def __truediv__(self, other):
        return Vector2D(self.x / other.x, self.y / other.y)
    def __str__(self):
        return "({}, {})".format(self.x, self.y)

v1 = Vector2D(30, 40)
v2 = Vector2D(10, 20)
v3 = v1 * v2
print('v1 * v2 = ' + str(v3))
v4 = v1 / v2
print('v1 / v2 = ' + str(v4))


# In[25]:


# 도전문제 4.2

class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)
    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)
    def __mul__(self, other):
        return Vector2D(self.x * other.x, self.y * other.y)
    def __truediv__(self, other):
        return Vector2D(self.x / other.x, self.y / other.y)
    def __neg__(self):
        return Vector2D(-self.x, -self.y)
    def __str__(self):
        return "({}, {})".format(self.x, self.y)
    
v1 = Vector2D(10, 20)
print("-v1 = " + str(-v1))


# In[29]:


a = 'abc'
print("a id is " + str(id(a)))
b = 'abc'
print("b id is " + str(id(b)))
a = a + 'c'
print("a id is " + str(id(a)))
print("b id is " + str(id(b)))

print(id(a))

if a is b:
    print("yeah")
else:
    print("Noooo")


# In[4]:


# 도전문제 5
    
class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)
    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)
    def __mul__(self, other):
        return Vector2D(self.x * other.x, self.y * other.y)
    def __truediv__(self, other):
        return Vector2D(self.x / other.x, self.y / other.y)
    def __neg__(self):
        return Vector2D(-self.x, -self.y)
    def __lt__(self, other):
        return ((self.x ** 2) + (self.y ** 2)) < ((other.x ** 2) + (other.y ** 2))
    def __le__(self, other):
        return ((self.x ** 2) + (self.y ** 2)) <= ((other.x ** 2) + (other.y ** 2))
    def __ge__(self, other):
        return ((self.x ** 2) + (self.y ** 2)) >= ((other.x ** 2) + (other.y ** 2))
    def __gt__(self, other):
        return ((self.x ** 2) + (self.y ** 2)) > ((other.x ** 2) + (other.y ** 2))
    def __eq__(self, other):
        return ((self.x ** 2) + (self.y ** 2)) == ((other.x ** 2) + (other.y ** 2))
    def __ne__(self, other):
        return ((self.x ** 2) + (self.y ** 2)) != ((other.x ** 2) + (other.y ** 2))
    def __str__(self):
        return "({}, {})".format(self.x, self.y)
    
v1 = Vector2D(30, 40)
v2 = Vector2D(10, 20)
print("v1 > v2 = " + str(v1 > v2))
print("v1 >= v2 = " + str(v1 >= v2))
print("v1 < v2 = " + str(v1 < v2))
print("v1 <= v2 = " + str(v1 <= v2))

