# 문제 1

class Box():
    def __init__(self, l, h, d):
        self.length = l
        self.height = h
        self.depth = d

    def __str__(self):
        return f"가로: {self.length}, 세로: {self.height}, 높이: {self.depth}"

    def setLength(self, l):
        self.length = l
    
    def setHeight(self, h):
        self.height = h
    
    def setDepth(self, d):
        self.depth = d

    def volume(self):
        self.volume = self.length * self.height * self.depth
        return self.volume

b1 = Box(100, 100, 100)

print(b1.volume())

b1.setDepth(4)
b1.setHeight(3)
b1.setLength(8)

print(b1)


# 문제 2

class Dog():
    dog_count = 0

    def __init__(self, name, age, color):
        self.name = name
        self.age = age
        self.color = color
        dog_count += 1

    def __str__(self):
        return f"이름: {self.name}, 나이: {self.age}, 색: {self.color}"

    def dog_num(self):
        print(f"지금까지 생성된 강아지의 수 = {Dog.dog_count}")

b1 = Dog("Molly", 10, "brown")
b2 = Dog("Daisy", 6, "black")
b3 = Dog("Bella", 7, "white")

print(b1.dog_count())