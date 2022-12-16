#
# 따라하며 배우는 파이썬과 데이터과학(생능출판사 2020)
# LAB 7-4 오늘의 명언을 골라주는 기능을 만들자, 187쪽
#
import random 
 
express = [] 
express.append("1 + 2") 
express.append("341 - 154") 
express.append("514 * 516") 
express.append("516 / 4") 
express.append("5151 + 1541")
express.append("65411 - 65156") 
express.append("12345679 * 81") 
express.append("142857 * 7") 
express.append("10761 - 9999") 
express.append("1577 - 1577") 
 
print("############################") 
print("#    오늘의    수학문제    #") 
print("############################") 
print("") 
dailyQuiz = random.choice(express) 
print(dailyQuiz, '=', eval(dailyQuiz))