#
# LAB 6-7 피보나치 함수 계산하기
#
def fibonacci(n): 
    if n < 2:
        return n

    a, b = 0, 1
    for _ in range(n-1):
        a, b = b, a + b

    return b

i = int(input("몇 번째 항: ")) 
print(fibonacci(i-1))  