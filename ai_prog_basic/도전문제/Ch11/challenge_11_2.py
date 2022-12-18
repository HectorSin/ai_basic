import matplotlib.pyplot as plt

x = [x / 10 for x in range(20)]          # 0에서 20까지의 정수를 생성
y = [(x / 10)**2 for x in range(20)]       # 0에서 20까지의 정수 x에 대해 x 제곱값을 생성
z = [(x / 10)**3 for x in range(20)]       # 0에서 20까지의 정수 x에 대해 x 세제곱값을 생성
i = [2**(x / 10) for x in range(20)]       # 0에서 20까지의 정수 x에 대해 2의 x제곱값을 생성

plt.plot(x, x, label='linear')      # 각 선에 대한 레이블
plt.plot(x, y, label='quadratic')
plt.plot(x, z, label='qubic')
plt.plot(x, i, label='power')

plt.xlabel('x label')               # x 축의 레이블
plt.ylabel('y label')               # y 축의 레이블
plt.title("My Plot")
plt.legend()                        # 디폴트 위치에 범례를 표시한다
plt.show()