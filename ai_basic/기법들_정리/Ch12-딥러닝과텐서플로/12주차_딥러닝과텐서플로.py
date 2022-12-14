# %% [markdown]
# # 딥러닝과 텐서플로
# ## 덧셈과 문자열 접합을 하고 난수 2개를 더하는 프로그램

# %%
# 연산자 오버로딩 예시
# +는 두 정수를 더하는 연산자
print(12+37)
# +는 두 문자열을 접합하는 연산자
print("python" + "is exciting")

# %%
# 라이브러리 불러오기
import random

# %%
# 정수 난수 생성

# [10,20] 사이의 난수를 생성하고 변수 a에 대입
a=random.randint(10,20)
# [10,20] 사이의 난수를 생성하고 변수 b에 대입
b=random.randint(10,20)

# %%
# 덧셈을 하고 결과를 출력

# a와 b를 더하여 변수 c에 대입
c=a+b
# 변수 a, b, c를 출력
print(a,b,c)


