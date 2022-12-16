import re

# 멀티라인 텍스트는 세 개의 따옴표를 사용하여 표현한다
text = '''101 COM PythonProgramming1
102 MAT LinearAlgebra
103 ENG ComputerEnglish'''

# Python Part1과 같이 숫자가 표함된 문자가 있을 경우
# 알파벳 문자나 줄바꿈문자(\n)이 아닌 순수하게 숫자로만 이루어진 단어를 추출하는 정규식
s = re.findall('[^a-zA-Z\\n]\d+', text)
print(s)

# 학수번호가 반드시 세자리 정수일 경우 다음과 같은 방법도 가능함
s = re.findall('\d{3}', text)
print(s)