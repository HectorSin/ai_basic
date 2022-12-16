import re

# 멀티라인 텍스트는 세 개의 따옴표를 사용하여 표현한다
text = '''101 COM PythonProgramming1
102 MAT LinearAlgebra
103 ENG ComputerEnglish'''

# 3개의 대문자로 이루어진 단어를 추출하는 정규식
s = re.findall('[A-Z][A-Z][A-Z]', text)
print(s)

# 3개의 대문자로 이루어진 단어를 추출하는 정규식
s = re.findall('[A-Z]{3}', text)
print(s)