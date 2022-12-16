#
# 6.7 여러 개의 값을 넘겨주고 여러 개의 값을 돌려받자
#
def get_square(a, b, c) :
    return a ** 2, b ** 2, c ** 2

a, b, c = 1, 2, 3
a_sq, b_sq, c_sq = get_square(a, b, c) 
print(a, '제곱 :', a_sq, ', ', b,'제곱 :', b_sq,', ',c,'제곱 :',c_sq)