t = "It's Not The Right Time To Conduct Exams. MY DEMAND IN BOLD AND CAPITAL. NO EXAMS IN COVID!!!"

count = 0

print('느낌표 개수 :', t.count('!'))

# t = list(t)
for ch in t:
    if ch.isupper():
        count += 1
print('대문자 개수 :', count)