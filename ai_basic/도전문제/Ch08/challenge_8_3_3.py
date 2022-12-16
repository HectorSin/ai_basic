#
# 따라하며 배우는 파이썬과 데이터과학(생능출판사 2020)
# LAB 8-3 파티 동시 참석자 알아내기, 210쪽
#
partyA = set(["Park", "Kim", "Lee"]) 
partyB = set(["Park", "Choi"])

print("파티 A에만 참석한 사람 : ", end="")
for x in partyA.difference(partyB):
    print(x, end=" ")