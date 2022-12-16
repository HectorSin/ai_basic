#
# LAB 8-1 편의점 재고 관리 프로그램을 만들자
#
items = { "커피음료": 7, "펜": 3, "종이컵": 2, "우유": 1, "콜라": 4, "책": 5 } 

while True:
    option = int(input("메뉴를 선택하시오 1) 재고조회 2) 입고 3) 출고 4) 종료 : "))
    if option == 1:
        name = input("[재고조회] 물건의 이름을 입력하시오: ") 
        print('재고 :', items[name])
    elif option == 2:
        name, count = input("[입고] 물건의 이름과 수량을 입력하시오 : ").split()
        items[name] += int(count)
    elif option == 3:
        name, count = input("[출고] 물건의 이름과 수량을 입력하시오 : ").split()
        items[name] -= int(count)
    elif option == 4:
        print('프로그램을 종료합니다.')
        break