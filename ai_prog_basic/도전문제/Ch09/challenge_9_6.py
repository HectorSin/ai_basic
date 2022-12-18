import re
txt = 'abc@facebook.com와 bbc@google.com에서 이메일이 도착하였습니다.'
output = re.findall('\S+@[a-z.]+', txt)
for text in output:
    text_split = text.split('@')
    print('추출된 아이디 : ' + text_split[0] + ' , 도메인 : ' + text_split[1])