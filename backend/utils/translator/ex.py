import os
import sys
import urllib.request
import papago_key
client_id = papago_key.clnt_id # 개발자센터에서 발급받은 Client ID 값
client_secret = papago_key.clnt_secret # 개발자센터에서 발급받은 Client Secret 값
encText = urllib.parse.quote("반갑습니다")
data = "source=ko&target=en&text=" + encText
url = "https://openapi.naver.com/v1/papago/n2mt"
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
response = urllib.request.urlopen(request, data=data.encode("utf-8"))
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    print(response_body.decode('utf-8'))
    print(type(response_body.decode('utf-8')))
else:
    print("Error Code:" + rescode)