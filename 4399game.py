import urllib.request
import re

p = 50001
with open("4399小游戏统计.txt", mode="w") as f:
    f.write("")

for i in range(40001, p):
    try:
        url = "http://www.4399.com/flash/{}.htm".format(i)
        request = urllib.request.Request(url)
        response = urllib.request.urlopen(request)
        html = response.read().decode("GB2312")
        print("{}".format(i)+(re.search(r'(?<=<title>).*(?=,4399)', html)).group())
        with open("4399统计.txt", mode="a")as f:
            f.write("{}存在  ".format(i) + (re.search(r'(?<=<title>).*(?=,4399)', html)).group() + "\n")
    except:
        with open("4399小游戏统计.txt", mode="a")as f:
            f.write("{}不存在\n".format(i))
        pass
print("10001至{}已完成".format(p-1))
