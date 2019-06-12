import os
loop = "Y"
while loop == "Y":
    S = 0
    a_ = input ("请输入要判断的数\n")
    a = int ( a_ )
    bei = 2
    while (bei <= (a/2)):
        if (a % bei) == 0:
            S = 1
            break
        else:
            bei += 1
    if S == 0 :
        print ("Y")
    else:
        print ("N")
    loop = input ("是否继续 Y or N\n")
    while (loop != "Y") and (loop != "N"):
        print ("请重新输入\n")
        loop = input ("是否继续 Y or N\n")
os.system("pause")



    
