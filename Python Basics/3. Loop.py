for i in range(10):
    print("Heloooo!")

for i in range(5):
    print(i)
 
    
# adding 50 ta 1
result = 0
for i in range(50):
    result += 1
print(result)


# adding 1 + 2 + 3 ... 
result = 0
num = 1
for i in range(50):
    result += num
    num += 1
print(result)


# adding 1 + 2 + 3 ... 
result = 0
for i in range(51):
    result += i
print(result)


# range a to b
result = 0
for num in range(1, 51):
    result += num
print(result)


# increment += 5
for i in range(1, 20, 5):
    print(i)


# max
numbers = [1, 2, 5, 3, 4]
maxx = numbers[0]
for n in numbers:
    if(n > maxx):
        maxx = n
print(maxx)


# 5 multiple sum till 100
result = 0
for num in range(101):
    if(num % 5 == 0):
        print(num)
        result += num
print("Sum is : ", result)

# or
result = 0
for num in range(5, 101, 5):
        print(num)
        result += num
print("Sum is : ", result)


# nested loop
for i in range(1, 5):
    for j in range(4):
        print(i)


# looping in list : 
saarc = ["BD", "IND", "PAK", "AFG"]

for country in saarc:
    print(country, "is a member of SAARC")
  
    
# making list using range
li = list(range(11)) # 0 to 10
print(li)

li = list(range(2, 21, 2)) # 2, 4, 6 ... ,20
print(li)

# while loop :
i = 0
while i < 5:
    print(i)
    i += 1
    
i = 5
while i >= 0:
    i -= 1
    print(i)
    
# namta using while loop :

n = input("Enter a n : ")
n = int(n)

m = 1
while (m <= 10):
    print(n, "X", m, "=", n * m)
    m += 1

# for loop inside while loop :
cnt = 1
while(cnt <= 5):
    for i in range(3):
        print(i)
    print("--------")
    cnt += 1
    
# break and continue :

#
while True:
    n = input("Enter any number(0 to Exit)): ")
    n = int(n)
    if(n == 0):
        break;
    print("Square of", n, "is", n * n)
    
#   
while True:
    n = input("Enter any number(0 to Exit)): ")
    n = int(n)
    if(n < 0):
        print("Only positive number pls")
        continue
    if(n == 0):
        break;
    print("Square of", n, "is", n * n)

###
terminate = False ###
while not terminate:
    num1 = input("enter first number: ")
    num1 = int(num1)
    num2 = input("enter the second number: ")
    num2 = int(num2)
    
    while True:
        operation = input("pls enter add/sub or quit to exit:")
        if operation == "quit":
            terminate = True ###
            break
        if operation not in ["add", "sub"]:
            print("unknown operation")
            continue
        if operation == "add":
            print("result is :", num1 + num2)
            break
        if operation == "sub":
            print("result is :", num1 - num2)
            break


