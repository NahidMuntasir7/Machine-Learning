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

