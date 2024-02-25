# adding two numbers using function
def add(n1, n2):
    return n1 + n2

n = 10
m = 5
result = add(n, m)
print(result)
print(add(2.5, 3.5))

###
def PrintPyramid(n):
    for i in range(n):
        for j in range(i):
            print(i)
PrintPyramid(5)


# local global concept
def myfnc(x):
    print("inside myfnc", x)
    x = 10
    print("inside myfnc", x)

x = 20
myfnc(x) # 20, 10 # local variables of function (copies)
print(x) # 20 # remains 20

###
def myfnc(y):
    print("y =", y) # y can not be accessed outside 
    print("x =", x) # x is global

x = 20
myfnc(x) # 20, 20 (cause x is global)

# default parameter value
def myfnc(y = 10):
    print("y = ", y)
x = 20
myfnc(x) # 20
myfnc() # 10 as no parameter so default 10


# default parameter condition

def myfnc(x, y = 10, z = 0):            # kono ekta parameter e default value dile tar pore 
    print("x =", x, "y =", y, "z =", z) # sob koyta parameter e default value dite hobe must
  
x = 5                           
y = 6          # default paramater e argument dile value change hobe 
z = 7
myfnc(x, y, z) # 5 6 7
myfnc(x, y) # 5 6 0
myfnc(x) # 5 10 0


# now we want that, z e kono default value thakbe na
### nirdisto parameter e nirdisto value pathano

def myfnc(x, z, y = 10):
    print("x =", x, "y =", y, "z =", z)

myfnc(x = 1, y = 2, z = 5) # 1 2 5
a = 5
b = 6
myfnc(x = a, z = b) # 5 (10 default) 6
a = 1
b = 2
c = 3
myfnc(y = a, z = b, x = c) # 3 1 2



# sending list as parameter

def addnumbers(numbers):
    result = 0
    for i in numbers:
        result += i
    return result

result = addnumbers([1, 2, 3, 4])
print(result)


# python list e copy jay na... same main list tai jay sob jaygay

def listfnc(li):
    li[0] = 10

mylist = [1, 2, 3, 4]
listfnc(mylist)
print(mylist) # [10*, 2, 3, 4]

list1 = [1, 2, 3, 4, 5]
list2 = list1
list2[0] = 100 # list1 will be changed cause one main copy
print(list1) # [100*, 2, 3, 4, 5]


# builtin sum func

li = [1, 2, 3 ,4]
result = sum(li) 
print(result)


  




