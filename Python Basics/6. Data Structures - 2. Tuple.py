# tuple is like list but not mutable (list is mutable) and first bracket is used

x = (1, 2, 3)
print(x)

x = 1, 2, 3 # also tuple
print(x)

x = 1, # also tuple of 1 element*
print(x) 

x = () # empty tuple
print(x)

# tuple has also index

tpl = (1, 2, 3)
print(tpl[0])
print(tpl[2])

# tuple is not mutableeeeeee

li = [1, 2, 3]
li[0] = 7
print(li) # [7, 2, 3] mutable

tpl = (1, 2, 3)
# tpl[0] = 7  will throw errorrrrrr cause not mutable


# unpacking a tuple in some variables and then pack them again in tuple

numbers = (10, 20, 30, 40)
n1, n2, n3, n4 = numbers  # unpacking

print(n1) # 10
print(n3) # 30

t = n3, n4  # packing
print(t)


# we can store various things in tuple and loop on them

items = (1, 2, 5.5, ["a", "b", "c"], ("apple", "mango"))
for item in items:
    print(item, type(item))

# accessing any particualar value     
print(items[3])
print(items[3][0])
print(items[4][1])
print(items[4])


# tuple to list

tpl = (1, 2, 3)
li = list(tpl)
print(li)
