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


#
