# collection of items

# empty set

a = set()
print(type(a)) # <class 'set'>


# set from given elements

items = {"pen", "laptop", "cellphone", "pen"}
print(items) # all elements are distinct & in sorted order
# {'laptop', 'pen', 'cellphone'}


# adding items

sett = {"apple"}
sett.add("banana")
print(sett) # {'banana', 'apple'}

a = set() # empty set
a.add("mango")
print(a) # {'mango'}

b = set()
b.add("orange")
print(b) # {'orange'}


# set from a List or a Tuple or a String

li = [1, 2, 3, 4]
tpl = (1, 2, 3)
string = "Bangla desh"
A = set(li)
print(A) # {1, 2, 3, 4}

B = set(tpl)
print(B) # {1, 2, 3}

C = set(string) 
print(C) # {' ', 'l', 'e', 's', 'd', 'B', 'a', 'n', 'g', 'h'}


# checking an element belongs to a set or not

A = {1, 2, 3}
print(1 in A) # True


# Some set operations

A = {1, 2, 3, 4, 5}
B = {2, 4, 6, 8}

C = A & B # common
print(C)  # {2, 4}
C = A | B # all
print(C)  # {1, 2, 3, 4, 5, 6, 8}
C = A ^ B # not common
print(C) # {1, 3, 5, 6, 8}

C = A - B
print(C) # {1, 3, 5}
C = B - A
print(C) # {8, 6}
