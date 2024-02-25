# list
# list is mutable ... not like string (string is not immutable)

saarc = ["Bangladesh", "India", "Sri Lanka", "Pakistan", "Nepal", "Bhutan"]

print(saarc)

# append() method : adding new element
saarc.append("Afganistan") # main list will be changed cause mutable
print(saarc)


# sort() method : sorting the elements
saarc.sort()
print(saarc)



# reverse() method : reversing the elements

saarc = ["Bangladesh", "India", "Sri Lanka", "Pakistan", "Nepal", "Bhutan"]
saarc.reverse()
print(saarc)
a = [1, 5, 2, 3, 4, 7, 9]
a.reverse()
print(a)


# insert() method : inserting element at any point

fruits = ["mango", "banana", "orange"]
fruits.insert(0, "apple") # 1st position
print(fruits)
fruits.insert(2, "coconut") # 3rd position
print(fruits)


# remove() method : removing element from list

fruits = ['apple', 'mango', 'coconut', 'banana', 'orange']
fruits.remove("coconut") # will cause error if item is not present

print(fruits)

# to avoid error
item = "pineapple"
if item in fruits: ###
    remove(item)
else:
    print(item, "not in list")
    

# pop() method : remove and return the last element from list

fruits = ['apple', 'mango', 'coconut', 'banana', 'orange']
item = fruits.pop() # popped and returned
print(item) # orange
print(fruits) # ['apple', 'mango', 'coconut', 'banana']

# pop from particular index
item = fruits.pop(1)
print(item) # mango
print(fruits) # ['apple', 'coconut', 'banana']


# extend() method : adding two lists 

li = [1, 2, 3]
li2 = [3, 4, 5, 6]
li.extend(li2)
print(li)

# count() method : counting a elements occurance

li = [1, 2, 3, 3, 4, 5, 6]
a = li.count(3) # 2
b = li.count(10) # 0
print(a)
print(b)

# del() method : delete a element or the whole list
li = [1, 2, 3, 3, 4, 5, 6]
del(li[1]) # one element
print(li) 
del(li) # the list
# print(li) : will cause error


# empty list

li = []
for x in range(10):
    li.append(x)
print(li)

# list addition and multiplication

li1 = [1, 2, 3]
li2 = [4, 5, 6]

li = li1 + li2
print(li) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

li1 = [1, 2, 3]
li2 = li1 * 3
print(li2) # [1, 2, 3, 1, 2, 3, 1, 2, 3]


# joining string list

li = ["a", "b", "c"]
print(li)

str = "".join(li) # abc
print(str)

str = ",".join(li) # a,b,c
print(str)

str = "+".join(li) # a+b+c
print(str)


# list comprehensions

# example 1
li = [1, 2, 3, 4]
newli = []
for x in li:
    newli.append(2 * x)
print(newli) # [2, 4, 6, 8]

# using list comp.

newli = [2 * x for x in li]
print(newli) # [2, 4, 6, 8]


# example 2
li = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 

evennumbers = []
for x in li:
    if x % 2 == 0:
        evennumbers.append(x)

print(evennumbers)

# using list comp.

evennumbers = [x for x in range(1, 11) if x % 2 == 0]
print(evennumbers)
