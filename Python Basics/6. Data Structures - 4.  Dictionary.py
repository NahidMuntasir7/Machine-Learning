# Dictionary is like map

marks = {1 : 77, 2 : 76, 5 : 62, 4 : 78, 3 : 65}
print(type(marks)) # <class 'dict'>

print(marks[3]) # 65

marks = {"A" : 3, "B" : 4, "C" : 6}
print(marks["A"]) # 3


# empty dictionary

dt = {}
print(dt) # {}

dt[1] = "one"
dt[2] = "two"
print(dt) # {1: 'one', 2: 'two'}


### dictionary te non-mutable variable key hishebe use korte hoy, for example : number, string, tuple # set, list jayna karon mutable

dt = {"a": "A", "b" : "B", "c" : "C"}
dt[(1, 2, 3)] = "tuple" # tuple as key ( can not use list or set as key)

print(dt) # {'a': 'A', 'b': 'B', 'c': 'C', (1, 2, 3): 'tuple'}


# nesting in dictionary

marks = {"DH101" : {"Bangla": 74, "English": 73}, "DH1002": {"Bangla": 70, "English" : 75}}

print(marks["DH101"]) # {'Bangla': 74, 'English': 73}
print(marks["DH101"]["English"]) # 73
print(marks) # {'DH101': {'Bangla': 74, 'English': 73}, 'DH1002': {'Bangla': 70, 'English': 75}}


# Bangladesh info

bd_division_info = {} 
print(type(bd_division_info)) # <class 'dict'>

bd_division_info["Barishal"] = {"district": 6, "upazilla": 39, "union": 333}

bd_division_info["Chittagong"] = {"district": 11, "upazilla": 97, "union": 336}
bd_division_info["Dhaka"] = {"district": 6, "upazilla": 39, "union": 333}
bd_division_info["Khulna"] = {"district": 10, "upazilla": 59, "union": 270}
bd_division_info["Mymensingh"] = {"district": 4, "upazilla": 34, "union": 350}
bd_division_info["Rajshahi"] = {"district": 8, "upazilla": 70, "union": 558}
bd_division_info["Rangpur"] = {"district": 8, "upazilla": 58, "union": 536}
bd_division_info["Sylhet"] = {"district": 4, "upazilla": 38, "union": 334}

print(bd_division_info)


# keys() method : if i want to print only the keys

divisions = bd_division_info.keys()
print(divisions) # dict_keys(['Barishal', 'Chittagong', 'Dhaka', 'Khulna', 'Mymensingh', 'Rajshahi', 'Rangpur', 'Sylhet'])

for division in divisions:
    print("Division: ", division)
    

# dictionary te loop chalale sudhu key gula pabo :

for item in bd_division_info:
    print(item)
    

# if i want to print both key and value :

# way 1

for key in bd_division_info:
    print(key)
    print(bd_division_info[key])
    
# way 2

for key, value in bd_division_info:
    print(key)
    print(value)
  
