# basic
s = "hello"
print((len(s)))

country = "Bangladesh"
a = country[1]
print(a)

for c in country:
    print(c)


# string is immutable: can not change any element
country = "Bangladesh"
# country[1] = 'x' # can not do this but it's possible in list


#adding two strings
country = "Bangla" + "desh"
print(country)
num = "10" + "20"
print(num)


# find() method
country = "Bangladesh"
a = country.find("desh") # returns the position
print(a)                 # otherwise returns -1


# replace() method
country = "North Korea. Hello!"
newcountry = country.replace("North", "South")

print(newcountry) # South Korea. Hello!
print(country) # North Korea. Hello! # main string e no change*
                                     # string immutable
text = "hello"
text = text.replace("hello", "Hello") # main "hello" string did                                    # not changed
print(text) # Hello


# strip() method

text = " abc"
# main string wont be changed. cause this functions returns new string and that should be stored
# here new returned string is stored in a
a = text.lstrip() # bam diker space baad dey
print(a)
a = text.rstrip() # right
a = text.strip() # both side er space baad dey


# upper(), lower(), capitalize() method

s = "BangLadesH" # s won't be changed. so need to be stored

sup = s.upper()
print(sup)      # BANGLADESH
slo = s.lower()
print(slo)      # bangladesh
scap = s.capitalize()
print(scap)     # Bangladesh


# split() method
# to get all word string of a sentence string in a list

str = "I am a programmer!"
words = str.split()
print(words)  # ['I', 'am', 'a', 'programmer!']

for word in words:
    print(word)
    

# count(), startswith(), endswith() method 

str = "This is"
cnt = str.count("is")
print(cnt) # 2

s = "Bangladesh"
print(s.startswith("Ban")) # True
print(s.startswith("desh")) # True

# example 1
name = "Mr. Anderson"
if name.startswith("Mr."):
    print("Dear Sir")

# example 2
name = input("Enter your name:")
name = name.lower()

if name.startswith("mr"):
    print("Hello Sire, how are you?")
elif name.startswith("mrs") or name.startswith("miss") or name.startswith("ms"):
    print("Hello Madam, how are you?")
else:
    name = name.capitalize()
    str = "Hi " + name + "! How are you?"
    print(str)


# example 3
str = "a quick brown fox jumps over the lazy dog"
for c in "abcdefghijklmnopqrstuvwxyz":
    print(c, str.count(c))
    
