print(2 == 2)
print(2 <= 3)


#list
numbers = [1, 2, 3, 4 ,5]
print(numbers)
print(len(numbers))
print(numbers[1])

Saarc = ["BD", "AFG", "IND", "PAK", "BHU", "NEP", "SRI"]
print("BHU" in Saarc)
print("USA" not in Saarc) # ache ki nai


#if statement
Saarc = ["BD", "AFG", "IND", "PAK", "BHU", "NEP", "SRI"]

country = input("Enter a country :")
if country in Saarc: # x in List
    print(country, "is a member of Saarc")
else:
    print(country, "is not a member of Saarc")
    
print("Program Terminated")


marks = input("Enter your marks :")
marks  = int(marks)

if marks >= 80:
    grade = "A+"
elif (marks >= 70):
    grade = "A"
elif marks >= 60: # if one cond is true others won't be checked
    grade = "A-"  # so order matters
elif marks >= 50:
    grade = "B"
else:
    grade = "F"

print("Your grade is", grade)

# max among three
n1 = 20
n2 = 30
n3 = 25

if n1 > n2:
    maxn = n1
else:
    maxn = n2
if (n3 > maxn):
    maxn = n3
print("Maximum : ", maxn)

# leap year

year = input("Enter a year : ")
year = int(year)

if year % 100 != 0 and year % 4 == 0:
    print("Yes")
elif year % 100 == 0 and year % 400 == 0:
    print("Yes")
else:
    print("No")
