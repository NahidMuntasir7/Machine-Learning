# random number

import random

print(random.random()) # random number between 0 to 1
print(random.randint(10, 50)) # random number between a to b

# a game 

number = random.randint(1, 10)
attempts = 0

while True:
    inputnum = input("guess a number between 1 and 10 : ")
    inputnum = int(inputnum)
    attempts += 1
    
    if(inputnum == number):
        print("yes, your guess is correct!")
        break
    if(inputnum > number):
        print("incorrect! pls guess a smaller number")
    else:
        print("incorrect! pls guess a larger number")
print("you tried", attempts, "times to find the correct number")


# 5 / 2 = 2.5 but 5 // 2 = 2 (the int part)
# computer will play the game
# a game 2

number = random.randint(1, 1000)
attempts = 0
lo = 1
hi = 1000

while True:
    print("guess a number between 1 and 1000 : ")
    inputnum = (lo + hi) // 2
    print("my guess is : ", inputnum)
    attempts += 1
    
    if(inputnum == number):
        print("yes, your guess is correct!")
        break
    if(inputnum > number):
        print("incorrect! pls guess a smaller number")
        hi = inputnum - 1
    else:
        print("incorrect! pls guess a larger number")
        lo = inputnum + 1
print("you tried", attempts, "times to find the correct number")



# Program to check if a number is prime or not

num = 29

# To take input from the user
#num = int(input("Enter a number: "))

# define a flag variable
flag = False

if num == 1:
    print(num, "is not a prime number")
elif num > 1:
    # check for factors
    for i in range(2, num):
        if (num % i) == 0:
            # if factor is found, set flag to True
            flag = True
            # break out of loop
            break

    # check if flag is True
    if flag:
        print(num, "is not a prime number")
    else:
        print(num, "is a prime number")



# Fibonacci number

# Program to display the Fibonacci sequence up to n-th term

nterms = int(input("How many terms? "))

# first two terms
n1, n2 = 0, 1
count = 0

# check if the number of terms is valid
if nterms <= 0:
   print("Please enter a positive integer")
# if there is only one term, return n1
elif nterms == 1:
   print("Fibonacci sequence upto",nterms,":")
   print(n1)
# generate fibonacci sequence
else:
   print("Fibonacci sequence:")
   while count < nterms:
       print(n1)
       nth = n1 + n2
       # update values
       n1 = n2
       n2 = nth
       count += 1
