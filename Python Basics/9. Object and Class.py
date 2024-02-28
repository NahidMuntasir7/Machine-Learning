# class 1

class Car: # generally first letter capital
    name = "premio"
    color = "while"
    
    def start():
        print("starting the engine sire")
        
print("name of the car:", Car.name) # name of the car: premio
print("Color:", Car.color) # Color: while
Car.start() # starting the engine sire


# class 2

class Carr:
    name = ""
    color = ""
    
    def start():
        print("starting the engine!")
        
Carr.name = "axio"
Carr.color = "black"
print("name of the car:", Carr.name) # name of the car: axio
print("Color:", Carr.color) # starting the engine!
Carr.start() # starting the engine!

# print(dir(Carr)) # will print the attributes


# making a object of Carrr class
class Carrr:
    name = ""
    color = ""
    
    def start():
        print("starting the engine!")


# creating a Carrr object
mycar = Carrr() ###
mycar.name = "allion"
print(mycar.name)
# mycar.start() # will cause problem... Carrr.start() takes 0 positional arguments but 1 was given
# karon method er vetore object apnaapni chole ashe

# to fix it, write:
    # def start(self): # self chara onno kichu dileo cholbe
        # print("starting the engine!")
        
# eta korle self e object er reference jabe

class CaR:
    name = ""
    color = ""
    
    def __init__(self, name, color): # like C++ Constructor with parameters
        self.name = name    # instance attribute: alada kore name ar color naam e kono
        self.color = color  # variable declare kori nai 
                            # variable parameter same naam e no  prob...
    def start(self): # self to avoid error
        print("starting the engineee")
        
mycarr = CaR("bugatti", "black")
print(mycarr.name) # bugatti
print(mycarr.color) # black
mycarr.start() # starting the engineee


CaR.start(mycar) # starting the engineee
                 # proof je self e object pathano hoy but we won't use like this
  

# multiple object

class CAR:
    def __init__(self, n, c):
        self.name = n
        self.color = c
    
    def start(self):
        print("name:", self.name)
        print("color:", self.color)
        print("starting the car sire")

mycar1 = CAR("ford", "white")
mycar1.start() # name: ford # color: white # starting the car sire
mycar2 = CAR("ferrari", "red") 
mycar2.start() # name: ferrari # color: red # starting the car sire


# adding a attribute from outside(do not want to change the class)

mycar1.year = 2024
print(mycar1.name, mycar1.color, mycar1.year) # ford white 2024


                 


