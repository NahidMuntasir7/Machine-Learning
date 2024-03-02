# method overriding

# Vehicle class er turn() method ta override korbo Car class e

class Vehicle:
    """base class for all vehicles"""  # docstring

    def __init__(self, name, manufacturer, color):
        self.name = name
        self.manufacturer = manufacturer
        self.color = color

    def turn(self, direction):
        print("turning", self.name, "to", direction)


class Car(Vehicle):  # Car class inherits Vehicle class
    """Car class"""

    def __init__(self, name, manufacturer, color, year):
        self.name = name
        self.manufacturer = manufacturer
        self.color = color
        self.year = 2017
        self.wheels = 4
        print("a new car has been created. name:", self.name)
        print("it has", self.wheels, "wheels")
        print("the car was built in", self.year)

    def change_gear(self, gear_name):
        """method of changing gear"""
        print(self.name, "is changing gear to", gear_name)

    def turn(self, direction):
        print(self.name, "is turning", direction)


if __name__ == "__main__":
    c = Car("Mustang 5.0 GT Coupe", "Ford", "Red", 2017)
    v = Vehicle("Softail Delux", "Harley-Davidson", "Blue")

    c.turn("right") # Mustang 5.0 GT Coupe is turning right (Car class er ta call holo)
    v.turn("right") # turning Softail Delux to right (Vehicle class er ta call holo)
    # jar jar tar tar call hobe





# child class er mothod hote parent class er method ke call
# super()

class Vehicle:
    """base class for all vehicles"""  # docstring

    def __init__(self, name, manufacturer, color):
        print("creating a car")
        self.name = name
        self.manufacturer = manufacturer
        self.color = color

class Car(Vehicle):  # Car class inherits Vehicle class
    """Car class"""

    def __init__(self, name, manufacturer, color, year):
        print("creating a car")
        super().__init__(name, manufacturer, color) # super() method: using the parent's (name manufacturer color)
        self.year = 2017
        self.wheels = 4
        print("a new car has been created. name:", self.name)
        print("it has", self.wheels, "wheels")
        print("the car was built in", self.year)

    def change_gear(self, gear_name):
        """method of changing gear"""
        print(self.name, "is changing gear to", gear_name)

    def turn(self, direction):
        print(self.name, "is turning", direction)


if __name__ == "__main__":
    c = Car("Mustang 5.0 GT Coupe", "Ford", "Red", 2017)
    print(c.name, c.year, c.wheels)
    # Mustang 5.0 GT Coupe 2017 4




# learning inheritance using turtle

import turtle

class AjobTurtle(turtle.Turtle): # turtle module er Turtle class ke inherit korlo
    """AjobTurtle is a class for new type of turtle"""
    pass # mane statement thakbe but apatoto kichu kora lagbe na

if __name__ == "__main__":
    montu = AjobTurtle()
    print(type(montu)) # <class '__main__.AjobTurtle'>

    montu.left(30)
    montu.forward(200)
    turtle.done()


# now as it is an AjobTurtle, amra kichu method override kore change korbo
# like pechone bolle samne jabe, dane bolle baam e jabe ei type

import turtle

class AjobTurtle(turtle.Turtle):
    """Ajobturtle is a class for new type of turtle"""

    def forward(self, pixel):
        super().backward(pixel) # from parent

    def backward(self, pixel):
        super().forward(pixel) # from parent

    def left(self, angle):
        super().right(angle) # from parent

    def right(self, angle):
        print("i won't turn right, because i am ajob!")


if __name__ == "__main__":
    montu = AjobTurtle() # AjobTurtle
    montu.left(30)
    montu.forward(200)
    montu.left(45)
    montu.backward(100)
    montu.right(90)
    montu.forward(10)

    jhontu = turtle.Turtle() # main turtle
    jhontu.shape("turtle")
    jhontu.left(30)
    jhontu.forward(200)
    jhontu.left(45)
    jhontu.backward(100)
    jhontu.right(90)
    jhontu.forward(10)

    turtle.done()
 
