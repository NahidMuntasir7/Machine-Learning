# making the Vehicle class

class Vehicle:
    """base class for all vehicles"""

    def __init__(self, name, manufacturer, color):
        self.name = name
        self.manufacturer = manufacturer
        self.color = color

    def drive(self):
        print("Driving", self.manufacturer, self.name)

    def turn(self, direction):
        print("turning", self.name, "to", direction)

    def brake(self):
        print(self.name, "is stopping!")

if __name__ == "__main__":       # works fine without this line ...
    v1 = Vehicle("Veyron", "Bugatti", "Black")
    v2 = Vehicle("Softail Delux", "Harley-Davidson", "Blue")
    v3 = Vehicle("Mustang 5.0 GT Coupe", "Ford", "Red")

    v1.drive() # Driving Bugatti Veyron
    v2.drive() # Driving Harley-Davidson Softail Delux
    v3.drive() # Driving Ford Mustang 5.0 GT Coupe

    v1.turn("left") # turning Veyron to left
    v2.turn("right") # turning Softail Delux to right

    v1.brake() # Veyron is stopping!
    v2.brake() # Softail Delux is stopping!
    v3.brake() # Mustang 5.0 GT Coupe is stopping!




# Car class inherits from Vehicle class


class Vehicle:
    """base class for all vehicles""" # docstring

    def __init__(self, name, manufacturer, color):
        self.name = name
        self.manufacturer = manufacturer
        self.color = color

    def drive(self):
        print("Driving", self.manufacturer, self.name)

    def turn(self, direction):
        print("turning", self.name, "to", direction)

    def brake(self):
        print(self.name, "is stopping!")


# creating the Car class

class Car(Vehicle):  # Car class inherits Vehicle class
    """Car class"""
    def change_gear(self, gear_name):
        """method of changing gear"""
        print(self.name, "is changing gear to", gear_name)

if __name__ == "__main__":
    c = Car("Mustang 5.0 GT Coupe", "Ford", "Red") # using it's parents constructor
    c.drive() # Driving Ford Mustang 5.0 GT Coupe
    c.brake() # Mustang 5.0 GT Coupe is stopping!
    c.change_gear("p") # Mustang 5.0 GT Coupe is changing gear to p




# now a constructor of Car class 

class Vehicle:
    """base class for all vehicles""" # docstring

    def __init__(self, name, manufacturer, color):
        self.name = name
        self.manufacturer = manufacturer
        self.color = color

    def drive(self):
        print("Driving", self.manufacturer, self.name)

    def turn(self, direction):
        print("turning", self.name, "to", direction)

    def brake(self):
        print(self.name, "is stopping!")


# creating the car class with a constructor

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

if __name__ == "__main__":
    c = Car("Mustang 5.0 GT Coupe", "Ford", "Red", 2017) 

# a new car has been created. name: Mustang 5.0 GT Coupe
# it has 4 wheels
# the car was built in 2017
