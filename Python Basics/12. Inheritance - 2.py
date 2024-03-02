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

