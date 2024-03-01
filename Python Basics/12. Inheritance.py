# making the vehicle class

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

if __name__ == "__main__":
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




