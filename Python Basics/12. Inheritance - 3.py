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


if __name__ == "main":
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
 
