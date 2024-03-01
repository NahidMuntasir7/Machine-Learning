# saving some lines in file

lines = ["this is first line.", "this is second line.", "this is third line."]

with open("filee.txt", "w") as fp:
    for line in lines:
        fp.write(line + "\n")


# now to open the file

with open("filee.txt", "r") as fp:  # r = read; binary file porte rb = read binary
    content = fp.read()
    print(content)


# now reading one line by one line

# way 1
with open("filee.txt", "r") as fp:
    lines = fp.readlines()
    print(lines) # a list: ['this is first line.\n', 'this is second line.\n', 'this is third line.\n']
    for line in lines:
        print(line)


# way 2
with open("filee.txt", "r") as fp:
    for line in fp:
        print(line)



# handling exception


# division error

def div(a, b):
    return a/b

print(div(10, 2)) # 5.0
# print(div(3, 0)) # ZeroDivisionError: division by zero
print(div(9, 3))


# solving it

def divv(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        print("can not divide by zero")


print(divv(10, 2)) # 5.0
print(divv(3, 0)) # can not divide by zero # None
print(divv(9, 3)) # 3.0


# adding one more
def div(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        print("can not divide by zero")    # its like if.. elif.. elif.. ekta true hole baki gula ar check hoynaa
    except TypeError:
        print("unsupported type. did you used string?")

print(div(10, 2)) # 5.0
print(div(3, 0)) # can not divide by zero # None
print(div(9, 3)) # 3.0
print(div("12", 3)) # unsupported type. did you used string? # None



# file related error

import io

filename = "file.txt"
mode = "r"

try:
    with open(filename, mode) as fp:
        content = fp.read()
        print(content)
except FileNotFoundError: # file na pele
    print(filename, "not found. pls check file's name is correct or not")
except io.UnsupportedOperation: # file ta read mode e na khulle
    print("are u sure?", filename, "is readable?")



# unknown error eraate

try:
    # ja iccha tai
    print(5/0)
except Exception as e:
    print(e)
