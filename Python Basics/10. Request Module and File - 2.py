# downloading photo from internet

import requests

img_url = "https://goo.gl/JxktPw"
r = requests.get(img_url)

with open("pybook.png", "wb") as f: # wb = write binary
    f.write(r.content) # r.text won't work; r.content = byte class er object



# command line argument

# command line thekei jeno chobir link ar filename deya jaay
# so that, baar baar chobir link ar naam change na kora laage

import sys
import requests

# example
a = 10
b = 5
print(a + b)

# to convert it into command line argument

print(sys.argv) # ['hello.py', '3', '6']
print(type(sys.argv)) # <class 'list'>


# input in terminal : python hello.py 3 6
arguments = sys.argv
a = int(arguments[1])
b = int(arguments[2])

print(a + b) # 9


# now converting photo download into command line argument

# import sys
# import requests

img_url = sys.argv[1]
file_name = sys.argv[2]
r = requests.get(img_url)

with open(file_name, "wb") as f:
    f.write(r.content)

# command line: python hello.py http://goo.gl/Q7LmXw career.png

# this will download a photo from that link and name that file career.png
