# page 42
# client to server two types of requests :
# 1. GET(webpage dekhte chaile) 2. POST(data server e pathate chaile, like form)


import requests

url = "https://www.facebook.com/"
res = requests.get(url)

print(res.ok) # True
print(res.reason) # OK
print(res.status_code) # 200, mane all okk


# now we want to see the content of a web page(HTML Code)

res = requests.get("https://example.com/")
print(res.ok)
print(res.text) # shows  the HTML code
print(type(res.text)) # <class 'str'> eita ekta string

# now if we can save this string in a ** file ** we can download this webpage
# for more details page 47


# creating a file and writing in a file

fp = open("test.txt", "w") # file name and write mode
fp.write("this is a file created with python") # writing in that file
fp.close() # closing the file

# file ta 2nd bar w mode e open korle ager lekha gula muche jabe... notun kichu lehhle ta save hobe
# ar file ekbar close korle sekhane ar kichu lekha jabe na


# another way

with open("testt.txt", "w") as f:
    f.write("Hello, Python")

# ekhane f.close() lekha lagbe na
# sob kichu with block er vetore lekhte hobe

# jodi file e 2nd bar lekhte chai without removing the previous content
# then file ta khulte hobe "a" (append) mode e



# saving the webpage as a file

url = "https://example.com/"
response = requests.get(url)

with open("helo.html", "w") as f:
    f.write(response.text) # saves the html file


# facebook link kaj kore na uporer way te...
# so, unicode related error erate
# bolte hobe file er encoding jeno amader response text er encoding er moto hoy
 
url = "https://www.facebook.com/"
response = requests.get(url)

with open("fb.html", "w", encoding=response.encoding) as f: 
    f.write(response.text) # works very fine



# now file ta directly browser e open korte chaile

import requests
import os
import webbrowser as wb

url = "https://www.facebook.com/"
response = requests.get(url)

with open("fbook.html", "w", encoding=response.encoding) as f:
    f.write(response.text)

filepath = os.path.realpath("fbook.html") # will give the path of the file
print(filepath) # C:\Users\User\PycharmProjects\pythonProject\fbook.html

wb.open("file://" + filepath) # directly opens the webpage in browser
