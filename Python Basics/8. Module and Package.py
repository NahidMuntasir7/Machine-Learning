# Module and Package
# Standard library

import math
import datetime
import webbrowser


print(math.pi)
print(math.pow(2, 3))
print(math.sqrt(25))
print(math.floor(5.2))
print(math.ceil(5.2))

# package_name.module_name.function_name()

today = datetime.date.today()
print(today) # 2024-02-27
today = datetime.datetime.today()
print(today) # 2024-02-27 13:09:18.936098



# from datetime import datetime

# d = datetime.today() # without using the package
# print(d) # 2024-02-27 13:15:18.749107


# import webbrowser

url = "http://abc.com"
webbrowser.open(url)

# aliasing (like macro)

# import webbrowser as wb

url = "http://abc.com"
wb.open(url)
