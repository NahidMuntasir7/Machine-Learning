# downloading photo from internet

import requests

img_url = "https://goo.gl/JxktPw"
r = requests.get(img_url)

with open("pybook.png", "wb") as f: # wb = write binary
    f.write(r.content) # r.text won't work; r.content = byte class er object



# command line argument

# command line thekei jeno chobir link ar filename deya jaay
# so that, baar baar chobir link ar naam change na kora laage

