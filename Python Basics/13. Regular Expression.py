# we need to find out the strings that ends with "land"

s = "Afganistan, America, Bangladesh, Canada, Denmark, England, Greenland, Iceland, Netherlands, New Zealand, Sewden, Switzerland"

countries = s.split(",")
print(countries) # ['Afganistan', ' America', ' Bangladesh', ' Canada', ' Denmark', ' England', ' Greenland', ' Iceland', ' Netherlands', ' New Zealand', ' Sewden', ' Switzerland']

li = [item for item in countries if item.endswith("land")]
print(li) # [' England', ' Greenland', ' Iceland', ' New Zealand', ' Switzerland']



# to include Netherlands

li = [item for item in countries if item.endswith("land") or item.endswith("lands")]
print(li) # [' England', ' Greenland', ' Iceland', ' Netherlands', ' New Zealand', ' Switzerland']



import re # importing regular expression

s = "Afganistan, America, Bangladesh, Canada, Denmark, England, Greenland, Iceland, Netherlands, New Zealand, Sewden, Switzerland"

li = re.findall(r'(\w+lands*)', s) # ['England', 'Greenland', 'Iceland', 'Netherlands', 'Zealand', 'Switzerland']
print(li)

# Zealand



# search() function to check a string inside a string or not?

# import re

match = re.search('Bangla', 'Bangladesh')
print(match)  # <re.Match object; span=(0, 6), match='Bangla'>
print(match.group())  # 'Bangla'

match = re.search('des', 'Bangladesh') # this returns a object
print(match.group()) # 'des'

match = re.search('dets', 'Bangladesh') # this will return none
print(match.group()) # error.....


##############################################################################

### dot mane kono character newline(\n) chara
# starts from left and takes spaces also

import re
s = "Bangladesh"
match = re.search('.', s)
print(match.group()) # 'B'

match = re.search('B.n', s)
print(match.group()) # 'Ban'

match = re.search('B.n...', s)
print(match.group()) # 'Bangla'

s = "Bangladesh is our homeland"
match = re.search('............', s)
print(match.group()) # 'Bangladesh i'


### if we want to take only characters or numbers **without space** then use \w

s = "Bangladesh is our homeland"
match = re.search("o\w\w", s) # 'our' and 'ome'.. will return the first one 
print(match.group()) # 'our'

match = re.search("i\w\w", s) # no word like this in this format
print(match) # none


# for ek ba ekadhik character use \w+
s = "Bangladesh is our homeland"
match = re.search("B\w+h", s) 
print(match.group()) # 'Bangladesh'

s = "Bangladesh is our homeland"
match = re.search("B\w+o", s)
print(match.group()) # error none

s = "Bangladesh is our homeland"
match = re.search("B\w+s", s)
print(match.group()) # Banglades



# with spaces
s = "Bangladesh is our homeland"
match = re.search("B.+h", s) # last h porjonto jabe
print(match.group()) # Bangladesh is our h

# first h pelei theme jabe: use ?
s = "Bangladesh is our homeland"
match = re.search("B.+?h", s) # last h porjonto jabe
print(match.group()) # Bangladesh



# extract phone number from a string: use \d
text = "Phone number: 01711101001."
match = re.search('\d+', text)
print(match.group()) # 01711101001

text = "Phone number: 5, Phone number: 01711101001."
match = re.search('\d+', text)
print(match.group()) # 5  :|

# to get the number we need to improve.. use: \d{X}
text = "Phone number: 5, Phone number: 01711101001."
match = re.search('\d{11}', text)
print(match.group()) # 01711101001


# to ignore the spaces use: \s like \d{3}\s*\d{8}

# \s* mane 0 ba tar beshi space thakte pare
# \s+ mane 1 ba tar beshi space thakte pare
text = "Phone number: 5, Phone number: 017 11101001."
match = re.search('\d{3}\s*\d{8}', text)
print(match.group()) # 01711101001


# if ekadhik phone number use findall method instead of search method

# regular expression er aage try to write r ; r = raw string to ignore \ (backslash)

text = "multiple phone numbers, 01711111111, 01811111111, 01910101010, 00000000000 123-123"
result = re.findall(r'\d{3}\s*\d{8}', text)
print(result) #  ['01711111111', '01811111111', '01910101010', '00000000000']






