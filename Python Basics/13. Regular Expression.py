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

print(match.group())  # Bangla
