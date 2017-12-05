#! /bin/python
import re
import urllib
import urllib2


#p = r"\w+\s+\w+"
p = r"(?P<name>\w+)\s+(?P<url>.*)"
#m = re.compile(p)
for line in open('data.txt').readlines():
    m = re.match(p, line)
    name = "data/"+m.group('name') + ".txt"
    url =  m.group('url')

    f = urllib2.urlopen(url)
    with open(name, "w") as code:
        code.write(f.read())
    
