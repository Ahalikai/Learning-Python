













#50
class American(object):
    @staticmethod

input()
#48 49
l = filter(lambda x:x % 2 == 0, range(1, 21))
l = map(lambda x:x**2, range(1, 21))
for i in l:
    print(i, end=" ")
print()

#45 46 47
l = []
for i in range(1, 11):
    l.append(i)

envn_sqrt = map( lambda x:x**2, filter(lambda x:x%2==0, l) )
for i in envn_sqrt:
    print(i, end=" ")
print()

sqrtNum = map(lambda x:x ** 2, l)
for sq in sqrtNum:
    print(sq, end=" ")

enveNum = filter(lambda x:x%2==0, l)
for e in enveNum:
    print(e, end=" ")

#42-44 easy

#33-41
def dict_n(n):
    l = list()
    for i in range(1, n + 1):
        l.append(i ** 2)
    print(l[-5:])
    print(tuple(l))
    d = dict()
    for i in range(1, n + 1):
        d[i] = i ** 2
    for (k, v) in d.items():
        print(v, end=" ")
    print()
    for k in d.keys():
        print(k, end=" ")
dict_n(20)

#29-32 easy

#28
def sum_str(s):
    total = n = 0
    for i in range(len(s)):
        if 49 <= ord(s[i]) <= 57:
            n += int(s[i])
        else:
            total += n
            n = 0
    print(total + n)
str1 = "b532x2x3c4b5"
sum_str(str1)

#27
def Print_intTstr(s):
    print(str(s))
Print_intTstr(12345)

#26
def a_b(a, b):
    return a + b
print(a_b(1, 1))

#25
class Person:
    name = "Person"
    def __init__(self, name = None):
        self.name = name
tom = Person("tom")
print("%s name is %s" % (Person.name, tom.name))
aha = Person
aha.name = "aha"
print("%s name is %s" % (Person.name, aha.name))

#24
print(abs.__doc__)
print(int.__doc__)

#23
def squeue_2(n):
    '''
    :param n: input an int
    :return: n ** 2
    '''
    return n ** 2
print(squeue_2(2))
print(squeue_2.__doc__)

#22
freq ={}
s = input().split(" ")
for word in s:
    freq[word] = freq.get(word,0) + 1
words = sorted(freq.keys())
for word in words:
    print("{} : {}".format(word, freq[word]))

#21
import math
now = [0, 0]
while True:
    n = input()
    if not n:
        break
    a = n.split(" ")
    if a[0] == "UP":
        now[1] += int(a[1])
    elif a[0] == "DOWN":
        now[1] -= int(a[1])
    elif a[0] == "LETF":
        now[0] -= int(a[1])
    elif a[0] == "RIGHT":
        now[0] += int(a[1])
print("{}, {}".format(now[0], now[1]))
print(int (round( math.sqrt(now[0]**2 + now[1]**2) ) ) )

#20
result = []
def putNum(n):
    i = 0
    while i < n:
        j = i
        i += 1
        if j % 7 == 0:
            yield j
for i in putNum(100):
    result.append(i)
print(result)

#19
from operator import itemgetter, attrgetter
l = []
while True:
    s = input()
    if not s:
        break
    s.split(",")
    l.append(tuple(s.split(",")))
print(sorted(l, key=itemgetter(0, 1, 2)))

#18
import re
l = input().split(",")
result = []
for s in l:
    if len(s) < 6 or len(s) > 12:
        continue
    elif re.search("[S#@]",s) and re.search("[0-9]", s) and re.search("[a-z]", s) and re.search("[A-Z]", s):
        result.append(s)
print(result)

#17
result = 0
while True:
    s = input()
    if not s:
        break
    value = s.split(" ")
    if value[0] == 'D':
        result += int(value[1])
    else:
        result -= int(value[1])
print(result)

#16
a = input().split(",")
num = [x for x in a if int(x) % 2 != 0]
print(num)

#15
a = input()
result = 0
n1 = int("%s" % a)
n2 = int("%s%s" % (a, a))
n3 = int("%s%s%s" % (a, a, a))
n4 = int("%s%s%s%s" % (a, a, a, a))
print(n1 + n2 + n3 + n4)

#14
up = 0
low = 0
s = input()
for i in s:
    if i.isupper():
        up += 1
    if i.islower():
        low += 1
print("upper = {}, lower = {} ".format(up, low))

#13
a = 0
b = 0
s = input()
for i in s:
    if i.isdigit():
        b += 1
    if i.isalpha():
        a += 1
print("alpha = {}, digit = {} ".format(a, b))

#12
result = []
for i in range(1000, 3000):
    if (int(i / 1000) % 2 == 0) and (int( (i % 1000 ) / 100) % 2 == 0 )and( int( (i % 100 ) / 10) % 2 == 0 )and i % 2 == 0:
        result.append(i)
print(result)

#11
value = input().split(",")
result =[]
for p in value:
    intp = int(p, 2)
    if(intp % 5 == 0):
        result.append(p)
print(result)

#10
s = input().split(" ")
print(sorted(list(set(s))))

#9
l = []
while True:
    s = input()
    if(s):
        l.append(s.upper())
    else:
        break
print(l)

#8
items = [x for x in input().split(",")]
items.sort()
print(items)

#7
a = input().split(",")
X = int(a[0])
Y = int(a[1])
result = []
for i in range(X):
    row = []
    for j in range(Y):
        row.append(i * j)
    result.append(row)
print(result)

#6
import math
C = 50
H = 30
result = []
l = input().split(",")
for D in l:
    result.append(str(int( math.sqrt(2*C*float(D)/ H)  )))
print(result)

# 5
class InputOutString(object):
    def __init__(self):
        self.s = ""
    def getString(self):
        print("get a string")
        self.s = input()
    def outString(self):
        print(self.s.upper())
strObj = InputOutString()
strObj.getString()
strObj.outString()

#4
import re
print("write some int")
#a = input()
a = "34,67,55"
l = a.split(",")
k = re.findall(r'[0-9]+', a)
t = tuple(k)
print(l, t)

#3
print('get an int')
#n = int(input())
n = 8
d = dict()
for i in range(1, n + 1):
    d[i] = i * i
print(d)

#2
def Mut_1(n):
    result = 1;
    for i in range(1, n + 1):
        result *= i
    print(result)
Mut_1(8)

#1
l = []
for i in range(2000, 3200):
    if(i % 7 == 0):
        if(i % 5):
            l.append(i)
for i in range(len(l) - 1):
    print(l[i], end=', ')
print(l[len(l) - 1])










