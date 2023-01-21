
# 100道练习带你玩转Numpy
# https://zhuanlan.zhihu.com/p/101985294

import numpy as np



#40
z = np.random.random(10)
z.sort()
print(z)
print(sorted.__doc__)

#39
print(np.linspace(0, 1, 11, endpoint=False)[1:])

#38
def gen(n):
    for i in range(n):
        yield i
z = np.fromiter(gen(10), dtype=float, count=-1)
print(z)

#37
z = np.zeros((5, 5))
z += np.arange(5)
print(z)

#36
z = np.random.uniform(0, 10, 10)
print(z - z % 1)
print(np.floor(z))
print(np.ceil(z) - 1)
print(z.astype(int))
print(np.trunc(z))

#35
a = np.ones(3) * 1
b = np.ones(3) * 2
np.add(a, b, out=b)
np.divide(a, 2, out=a)
np.negative(a, out=a)
np.multiply(a, b, out=a)

#34
print( np.arange('2016-07', '2016-08', dtype='datetime64[D]') )

#33
y = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today = np.datetime64('today', 'D')
torr = np.datetime64('today', 'D') + np.timedelta64(1, 'D')

print(str(y))
print(str(today))
print(str(torr))

#32

#31
#np.info(np.seterr)
defa = np.seterr(all="ignore")
z = np.ones(1) / 0
np.seterr(**defa)
print(np.ones(1) / 0)
input()

#30
z1 = np.random.randint(0, 10, 10)
z2 = np.random.randint(0, 10, 10)
print(np.intersect1d(z1, z2))

#29
z = np.random.uniform(-10, 10, 10)
print(np.copysign( np.ceil(np.abs(z)), z))
np.info(np.copysign)
np.info(np.ceil) #向上取整

input()
#27 28
print(np.array(0) /np.array(0) )
print(np.array(0) //np.array(0) )
print(np.array([np.nan]).astype(int).astype(float) )

#26 -1 is start
# -1 + 0 + 1 + 2 + 3 + 4 = 9
print(sum(range(5),-1))
print(sum.__doc__)

#25
z = np.arange(11)
z[(3 <= z) & (z <= 8)] *= -1
print(z)

#24  (5, 3) * (3, 2) = (5, 2)
print(np.dot(np.ones((5, 3)), np.ones((3, 2))))

#23
color = np.dtype([
    ("r", np.ubyte, 1),
    ("g", np.ubyte, 1),
    ("b", np.ubyte, 1),
    ("a", np.ubyte, 1)
])
print(color)
np.info(np.dtype)

#22
z = np.random.random((5, 5))
zmin, zmax = z.min(), z.max()
z = (z - zmin) / (zmax - zmin)
print(z)

#21
print(np.tile( np.array([[0, 1], [1, 0]]), (4, 4)))

#20
print(np.unravel_index(100, (6, 7 ,8)))

#19
z = np.zeros((8, 8), dtype='uint8')
z[1::2, ::2] = 1
z[::2, 1::2] = 1
print(z)

#18
print(np.diag(1 + np.arange(4), -1) )
np.info(np.diag)

#17
print(0*np.nan)
print(np.nan==np.nan)
print(np.inf>np.nan)
print(np.nan-np.nan)
print(0.3==3*0.1)
'''
nan
False
False
nan
False
'''
#15 16
z = np.ones((10, 10))
z[1:-1, 1:-1] = 0
z = np.pad(z, pad_width=1, mode='constant', constant_values=0)
print(z)

#11-14
print(np.eye(3))
print(np.random.random((3,3,3)))
print(np.random.random((3,3,3)).min())
print(np.random.random(30).mean())

#10
print(np.nonzero([1,2,0,0,4,0]))

#9
print(np.arange(9).reshape(3, 3))

#8
z = np.arange(50)
print(z[::-1])

#7
print(np.arange(10, 50))

#6
z = np.zeros(10)
z[4] = 1
print(z)

#5
np.info(np.add)

#4
z = np.zeros((10, 10))
print("%d bytes" % (z.size * z.itemsize))

#3
print(np.zeros(10) )

#2
print(np.__version__)
np.show_config()

#1
