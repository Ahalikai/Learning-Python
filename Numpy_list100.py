
# 100道练习带你玩转Numpy
# https://zhuanlan.zhihu.com/p/101985294

import numpy as np
import scipy.spatial
from io import StringIO
import matplotlib.pyplot as plt
import pandas as pd



#75
#np.info(np.cumsum)
def win_move(a, n = 3):
    r = np.cumsum(a, dtype=float)
    r[n:] = r[n:] - r[:-n]
    return r[n - 1:] / n
z = np.arange(20)
print(win_move(z, n = 3))

#74
#np.info(np.repeat)
#np.info(np.bincount)
C = np.bincount([1,1,2,3,4,4,6])
print(np.repeat(np.arange(len(C)), C))

#73
#np.info(np.roll)
f = np.random.randint(0, 100, (10, 3))
F = np.roll(f.repeat(2, axis=1), -1, axis=1)
F = F.reshape(len(F) * 3, 2)
F = np.sort(F, axis=1)
G = F.view(dtype=[('p0', F.dtype), ('p1', F.dtype)])
print(np.unique(G))
input()
#72
a = np.arange(25).reshape((5, 5))
a[[0, 1]] = a[[1, 0]]
print(a)

#71
a = np.ones((5, 5, 3))
b = 2 * np.ones((5, 5))
print(a * b[:,:,None])

#70
z = np.array([1, 2, 3, 4, 5])
nz = 3
z0 = np.zeros(len(z) + (len(z) - 1) * nz )
z0[::nz + 1] = z
print(z0)

#69
A, B = np.random.randint(0, 1, (5, 5)), np.random.randint(0, 1, (5, 5))
np.diag(np.dot(A, B))

np.sum(A * B.T, axis=1)

np.einsum("ij, ji->i", A, B)

#68
D = np.random.uniform(0, 1, 100)
S = np.random.randint(0, 10, 100)
D_sum = np.bincount(S, weights=D)
D_count = np.bincount(S)
print(D_sum / D_count)

print(pd.Series(D).groupby(S).mean())

#67
A = np.random.randint(0, 10, (3, 4, 3, 4))
print( A.sum(axis=(-2, -1)) )

print( A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1) )


#66
h, w = 5, 5
img = np.random.randint(0, 2, (h, w, 3)).astype(np.ubyte)
F = img[..., 0]*(256**2) + img[..., 1]*256 + img[..., 2]
print( len(np.unique(F)) )

#65
X = [1,2,3,4,5,6]
I = [1,3,9,3,4,1]
print( np.bincount(I,X) )

#64
#np.info(np.bincount)
z = np.ones(10)
i = np.random.randint(0, len(z), 20)
z += np.bincount(i, minlength=len(z))
print(z)

print(np.add.at(z, i , 1))

#63 未掌握
class Name(np.ndarray):
    def __new__(cls, array, name = "no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        if obj is None:return
        self.info = getattr(obj, 'name', "no name")

z = Name(np.arange(10), "range_10")
print(z.name)

#62
#np.info(np.nditer)
#
it = np.nditer([np.arange(3).reshape((3,1)), np.arange(3).reshape((3,1))
                , None])
#print(*it)
for x, y, z in it:
    z[...] = x + y #。。。默认
print(it.operands[2])
input()

#61
z = np.random.uniform(0, 1, 10)
print(z.flat[np.abs(z - 0.5).argmin()])

#60
print(~np.random.randint(0, 3, (3, 10)).any(axis=0).any())

#58 59
x = np.random.randint(0, 10, (5, 5))
print(x)
print()
print(x[x[:,3].argsort()])
input()
print(x - x.mean(axis=1, keepdims=True))

#57
np.info(np.put)
n, p = 10, 3
z = np.zeros((n, n))
np.put(z, np.random.choice(range(n**2), p, replace=False), 1)
print(z)

#56
x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
xx, yy = np.meshgrid(x, y, sparse=True)
z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
h = plt.contourf(x,y,z)
plt.show()

np.info(np.meshgrid)

x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10) )
D = np.sqrt(x**2 + y**2)
sigma, mu = 1.0, 1.0
print(np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) ) )

#55
z = np.arange(9).reshape((3, 3))
for i, v in np.ndenumerate(z):
    print(i, v)
for i in np.ndindex(z.shape):
    print(i, z[i])

#54
s = StringIO("1, 2, 3, 4, 5, 6, , , 7, 8, , 9, 10, 11")
d = np.genfromtxt(s, dtype='int', delimiter=",")

#53
z = np.arange(10, dtype=np.int32)
z = z.astype(np.float32, copy=False)
print(z)

#52
z = np.random.random((10, 2))
x, y = np.atleast_2d(z[:,0], z[:,1])
D = np.sqrt( (x - x.T) ** 2 + (y - y.T) ** 2)
print(D)
D = scipy.spatial.distance.cdist(z, z)
print(D)

#51
print(np.zeros(10, [('position', [('x', float, 1), ('y', float, 1) ]),
                    ('color', [('r', float, 1), ('g', float, 1), ('b', float, 1)]) ]
               ) )
input()
#50
z = np.arange(100)
v = np.random.uniform(0, 100)
index = (np.abs(z - v).argmin())
print(z[index])

#49
np.set_printoptions(threshold=np.nan)
Z = np.zeros((16,16))
print (Z)

#48
for dtype in [np.int8, np.int32, np.int64]:
  print(np.iinfo(dtype).min)
  print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
  print(np.finfo(dtype).min)
  print(np.finfo(dtype).max)
  print(np.finfo(dtype).eps)

#47
x = np.arange(8)
y = x + 0.5
c = 1.0 / np.subtract.outer(x, y)
print(np.linalg.det(c)) #求行列式
#np.info(np.linalg.det)

#46
z = np.zeros((5, 5), [('x', float), ('y', float)])
z['x'], z['y'] = np.meshgrid(np.linspace(0, 1, 5),
                             np.linspace(0, 1, 5))
print(z)
#np.info(np.meshgrid)

#45
z = np.random.random(10)
z[z.argmax()] = 0
print(z)

#44
z = np.random.random((10, 2))
x, y = z[:,0], z[0:,1]
r = np.sqrt(x**2 + y**2)
t = np.arctan2(y, x)
print(r)
print(t)

#43
z = np.zeros(10)
z.flags.writeable = False
z[0] = 1

input()
#42
a = np.random.randint(0, 2, 5)
b = np.random.randint(0, 2, 5)

print(np.allclose(a, b) )
print(np.array_equal(a, b) )

#41
np.info(np.add.reduce)
print(np.add.reduce(np.arange(10)))

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
