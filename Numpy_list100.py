
# 100道练习带你玩转Numpy
# https://zhuanlan.zhihu.com/p/101985294

import numpy as np
from numpy.lib import stride_tricks
import scipy.spatial
from io import StringIO
import matplotlib.pyplot as plt
import pandas as pd

#100
X = np.random.randn(100)
N = 10000
idx = np.random.randint(0, X.size, (N, X.size))
means = X[idx].mean(axis=1)
print(np.percentile(means, [2.5, 97.5]))

#99
X = np.asarray([[1.0, 0.0, 3.0, 8.0],
                [2.0, 0.0, 1.0, 1.0],
                [1.5, 2.5, 1.0, 0.0]])
n = 4
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
M &= (X.sum(axis=-1) == n)
print(X[M])

#98

input()
#97
A = np.random.uniform(0,1,10)
B = np.random.uniform(0,1,10)
print( np.einsum('i->', A) )
print( np.einsum('i, i->i', A, B) )
print( np.einsum('i, i', A, B) )
print( np.einsum('i, j->ij', A, B) )

#96

#95
I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
B = ( (I.reshape(-1, 1) & (2**np.arange(8))) != 0).astype(int)
print(B[:,::-1])

print(np.unpackbits(I[:, np.newaxis], axis=1))

#94
Z = np.random.randint(0, 5, (10, 3))
E = np.all(Z[:,1:] == Z[:,:-1], axis=1)
print(Z[~E]) #1

print(Z[Z.max(axis=1) != Z.min(axis=1), :])#2

#93
a = np.random.randint(0, 5, (8, 3))
b = np.random.randint(0, 5, (2, 2))
c = (a[..., np.newaxis, np.newaxis] == b)
print( np.where(c.any((3, 1)).all(1))[0] )

#92
x = np.random.rand()
#1
print(np.power(x, 3))
#2
print(x**3)
#3
np.einsum('i,i,i->i',x,x,x)

#91
z = np.array([("Hello", 2.5, 3),
              ("World", 3.6, 2)])
print(np.core.records.fromarrays(z.T, names='c1, c2, c3', formats='S8, f8, i8').dtype)

#90
def cartesian(a):
    a = [np.asarray(a) for a in a]
    shape = (len(x) for x in a)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(a), -1).T

    for n, arr in enumerate(a):
        ix[:, n] = a[n][ix[:, n]]
    return ix

print(cartesian(([1, 2, 3], [4, 5], [6, 7])))

#89
z = np.arange(10000)
np.random.shuffle(z)
n = 5
#Slow
print(z[np.argsort(z)[-n:]])
#Quick
print(z[np.argpartition(-z, n)[:n]])

#88

#87
z = np.ones((16, 16))
k = 4
s = np.add.reduceat(np.add.reduceat(z, np.arange(0, z.shape[0], k), axis=0),
                    np.arange(0, z.shape[1], k), axis=1)
print(s)

#86
p, n = 3, 3
m = np.ones((p, n, n))
v = np.ones((p, n, 1))
print(np.tensordot(m, v, axes=[[0, 2], [0, 1]]))

#85
class Symetric(np.ndarray):
    def __setitem__(self, key, value):
        i, j = key
        super(Symetric, self).__setitem__((i, j), value)
        super(Symetric, self).__setitem__((j, i), value)
    def symetric(z):
        return np.asarray(z + z.T - np.diag(z.diagonal())).view(Symetric)
input()

#84
z = np.random.randint(0, 5, (5, 5))
n = 3
i = 1 + (z.shape[0] - 3)
j = 1 + (z.shape[1] - 3)
print(stride_tricks.as_strided(z, shape=(i, j, n, n), strides=z.strides * 2 ))

#83
z = np.random.randint(0, 5, 10)
print(np.bincount(z).argmax())

#82
#np.info(np.linalg.svd)
z = np.arange(9).reshape((3, 3))
u, s, v = np.linalg.svd(z)
print(np.sum(s > 1e-10))

#81
z = np.arange(0, 15, dtype=np.uint32)
print( stride_tricks.as_strided(z, (11, 4), (4, 4)) )

#80 未掌握

#78 79
def distance(P0, P1, P):
    T = P1 - P0
    L = (T**2).sum(axis=1)
    U = -((P0[:, 0] - p[..., 0]) * T[:, 0] + (P0[:, 1] - p[..., 1]) * T[:, 1]) / L
    U = U.reshape(len(U), 1)
    D = P0 + U * T - P
    return np.sqrt((D**2).sum(axis=1))

p0 = np.random.uniform(-10, 10, (10, 2))
p1 = np.random.uniform(-10, 10, (10, 2))
p = np.random.uniform(-10, 10, (1, 2))
print( distance(p0, p1, p) )

p = np.random.uniform(-10, 10, (10, 2))
print(np.array([distance(p0, p1, p_i) for p_i in p]))

input()
#77
z = np.random.randint(0, 2, 100)
print(np.logical_not(z, out=z))
z = np.random.uniform(-1.0, 1.0, 100)
print(z)
print(np.negative(z, out=z))

#76
def roll(a, n):
    s = (a.size - n + 1, n)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=s, strides=strides)
print( roll(np.arange(10), 3) )

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
