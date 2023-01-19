
# 100道练习带你玩转Numpy
# https://zhuanlan.zhihu.com/p/101985294

import numpy as np










input()
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
