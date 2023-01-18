import numpy as np
import time
from threading import Thread
import matplotlib.pyplot as pit

score = np.random.randint(40, 100, (10, 6))
print(np.vsplit(score, 2))

print(np.split(score, 2))
print(np.where(score > 60, 1, 0))
print(score[:2,])
print(np.any(score[:2,] > 80))

ones = np.zeros([4,5])

for i in range(4):
    for j in range(5):
        ones[i][j] = i * j
print(ones.T)
print(np.unique(ones))
print(ones.flatten())
print(ones.ravel())

a = np.arange(10)
a = np.linspace(0, 100, 100, endpoint=False, retstep=True)
a = np.arange(10, 50, 2)
a = np.logspace(0, 2, 12)
a = np.random.normal(1, 1, 10000)

a = np.random.uniform(-1, 1, 10000)

pit.figure(figsize=(20, 8), dpi=100)
pit.hist(a, 1000)
#pit.show()

#print(a)

start = time.time()

def CountDown(n):
    while n > 0:
        n -= 1

t1 = Thread(target=CountDown, args=[100000 // 2])
t2 = Thread(target=CountDown, args=[100000 // 2])

t1.start()
t2.start()
t1.join()
t2.join()

print("Time used:", (time.time() - start))

