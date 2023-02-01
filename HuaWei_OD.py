'''
3.压缩列表求积
1.列表中第一个数字表示值,第二个数字表示个数
2.输出的也是一个压缩列表,数值相同的且连续的需要合并
3.列表长度不足的补0
给你任意两个压缩列表,输出两个列表相乘后的压缩列表
(压缩列表的解压后的个数为0-2^32之间)
a =  [2, 2, 2, 4, 4, 4, 4, 4 ]      压缩后为 [[2, 3], [4, 5]]
b = [4, 4, 4, 2, 2, 2, 2, 2, 2]     压缩后为 [[4, 3], [2, 6]]
a*b = [8, 8, 8, 8, 8, 8, 8, 8, 0]   压缩后为 [[8, 8], [0, 1]]

a = [1, 1, 2, 2, 2, 2, 2, 2, 2]     压缩后为 [[1, 2], [2, 7]]
b = [2, 2, 2, 4, 4, 4, 4, 4]        压缩后为 [[2, 3], [4, 5]]
a*b = [2, 2, 4, 8, 8, 8, 8, 8, 0]   压缩后为 [[2, 2], [4, 1], [8, 5], [0, 1]]

现有
压缩列表a = [[1, 9000000], [2, 88888888], [3, 99999999], [4, 800000000]]
压缩列表b = [[2, 20000000], [3, 88888888], [2, 19999999], [1, 100000000]]
求 a*b 的压缩列表
'''
import numpy as np
a = [[1, 2], [2, 7]]
b = [[2, 3], [4, 5]]

# 更优解
a_i = 0
b_i = 0
result = []

def res_append(num, len):
    if result == [] :
        result.append([int(num), len])
    elif(result[-1][0] == int(num)):
        result[-1][1] = result[-1][1] + len


len_now = a[a_i][1]
while a_i < len(a) and b_i < len(b):
    if(len_now > 0):
        len_now = len_now - b[b_i][1]
        if(len_now > 0):
            res_append(a[a_i][0] * b[b_i][0], b[b_i][1])
            b_i = b_i + 1
    else:
        print(len_now)
        if (a_i < len(a)):
            len_now = len_now + a[a_i][1]
            a_i = a_i + 1
        if(len_now < 0):
            res_append(a[a_i][0] * b[b_i][0], len_now + b[b_i][1])


        if (len_now == 0):
            b_i = b_i + 1
if len_now != 0:
    result.append([int(0), abs(len_now)])
print(result)



'''
# 暴力解
a_org = []
b_org = []
res_org = []
for i in a:
    for j in range(i[1]):
        a_org.append(i[0])
for i in b:
    for j in range(i[1]):
        b_org.append(i[0])

max_len = max(len(a_org), len(b_org))
min_len = min(len(a_org), len(b_org))

res_org = np.zeros(max_len)
for i in range(min_len):
    res_org[i] = a_org[i] * b_org[i]

result = []
num, totol = res_org[0], 1
for i in range(1, len(res_org)):
    if res_org[i] == num:
        totol = totol + 1
    else:
        result.append([int(num), totol])
        num = res_org[i]
        totol = 1
result.append([int(num), totol])
#print(res_org)
print(result)
'''