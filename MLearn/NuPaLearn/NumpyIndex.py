import numpy as np

A=np.arange(3,15).reshape((3,4))

#索引
print(A)
print(A[2])
print(A[1][2])
print(A[1,2])
print(A[2,:])#:代表所有
print(A[2,1:3])
print(A.flatten())#變為一列

for col in A:
    print(col)

for row in A.T:
    print(row)

for item in A.flat:#f是objlat
    print(item)

