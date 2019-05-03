import numpy as np

A=np.array([1,1,1])
B=np.array([2,2,2])

#合併
C=np.vstack((A,B))#vertical stack 上下合併
D=np.hstack((A,B))#horizontal stack 左右合併
print(C)
print(D)

print(A[np.newaxis,:])#np.newaxis增加維度 可再行或列
print(A[:,np.newaxis])

X=np.array([1,1,1])[:,np.newaxis]
Y=np.array([2,2,2])[:,np.newaxis]

W=np.vstack((X,Y))
Z=np.hstack((X,Y))
print(W)
print(Z)

T=np.concatenate((X,Y,Y,X),axis=1)#多個array合併 axis 0上下merge 1左右merge

print(T)