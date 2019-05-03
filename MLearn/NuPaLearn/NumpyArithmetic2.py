import numpy as np

A=np.arange(2,14).reshape((3,4))

print(A)

#找出索引值
print(np.argmin(A))#找出最小索引值
print(np.argmax(A))

#平均
print(np.mean(A))#平均
print(A.mean())#同np.mean(A)
print(np.average(A))

#中位數
print(np.median(A))

#累加And差
print("-------------0")
print(np.cumsum(A))
print(np.diff(A))

#找出行列的索引值
print(np.nonzero(A))

#排序
print(np.sort(A))
X=np.arange(14,2,-1).reshape((3,4))
print(np.sort(X))#對每行做排序

#矩陣反向
print("-------------")
print(np.transpose(A))
print("-------------0")
print(A.T)#同為np.transpose
print("-------------1")
print(A.T.dot(A))
print("-------------")


print(np.clip(A,5,9))#min 5 max 9 小於5的值變為5 大於9的值變為9

