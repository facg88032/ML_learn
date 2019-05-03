import numpy as np

a = np.arange(4)

print(a)

b=a
c=a
d=b

print(d)
a[0]=11

print(d)

print(b is a)#boolean type

d[1:3]=[22,33]
print(d)

#deep copy(只有值複製過去 但沒有關聯 也就是當a改變b不變)
x=a.copy()
print(x)
a[1]=77
print(x)
print(d)