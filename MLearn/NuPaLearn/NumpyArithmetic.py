import numpy as np

a=np.array([10,20,30,40])
b=np.arange(4)
print(a,b)
print(b<3)#boolean值
c=a-b
print(c)

d=a+b
print(d)

e=b**2
print(e)

f=np.sin(a)
print(f)

x=np.array([[1,5],
            [2,3]])
y=np.arange(6).reshape((2,3))
#z=x*y
z_dot=np.dot(x,y)
z_dot_2=x.dot(y)
#print(z)
print(z_dot)
print(z_dot_2)

g=np.random.random((2,4))#從0~1隨機數字
print(g)
print(np.sum(g))
print(np.min(g))
print(np.max(g))

print(np.sum(g,axis=1))#axis指定行或指定列 1為行 0為列
print(np.min(g,axis=0))
print(np.max(g,axis=1))