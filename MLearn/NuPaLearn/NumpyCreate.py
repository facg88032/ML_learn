import numpy as np

a=np.array([2,23,4],dtype=np.int)#有分32跟64bytes  float也是
print(a.dtype)


b=np.array([[2,7,1],[7,5,3]])
print(b)


c=np.zeros((3,4)) #數值為0 的自訂矩陣

print(c)

d=np.ones((3,4)) #數值為1 的自訂矩陣

print(d)

e=np.empty((2,3))#數值為空 的自訂矩陣

print(e)

f=np.arange(10,20,2)#同為python自帶range() 起始值為10 終值為20 每部為2

print(f)

g=np.arange(12).reshape((3,4))#從0到11 並重新定義為(3,4)的矩陣

print(g)

h=np.linspace(1,10,5)#起始值為1 終值為10 將這段長分為五段