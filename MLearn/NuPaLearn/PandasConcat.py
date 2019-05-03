import pandas as pd
import numpy as np

df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])

#res=pd.concat([df1,df2,df3],axis=0) #axis 合併方向
res=pd.concat([df1,df2,df3],axis=0,ignore_index=True)#ignore_index索引重新排序
print(res)

#join['inner','outer']
df4 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
df5 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d', 'e'], index=[2,3,4])
res=pd.concat([df4,df5],join='inner',ignore_index=True,axis=0,sort=True)
print(res)
res=pd.concat([df4,df5],join='outer',ignore_index=True,axis=0,sort=True)
print(res)

res=pd.concat([df4,df5],axis=1,join_axes=[df4.index])
print(res)

#append
df6 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df7 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df8 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
res=df6.append(df7,ignore_index=True)
print(res)
res=df6.append([df7,df8],ignore_index=True)
print(res)

s1=pd.Series([1,2,3,4],index=['a','b','c','d'])
res=df6.append(s1,ignore_index=True)
print(res)