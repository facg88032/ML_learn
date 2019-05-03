import pandas as pd
import numpy as np

dates=pd.date_range('20130101',periods=6)
df=pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['A','B','C','D'])
print(df)

print(df['A'],df.A)
print(df[0:3],df['20130102':'20130105'])
print("-----------------")
#select by labe:loc
print(df.loc['20130102'])
print(df.loc[:,["A","B"]])
print(df.loc['20130101',["A","B"]])
print("---------------")
#select by position:iloc
print(df.iloc[3])
print(df.iloc[3,1])
print(df.iloc[1:2,1:3])
print(df.iloc[[1,3,5],1:3])
print("------------------")
#mixed selection:ix
print(df.ix[:3,['A','C']])
print("-----------------")
#boolean indexing
print(df[df.A>8])
