import numpy as np
import pandas as pd


dates=pd.date_range('20130101',periods=6)
df=pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['A','B','C','D'])
df.iloc[0,1]=np.nan
df.iloc[1,2]=np.nan

#print(df.dropna(axis=0,how='any')) #how={'any','all'} 除掉有NaN的行或列

#print(df.fillna(value=0)) #NaN轉為0

#print(df.isnull()) #boolean 顯示有無NAN值

print(np.any(df.isnull())==True)#判斷裡面所有數據有無任意一個NaN


