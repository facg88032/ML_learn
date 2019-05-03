import csv
import numpy as np
import pandas as pd
import math


# read model
w = np.load('model3.npy')


test_x=[]
text=pd.read_csv('test.csv',encoding='big5', header=None)
row=text.index

for r in row:

  if r%18==0 :
    test_x.append([])
    for c in range(2,11):
        test_x[r//18].append(float(text.iloc[r][c]))
  else:
    for c in range(2,11):
        if text.iloc[r][c]!="NR":
            test_x[r//18].append(float(text.iloc[r][c]))
        else:
            test_x[r//18].append(float(0))

test_x = np.array(test_x)
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

filename = "predict2.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()

ans= pd.read_csv('ans.csv')
predict=pd.read_csv(filename)


z=[]

for i in range(240):
    z.append((ans.iloc[i][1]-predict.iloc[i][1])**2)


z=np.array(z)
variance=z.sum()/len(ans)
devation=math.sqrt(variance)

print(devation)

