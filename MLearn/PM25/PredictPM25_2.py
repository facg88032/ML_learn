
import numpy as np
import pandas as pd
import csv
import math

data=[]

for i in range(18):
    data.append([])


text=pd.read_csv('train2.csv',encoding='big5')
row=text.index
for r in row:
    n_row=text.iloc[r]

    for i in range(3,27):
        if n_row[i]!="NR":
            data[r%18].append(float( n_row[i]))
        else:
            data[r % 18].append(float(0))



x=[]
y=[]
amount=240

for m in range(12):

    for d in range((amount-9)): #連續10小時為一筆data
        x.append([])
        for c in range(18): #18種汙染物
            for h in range(9):
              x[(amount-9)*m+d].append(data[c][amount*m+d+h])
        y.append(data[9][amount*m+d+9])

x=np.array(x)
y=np.array(y)

x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

w=np.zeros(len(x[0]))
l_rate=10
iteration=100000


x_t=x.T
print(w)
s_gra=np.zeros(len(x[0]))

for i in range(iteration):
    hypo = np.dot(x, w)
    loss = hypo-y
    variance=np.sum(loss**2)/len(x)
    devation=math.sqrt(variance)
    gra=2*np.dot(x_t,loss)
    s_gra=s_gra+gra**2
    ada = np.sqrt(s_gra)
    w=w-l_rate*gra/ada

    print ('iteration: %d | Cost: %f  ' % ( i,devation))


# save model
np.save('model2.npy',w)


