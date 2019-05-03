import csv
import numpy as np
import pandas as pd
data=[]

text=pd.read_csv('train.csv',encoding='big5')



for i in range(180,4320,360):

    for j in range(180):
       data.append(text.iloc[i+j])


data=np.array(data)

data=pd.DataFrame(data)

row=data.index
filename = "train2.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(['日期','	測站','測項','0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23'])
for i in row:
    s.writerow(data.iloc[i])
