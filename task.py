import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


df=pd.read_csv("DataSet for the Data Science Test.csv")
print("df=\n",df)

df.dropna()

print(df.head())

print(df.tail())

Y1=df['Category']
from sklearn.preprocessing import LabelEncoder
Y1=LabelEncoder().fit_transform(Y1)
print("Y=",Y1)

Y=Y1[:3697]
print("shape(y)=",np.shape(Y))

'''
header_list=['Category']
gf = pd.DataFrame(Y)
print(gf.head())
print(gf)
gf.to_csv('encoded Y.csv',header=header_list,index=False)
'''

x11=df['CompanyName']
x22=df['Financial Department']
x33=df['Financial AccountGroup']
x44=df['VendorName']
x5=df['Amount Month 1']
x6=df['Amount Month 2']
x7=df['Amount Month 3']
x8=df['Amount Month 4']

x1=LabelEncoder().fit_transform(x11)
x2=LabelEncoder().fit_transform(x22)
x3=LabelEncoder().fit_transform(x33)
x4=LabelEncoder().fit_transform(x44)

for i in range(len(x8)):
    if x8[i]=="#NAME?":
        x8[i]=0
    if x7[i]=="#NAME?":
        x7[i]=0
    if x6[i]=="#NAME?":
        x6[i]=0
    if x5[i]=="#NAME?":
        x5[i]=0

X=[x1[:3697],x2[:3697],x3[:3697],x4[:3697],x5[:3697],x6[:3697],x7[:3697],x8[:3697]]
X=np.matrix(X)
X=np.transpose(X)
print("shape(X)=",np.shape(X))

'''
header_list=['CompanyName','Financial Department','Financial AccountGroup','VendorName','Amount Month 1','Amount Month 2','Amount Month 3','Amount Month 4']
gf = pd.DataFrame(X)
print(gf.head())
print(gf)
gf.to_csv('encoded X.csv',header=header_list,index=False) 
'''

#training-lr

print("\n-----------LR--------------")
from sklearn.linear_model import LinearRegression
mdl1=LinearRegression()
mdl1.fit(X,Y)
print("\nmdl1.score(x,y)=",mdl1.score(X,Y)*100,"%")

#training-nb
print("\n-----------nb--------------")
from sklearn.naive_bayes import GaussianNB
mdl2=GaussianNB()
mdl2.fit(X,Y)
print("\nmdL2.score(x,y)=",mdl2.score(X,Y)*100,"%")

#training-svm
print("\n-----------svm--------------")
from sklearn import svm
mdl3=svm.SVC()
mdl3.fit(X,Y)
print("\nmdl3.score(x1,y)=",mdl3.score(X,Y)*100,"%")

#training-svm(by pipeline)
print("\n-----------svm--------------")
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
mdl4 = make_pipeline(StandardScaler(), SVC(gamma='auto'))
mdl4.fit(X,Y)
print("\nmdl4.score(x1,y)=",mdl4.score(X,Y)*100,"%")

#training-RF()
print("\n-----------RF--------------")
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
mdl5 = RandomForestClassifier(max_depth=2, random_state=0)
mdl5.fit(X,Y)
print("\nmdl5.score(x,y)=",mdl5.score(X,Y)*100,"%")

#training-knn
print("\n-----------KNN--------------")
from sklearn.neighbors import KNeighborsClassifier
mdl6=KNeighborsClassifier(n_neighbors=3)
mdl6.fit(X,Y)
print("\nmdl6.score(x,y)=",mdl6.score(X,Y)*100,"%")

filename= 'finalized_model.sav'
joblib.dump(mdl6, filename)
loaded_model = joblib.load(filename)      #load the model frim disk

# Task 1. Predict the label for the bills with blank categories

#testing prediction

I=[x1[3697:],x2[3697:],x3[3697:],x4[3697:],x5[3697:],x6[3697:],x7[3697:],x8[3697:]]
I=np.matrix(I)
I=np.transpose(I)
#print(np.shape(I))

R=np.zeros(1023)
for r in range(1023):
    R[r]=mdl6.predict(I[r])

#print('R=',R)

'''
header_list=['Category']
gf = pd.DataFrame(R)
print(gf.head())
print(gf)
gf.to_csv('Task 1 predicted Y(encoded).csv',header=header_list,index=False)
'''

R=list(R)
z=R
for i in range(len(z)):
    if R[i]==0:
        z[i]='X0'
    if R[i]==1:
        z[i]='X1'
    if R[i]==2:
        z[i]='X2'
    if R[i]==3:
        z[i]='X3'
    if R[i]==4:
        z[i]='X4'

print('label for the bills with blank categories =\n ',z)    
'''
header_list=['category']
gf = pd.DataFrame(z)
print(gf.head())
print(gf)
gf.to_csv('task 1 predicted Y(decoded).csv',header=header_list,index=False)
'''


# Task 2. Predict the label for the new set of bills which will be added in future

#testing

Q1=input("\n\nenter CompanyName = ")
Q2=input("enter Financial Department = ")
Q3=input("enter Financial AccountGroup = ")
Q4=input("enter VendorName = ")
Q5=float(input("enter amount month 1 = "))
Q6=float(input("enter amount month 2 = "))
Q7=float(input("enter amount month 3 = "))
Q8=float(input("enter amount month 4 = "))

from sklearn import preprocessing
LE1=preprocessing.LabelEncoder()
Q1=LE1.inverse_transform(Q1)
print('Q1=',Q1)


Q=np.zeros(8)

for i in range(len(Q)):
    if Q1=='B0':
        Q[i]=0
    if Q1=='B1':
        Q[i]=1
    if Q1=='B2':
        Q[i]=2
    if Q1=='B3':
        Q[i]=3
    if Q1=='B4':
        Q[i]=4
    if Q1=='B5':
        Q[i]=5
    if Q1=='B6':
        Q[i]=6
    if Q1=='B7':
        Q[i]=7
    if Q1=='B8':
        Q[i]=8

        
    if Q2=='D0':
        Q[i]=0
    if Q2=='D1':
        Q[i]=1
    if Q2=='D2':
        Q[i]=12
    if Q2=='D3':
        Q[i]=20
    if Q2=='D4':
        Q[i]=21
    if Q2=='D5':
        Q[i]=22
    if Q2=='D6':
        Q[i]=23
    if Q2=='D7':
        Q[i]=24
    if Q2=='D8':
        Q[i]=25
    if Q2=='D9':
        Q[i]=26
    if Q2=='D10':
        Q[i]=2
    if Q2=='D11':
        Q[i]=3
    if Q2=='D12':
        Q[i]=4
    if Q2=='D13':
        Q[i]=5
    if Q2=='D14':
        Q[i]=6
    if Q2=='D15':
        Q[i]=7
    if Q2=='D16':
        Q[i]=8
    if Q2=='D17':
        Q[i]=9
    if Q2=='D18':
        Q[i]=10
    if Q2=='D19':
        Q[i]=11
    if Q2=='D20':
        Q[i]=13
    if Q2=='D21':
        Q[i]=14
    if Q2=='D22':
        Q[i]=15
    if Q2=='D23':
        Q[i]=16
    if Q2=='D24':
        Q[i]=17
    if Q2=='D25':
        Q[i]=18
    if Q2=='D26':
        Q[i]=19
    

    if Q3=='C0':
        Q[i]=0
    if Q3=='C1':
        Q[i]=1
    if Q3=='C2':
        Q[i]=5
    if Q3=='C3':
        Q[i]=6
    if Q3=='C4':
        Q[i]=7
    if Q3=='C5':
        Q[i]=8
    if Q3=='C6':
        Q[i]=9
    if Q3=='C7':
        Q[i]=10
    if Q3=='C8':
        Q[i]=11
    if Q3=='C9':
        Q[i]=12
    if Q3=='C10':
        Q[i]=1
    if Q3=='C11':
        Q[i]=3
    if Q3=='C12':
        Q[i]=4

Q[3]=Q4
Q[4]=Q5
Q[5]=Q6
Q[6]=Q7
Q[7]=Q8

      
print(Q)
print(type(Q))
print(np.shape(Q))
Q=np.matrix(Q)
result=mdl6.predict(Q)
print("result1 = ",result)

if result==0:
    print('category is X0')
if result==1:
    print('category is X1')
if result==2:
    print('category is X2')
if result==3:
    print('category is X3')
if result==4:
    print('category is X4')
