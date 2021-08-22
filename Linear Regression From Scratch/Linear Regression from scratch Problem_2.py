# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 00:28:29 2020

@author: Danish
"""

import numpy as np
import pandas as pnd
import matplotlib.pyplot as mplt
#import array as arr
#import math as mp


data = pnd.read_csv('data2.csv')

X_a = data.iloc[:,0]        #X_a is the vector of data.size
n=len(X_a)
X_a = np.asmatrix(X_a)     
x1_max=np.max(X_a)


X_b = data.iloc[:,1]        #X_b is the vector of data.bedrooms
X_b = np.asmatrix(X_b)
x2_max=np.max(X_b)

 
one=np.asmatrix(np.ones(n))
comp_X=np.concatenate((one,X_a,X_b))

#mean normalization of X
mean_x1=np.mean(X_a)
x1_mean_matrix=[[mean_x1]*n] 

mean_x2=np.mean(X_b)
x2_mean_matrix=[[mean_x2]*n] 

zeros=np.asmatrix(np.zeros(n))
comp_X_mean_matrix=np.concatenate((zeros,x1_mean_matrix,x2_mean_matrix))

numerator=comp_X-comp_X_mean_matrix
max_matrix=np.array([1,x1_max,x2_max])
max_matrix=max_matrix.reshape(3,1)
X_temp=numerator / max_matrix
X_final=np.transpose(X_temp)

#mean normalization of Y

Y = data.iloc[:,2]          #Y=data.price
Y = np.asmatrix(Y)
Y_max=np.max(Y)
Y_mean=np.mean(Y)
Y_mean_matrix=[[Y_mean]*n]
Y_num=Y-Y_mean
Y_fina=Y_num/Y_max
Y_final=np.transpose(Y_fina)

theeta = np.array([[1.0],[1.0],[1.0]])
theeta = np.asmatrix(theeta)      #theeta variable

fig = mplt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_temp[1],X_temp[2],Y_fina,c='r',marker='+')
ax.set_xlabel('Size (Sq Ft)')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price (Hundreds)')
mplt.show()


J_list=[]
theeta0_list=[]
theeta1_list=[]
E=1        
itt=0
itt_list=[] 

############################################3

while E>0.001:
    itt=itt+1
    itt_list.append(itt)
    H = np.dot(X_final,theeta)
     
    temp=H-Y_final
    temp_t=np.transpose(temp)
    
    J=(0.5)*np.dot(temp_t,temp)
    indx=itt-1
    J_list.insert(indx,J)   
    
    
    print('iteration:',itt)
    print("E:",E)
    print("Theeta:",theeta)
    
    if itt==1:
        pass
    else:
        E=J_list[indx-1]-J_list[indx]
        
    temp_2=np.transpose(X_final)
    theeta=theeta-0.01*(np.dot(temp_2,temp))
    theeta0_list.append(theeta[0])
    theeta1_list.append(theeta[1])
       

fig = mplt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_temp[1],X_temp[2],Y_fina,c='r',marker='+')
ax.scatter(H, H, H, c='blue', marker='*')
ax.set_xlabel('Size (Sq Ft)')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price (Hundreds)')
mplt.show()

mplt.title('Cost funtion')
mplt.xlabel('No. of Iterations')
mplt.ylabel('J(θ)')

mplt.scatter(itt_list,J_list,color='green',marker='.')
mplt.show()
    
