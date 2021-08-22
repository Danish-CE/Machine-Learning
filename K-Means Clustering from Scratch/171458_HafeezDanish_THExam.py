# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 18:18:36 2021

@author: Danish
"""

import numpy as np
import matplotlib.pyplot as plt
import array as arr
#from array import *
   
x11=np.random.normal(3, 3, size=100)
x12=np.random.normal(7, 3, size=100)
x13=np.random.normal(13, 3, size=100)

X1=np.hstack([x11,x12,x13])

x21=np.random.normal(75, 3, size=100)
x22=np.random.normal(150, 3, size=100)
x23=np.random.normal(250, 3, size=100)

X2=np.hstack([x21,x22,x23])

X = np.vstack([X1,X2])

#Noise addition
X1_noise=np.random.normal(0,1,size=300)
X2_noise=np.random.normal(0,1,size=300)
G_noise=np.vstack((X1_noise,X2_noise))

X_final=X+G_noise

plt.scatter(X1,X2,c=None,marker='o',edgecolors='black')
plt.show()


def initialize_centroids(X_final):
    u1=np.random.choice(X_final[0])
    u2=np.random.choice(X_final[1])
    u=np.vstack((u1,u2))
    return u

centroids = {}
Output={}
C=arr.array('f', [])
distx1=arr.array('f', [])
distx2=arr.array('f', [])

Clust=X_final
c=Clust.transpose()
#number of clusters
K=3
n_iter=100
n=2


prev_Centroids=np.array([]).reshape(n,0) 
Centroids=np.array([]).reshape(n,0) 

#Step 1 Iniliatize centrouids
for i in range(K):
    rand=np.random.randint(0,299)
    Centroids=np.c_[Centroids,c[rand]]

#output will be in form of Dictionary
Output={}
#Step 3 calculating means
for i in range(50):
      EuclidianDistance=np.array([]).reshape(300,0)
      for k in range(K):
          tempDist=np.sum((c-Centroids[:,k])**2,axis=1)
          EuclidianDistance=np.c_[EuclidianDistance,tempDist]
      Class=np.argmin(EuclidianDistance,axis=1)+1
     
      temp={}
      for k in range(K):
          temp[k+1]=np.array([]).reshape(2,0)
      for i in range(300):
          temp[Class[i]]=np.c_[temp[Class[i]],c[i]]
     
      for k in range(K):
          temp[k+1]=temp[k+1].T
    
      for k in range(K):
          Centroids[:,k]=np.mean(temp[k+1],axis=0)
          
      Output=temp
      
#Plotting    
color=['black','red','magenta']
for k in range(K):
    plt.scatter(Output[k+1][:,0],Output[k+1][:,1],facecolors=color[k], edgecolors='black')
plt.scatter(Centroids[0,:],Centroids[1,:],s=300,c='yellow',marker='*',label='Centroids')
plt.show()
print("Final Centroids\n",Centroids)


centroids_cluster=np.transpose(Centroids)
Cost=[]

for j in range(K):
    X_Cluster = Output[j+1]
    length=len(X_Cluster)
    print(centroids_cluster[j])
    for i in range(length):
        dist_temp=np.sum((X_Cluster[i]-centroids_cluster[j])**2,axis=0)
        
    Cost.insert(j,dist_temp)

for i in range(3):
    
    print('Centroids:',centroids_cluster[i],'with cost',Cost[i])