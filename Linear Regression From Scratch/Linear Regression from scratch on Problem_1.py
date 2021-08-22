# -- coding: utf-8 --
"""
Created on Tue Oct 19 00:28:29 2020

@author: Danish Hafeez 171458
"""
# Danish Hafeez 171458
# Problem #1

import numpy as np
import pandas as pnd
import array as arr
import matplotlib.pyplot as mplt

data = pnd.read_csv('data1.csv')

X = data.iloc[:, 0]
Population = X
X = np.asmatrix(X)
X = np.transpose(X)
n = len(X)
one = np.asmatrix(np.ones(n))
one = np.transpose(one)
comp_X = np.concatenate((one, X), axis=1)

Y = data.iloc[:, 1]
Y = np.asmatrix(Y)
Y = np.transpose(Y)  # Given output

theeta = np.array([[0.5], [0.5]])
theeta = np.asmatrix(theeta)  # theeta variable

mplt.title('DATA')
mplt.xlabel('Population:Ten Thousands')
mplt.ylabel('Profit:Millions(PKR)')
mplt.scatter(data.population, data.profit, color='red', marker='+')
#H_ = np.dot(comp_X,theeta)
#mplt.plot(data.population,H_,color='Black')
mplt.show()

J_list = []
H_list = []
E = 1
itt = 0
iteration_list = []
t_0 = arr.array('f', [])
t_1 = arr.array('f', [])
############################################

while E > 0.001:
    iteration_list.append(itt)
    itt = itt + 1
    indx = itt - 1
    # Hypothesis
    H = np.dot(comp_X, theeta)
    H_list.insert(indx, H)
    # Cost Function
    temp = H - Y
    temp_t = np.transpose(temp)
    J = (0.5) * np.dot(temp_t, temp)
    # Maintaining Cost Function Values       
    J_list.insert(indx, J)
    # calculate e
    if itt == 1:
        pass
    else:
        E = np.abs(J_list[indx] - J_list[indx - 1])

    print('iteration:', itt)
    print("E:", E)
    print("θ:", theeta)

    # Updating Theeta
    temp_2 = np.transpose(comp_X)
    theeta = theeta - 0.0001 * (np.dot(temp_2, temp))
    t_0.append(theeta[0])
    t_1.append(theeta[1])
####################
# Print and Plot final values of e,Itterations and Theeta
print('\n\n\niteration:', itt)
print("E:", E)
print("θ:", theeta)
mplt.title('Hypothesis')
mplt.xlabel('Population:Ten Thousands')
mplt.ylabel('Profit:Millions(PKR)')
mplt.scatter(data.population, data.profit, color='red', marker='+')
mplt.plot(data.population, H, color='blue')
mplt.show()

###### J(θ) plot
mplt.title('Cost funtion')
mplt.xlabel('No. of Iterations')
mplt.ylabel('J(θ)')
mplt.scatter(iteration_list, J_list, color='green', marker='.')
mplt.show()
#######
# contour plot

theta_0 = np.linspace(-10.88957715247941707, 10.8957742185600996, 100)
theta_1 = np.linspace(-4.1930327045272864, 4.1930329577956302, 100)

m, n = np.meshgrid(theta_0, theta_1, sparse=0, indexing='ij')

mesh_mat = np.zeros(shape=(theta_0.size, theta_1.size))

for i, value1 in enumerate(theta_0):
    for j, value2 in enumerate(theta_1):
        
        costc = 0
        theeta_temp = np.array([[value1],[value2]])
        theeta_temp = np.asmatrix(theeta_temp)      #theeta variable
        H_temp=np.dot(comp_X,theeta_temp)
        temp_1=H_temp-Y
        temp_1_t=np.transpose(temp_1)
    
        costc=(0.5)*np.dot(temp_1_t,temp_1)
        mesh_mat[i, j] = costc
        


mplt.contourf(theta_0, theta_1, mesh_mat, alpha=.7)
mplt.axhline(0, color='black', alpha=.5, dashes=[2, 4], linewidth=1)
mplt.axvline(0, color='black', alpha=0.5, dashes=[2, 4], linewidth=1)
r_theta = np.column_stack((t_0, t_1))
for i in range(len(J_list)-1):
    mplt.annotate('', xy=r_theta[i + 1, :], xytext=r_theta[i, :], arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1}, va='center', ha='center')

CS = mplt.contour(theta_0, theta_1, mesh_mat, linewidths=1, colors='black')
mplt.clabel(CS, inline=1, fontsize=8)
mplt.title("Contour Plot of Gradient Descent")
mplt.xlabel("thetha0")
mplt.ylabel("thetha1")
mplt.show()



ax=mplt.axes(projection='3d')
ax.plot_surface(m,n,mesh_mat,rstride=10,cstride=10,cmap='viridis',)
ax.set_xlabel('theeta_0')
ax.set_ylabel('theeta_1')
ax.set_zlabel('cost')

mplt.show()


# -*- coding: utf-8 -*-
"""
@author: Danish Hafeez 171458
"""
# Problem # 2

import numpy as np
import pandas as pnd
import matplotlib.pyplot as mplt

# import array as arr
# import math as mp


data = pnd.read_csv('data2.csv')

X_a = data.iloc[:, 0]  # X_a is the vector of data.size
n = len(X_a)
X_a = np.asmatrix(X_a)
x1_max = np.max(X_a)

X_b = data.iloc[:, 1]  # X_b is the vector of data.bedrooms
X_b = np.asmatrix(X_b)
x2_max = np.max(X_b)

one = np.asmatrix(np.ones(n))
comp_X = np.concatenate((one, X_a, X_b))

# mean normalization of X
mean_x1 = np.mean(X_a)
x1_mean_matrix = [[mean_x1] * n]

mean_x2 = np.mean(X_b)
x2_mean_matrix = [[mean_x2] * n]

zeros = np.asmatrix(np.zeros(n))
comp_X_mean_matrix = np.concatenate((zeros, x1_mean_matrix, x2_mean_matrix))

numerator = comp_X - comp_X_mean_matrix
max_matrix = np.array([1, x1_max, x2_max])
max_matrix = max_matrix.reshape(3, 1)
X_temp = numerator / max_matrix
X_final = np.transpose(X_temp)

# mean normalization of Y

Y = data.iloc[:, 2]  # Y=data.price
Y = np.asmatrix(Y)
Y_max = np.max(Y)
Y_mean = np.mean(Y)
Y_mean_matrix = [[Y_mean] * n]
Y_num = Y - Y_mean
Y_fina = Y_num / Y_max
Y_final = np.transpose(Y_fina)

theeta = np.array([[1.0], [1.0], [1.0]])
theeta = np.asmatrix(theeta)  # theeta variable

fig = mplt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_temp[1], X_temp[2], Y_fina, c='r', marker='+')
ax.set_xlabel('Size (Sq Ft)')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price (Hundreds)')
mplt.show()

J_list = []
theeta0_list = []
theeta1_list = []
E = 1
itt = 0
itt_list = []

############################################3

while E > 0.001:
    itt = itt + 1
    itt_list.append(itt)
    H = np.dot(X_final, theeta)

    temp = H - Y_final
    temp_t = np.transpose(temp)

    J = (0.5) * np.dot(temp_t, temp)
    indx = itt - 1
    J_list.insert(indx, J)

    print('iteration:', itt)
    print("E:", E)
    print("Theeta:", theeta)

    if itt == 1:
        pass
    else:
        E = J_list[indx - 1] - J_list[indx]

    temp_2 = np.transpose(X_final)
    theeta = theeta - 0.01 * (np.dot(temp_2, temp))
    theeta0_list.append(theeta[0])
    theeta1_list.append(theeta[1])

fig = mplt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_temp[1], X_temp[2], Y_fina, c='r', marker='+')
ax.scatter(H, H, H, c='blue', marker='*')
ax.set_xlabel('Size (Sq Ft)')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price (Hundreds)')
mplt.show()

mplt.title('Cost funtion')
mplt.xlabel('No. of Iterations')
mplt.ylabel('J(θ)')

mplt.scatter(itt_list, J_list, color='green', marker='.')
mplt.show()

