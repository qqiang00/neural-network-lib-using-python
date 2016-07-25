import numpy as np
import layer
import matplotlib.pyplot as plt

import neuralnetwork as nw

net=nw.neuralnetwork("mynet",structure=[2,2,1])

#inputlayer=layer.layer("input",2)

#net.addinputlayer(inputlayer)
train=[
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1],
    [0, 0, 0]
]
#print(net)
train=np.array(train)
#print(train)
x_train=train[:,0:2]
y_train=train[:,2:]

net.gradient_check(x_train,y_train,epsilon=1e-4,m=100,lamda=1)

net.fit(x_train,y_train,max_iterate=1e5,j_mini=5e-5,mini_batch_size=-1,learn_rate=0.9,lamda=0.0)



net.predict([1,0])
net.predict([0,1])
net.predict([0,0])
net.predict([1,1])
#print(net)
a=np.array(net.jcostlist)
plt.plot(a[:,0],a[:,1])
plt.show()

