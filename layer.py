
import sys
import numpy as np
import connection as nn

# broadcasting, z is a matrix,
# return a matrix in which each element is performed sigmoid function
# these two activation function are well tested and work well.
def sigmoid_func(z):
    return 1.0/(1+np.exp(-1*z))

def tahn_func(z):
    Y=np.exp(-1*z)
    return (1-Y)/(1+Y)

# compute derivation of two activation functions
def der_sigmoid_func(z):
    Y=sigmoid_func(z)
    return np.multiply(Y,(1-Y))

def der_tahn_func(z):
    Y=tahn_func(z)
    return 1-np.power(Y,2)

class layer:
    #type="neuron layer"
    def __init__(self, name="", shape=None, size=None, active_func=sigmoid_func,derive_func=der_sigmoid_func):
        # name of layer
        self.name=name
        # number of neurons in the layer
        if shape is not None:
            self._get_shape(shape)
            self.size = self.shape[0] * self.shape[1]
        elif size is not None:
            self._get_shape(size)
            self.size = self.shape[0] * self.shape[1]
        else:
            print("no neurons in this layer, redefine layer.")
            return
        # bias value matrix of cur layer, its length should be equal to capacity
        self.bias=np.random.normal(0,0.01,(self.size,1))
        # delta bias value matrix, length should be the neuron numbers of cur layer
        self.dbias = np.zeros((self.size,1))
        # linear summation of input
        self._z=np.zeros((self.size,1))
        # activation value matrix
        self.active=np.zeros((self.size,1))
        # loss
        self.loss=np.zeros((self.size,1))
        # connection FROM self.prelayer To self
        self.con=None

        self.postlayer=None
        self.prelayer=None

        # activation function for all cells in this layer
        self.active_func=active_func
        # derivation function of active function
        self.derive_func=derive_func

    def _get_shape(self,p):
        t=type(p)
        if t == int:
            self.shape=(p,1)
        elif t == tuple:
            self.shape=p

    # forward function
    def forward(self,x=None):
        # there is no input data, nor prelayer
        if (x is None) and (self.prelayer is None):
            print("need X")
            return
        #input layer
        elif self.prelayer is None:
            # check dimension of x
            x=np.array(x)
            x.shape=(self.size,1)
            self.active = x
            return
        else:
            self._z =np.dot(self.con.weight,self.prelayer.active)+self.bias
            # possible for a layer to get addtional input
            # self._z += x
            self.active=self.active_func(self._z)
            return

    # compute loss of the layer.
    def backward(self,y=None):
        # output layer but without data y
        if (self.postlayer is None) and (y is None):
            # y is needed
            print("need Y")
            return
        # output layer
        if self.postlayer is None:
            # convert y to a numpy array with the same shape of this layer
            y = np.array(y)
            y.shape = self.shape
            # compute loss of a sample
            self.loss = np.multiply(self.active - y, self.derive_func(self._z))
            # computation for accumulating dbias and dweight
            self.dbias += self.loss
            self.con.dweight += np.dot(self.loss, np.transpose(self.prelayer.active))
            return
        # treated as hidden layer only if has post layers.
        # no matter whether Y is or Not None
        if self.prelayer is not None:
            # compute loss
            self.temp=np.dot(np.transpose(self.postlayer.con.weight),self.postlayer.loss)
            self.loss=np.multiply(self.temp,self.derive_func(self._z))
            # accumulate dbias and dweight
            self.dbias += self.loss
            self.con.dweight += np.dot(self.loss, np.transpose(self.prelayer.active))
            pass
        # input layer
        else:
            pass


    # update bias and the weight of connection if there is one
    def update_bias_and_weight(self,learn_rate=0.01,mini_batch_size=3,lamda=0.01):
        if self.con is not None:
            self.con.update_weight(learn_rate,mini_batch_size,lamda)
        self.bias=self.bias-learn_rate*self.dbias/mini_batch_size
        # normally we do Not clear the assist matrix
        # these will be done each time before compute a JCost.
        # see computeJcost function in class neuralnetwork
        #self.reset_dbias_zeros()
        #self.reset_loss_zeros()

    def reset_dbias_zeros(self):
        self.dbias=np.zeros((self.size,1))

    def reset_loss_zeros(self):
        self.loss=np.zeros((self.size,1))

    # make a connection between two layers
    # be careful with the direction of connection
    def connect(self,postlayer):
        if self.postlayer is not None:
            print("already has a post layer:%s"%(self.postlayer.name))
            return
        new_con = nn.connection(self, postlayer)
        postlayer.con=new_con
        self.postlayer=postlayer
        postlayer.prelayer=self
        return new_con

    # disconnect two layers if possible
    # we may not clear the connection information of postlayer
    def disconnect(self,postlayer):
        if self.postlayer is None:
            print("cur layer:%s has no postlayer"%(self.name))
            return
        self.postlayer=None
        postlayer.prelayer=None
        #postlayer.con=None

    def __str__(self):
        str = "---- %s %s ----\n"%(self.name,self.shape)
        str +="Bias:\n%s\n"%(self.bias.reshape(self.shape))
        str +="Activation:\n%s\n"%(self.active.reshape(self.shape))
        str +="Connection:\n%s\n"%(self.con)
        return str















