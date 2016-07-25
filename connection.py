import numpy as np
import layer

# class for connections between layers.

class connection:
    def __init__(self, prelayer=None, postlayer=None):
        # prelayer
        self.prelayer=prelayer
        # post layer
        self.postlayer=postlayer
        self.shape=(self.postlayer.size,self.prelayer.size)
        self.size=self.postlayer.size*self.prelayer.size

        # weight value matrix
        self.weight=np.random.normal(0,0.01,self.shape)
        self.weight.shape=self.shape
        # delta weight matrix
        self.dweight=np.zeros(self.shape)

    def update_weight(self,learn_rate=0.02,trained_samples=1,lamda=1e-6):
        self.weight=self.weight-learn_rate*(self.dweight/trained_samples+self.weight*lamda)
        # do NOT clear dweight matrix to zero
        #self.reset_dweight_zeros()

    def reset_dweight_zeros(self):
        self.dweight=np.zeros(self.shape)

    def __str__(self):
        str ="---- weight of (%s -> %s) %s ----\n"%(self.prelayer.name,self.postlayer.name,self.shape)
        str+="%s\n"%(self.weight)
        return str
