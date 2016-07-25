# base class of network.
import layer
import numpy as np
import time

class trainingparameter:
    def __init__(self,max_iterate=1e3,jcost_min=1e-4,mini_batch_size=10,learn_rate=0.1,lamda=1e-6):
        self.max_iterate=max_iterate
        self.jcost_min=jcost_min
        self.m=mini_batch_size
        self.lear_rate=learn_rate
        self.lamda=lamda
        pass

class neuralnetwork:
    def __init__(self,name="network",structure=[1,1,1]):
        self.name=name
        self.inputlayer=None
        self.outputlayer=None
        self.layerlist=[]
        self.__buildnetworkfromstructure(structure)
        # store jcost data for analysis
        self.jcostlist=[]


    def __buildnetworkfromstructure(self,structure):
        self.layerlist = []
        for i in range(len(structure)):
            newlayer = layer.layer(name="layer%s"%(i), shape=structure[i])
            self.layerlist.append(newlayer)
            if i >= 1:
                self.layerlist[i - 1].connect(self.layerlist[i])

        self.inputlayer = self.layerlist[0]
        self.outputlayer = self.layerlist[len(self.layerlist) - 1]

    # add a specific layer to current network's last layer
    def addclassifier(self,classifier):
        self.outputlayer.connect(classifier)
        self.layerlist.append(classifier)
        self.outputlayer=self.layerlist[len(self.layerlist)-1]

    # add an layer to current network as an input layer
    def addinputlayer(self,inputlayer):
        self.layerlist.insert(0,inputlayer)
        self.inputlayer=self.layerlist[0]
        self.inputlayer.connect(self.layerlist[1])

    # give an hypothesis for a given x
    def predict(self,x=None):
        self._forward(x)
        print("predict of x:%s is: %s"%(x,self.outputlayer.active))
        return self.outputlayer.active

    # forward flow the network for predict or other purpose
    def _forward(self,x=None):
        if self.inputlayer is None:
            print("assign a input layer for network")
            return
        self.inputlayer.forward(x)
        nextlayer=self.inputlayer.postlayer
        while nextlayer is not None:
            nextlayer.forward()
            nextlayer=nextlayer.postlayer
        return self.outputlayer.active

    # backward flow the network for back propagation algorithm
    def _backward(self,y=None):
        if self.outputlayer is None:
            print("assign a output layer for network")
            return
        self.outputlayer.backward(y)
        curlayer=self.outputlayer.prelayer
        while (curlayer is not None) and \
                (curlayer is not self.inputlayer):
            curlayer.backward()
            curlayer=curlayer.prelayer
        return

    # check accordance of data set.
    def _check_dataset(self, x, y):
        # unequal numbers of x and y
        if x.shape[0] != y.shape[0]:
            return False
        else:
            return True

    # set some assist parameter matrix to zero
    def _reset_dparameter_zeros(self):
        for lyr in self.layerlist:
            lyr.reset_dbias_zeros()
            lyr.reset_loss_zeros()
            if lyr.con is not None:
                lyr.con.reset_dweight_zeros()
        pass

    # compute one sample cost
    def _computejcostxy(self, h, y):
        h = np.array(h)
        y = np.array(y)
        h = h - y
        return np.sum(np.multiply(h, h) / 2)

    # compute weight punishing cost
    def _computeJpunish(self, lamda=0.001):
        if lamda == 0:
            return 0
        j = 0.0
        for templayer in self.layerlist:
            if templayer.con is not None:
                j += np.sum(np.power(templayer.con.weight, 2))
        j = j * lamda / 2.0
        return j

    # compute Jcost
    def computeJcost(self, X, Y,start=0, length=10, lamda=1e-8, backward=True):
        sample_size = X.shape[0]
        if sample_size <= 0:
            return None
        jcost = 0.0
        if length <= 0:
            length = 1
        # clear matrix to zero for compute
        self._reset_dparameter_zeros()
        times = 0
        index=start
        while True:
            index = index % sample_size
            self._forward(X[index])
            jcost += self._computejcostxy(self.outputlayer.active, Y[index])
            if backward == True:
                self._backward(Y[index])
            times += 1
            index += 1
            # give an Jcost and update parameters
            if times == length:
                # give an Jcost
                jcost = jcost / length
                jcost += self._computeJpunish(lamda)
                break
        return jcost


    # train cur network
    # update parameters after predict m samples
    # stop training after iterate times update
    def fit(self,x_train,y_train,max_iterate=10000,j_mini=1e-4,mini_batch_size=10,learn_rate=0.01,lamda=1e-8):
        if self._check_dataset(x_train,y_train) == False:
            print("error in check_dataset")
            return
        sample_size=x_train.shape[0]
        self.jcostlist.clear()
        print("Start training. Please wait...")
        start_time=time.clock()
        index = 0
        iterate=0
        m=mini_batch_size
        if m <= 0:
            m=1
        while True:
            index =index % sample_size
            jcost=self.computeJcost(x_train,y_train,index,m,lamda,backward=True)
            index += m
            # update parameters
            for lyr in self.layerlist:
                lyr.update_bias_and_weight(learn_rate=learn_rate, lamda=lamda, mini_batch_size=m)
            iterate+=1
            if iterate % 1000 == 0:
                self.jcostlist.append([iterate, jcost])
                # print("JCost:%s"%(self.jcostlist[len(self.jcostlist)-1]))
            if iterate>=max_iterate or jcost<j_mini:
                break
        end_time=time.clock()
        print("Info of JCost:%s" % (self.jcostlist[len(self.jcostlist) - 1]))
        print("Training complete after %f second.\n"%(end_time-start_time))
        # training complete

    # equal to function fit, without using computeJcost()
    # train cur network
    # update parameters after predict m samples
    # stop training after iterate times update
    def fit2(self, x_train, y_train, max_iterate=10000, j_mini=1e-4, m=10, learn_rate=0.01, lamda=0.01):
        if self._check_dataset(x_train, y_train) == False:
            print("error in check_dataset")
            return
        times = 0
        sample_size = x_train.shape[0]
        jcost = 0.0
        self.jcostlist.clear()
        print("Start training. Please wait...")
        start_time = time.clock()
        while True:
            if m <= 0:
                m = 1
                index = np.random.random_integers(low=0, high=sample_size - 1)
            else:
                index = times % sample_size
            # gradient check

            self._forward(x_train[index])
            jcost += self._computejcostxy(self.outputlayer.active, y_train[index])

            self._backward(y_train[index])
            times += 1
            # give an Jcost and update parameters
            if times % m == 0:
                # give an Jcost
                jcost = jcost / m
                jcost += self._computeJpunish(lamda)
                # store cost to a list
                iterate_times = times / m
                if iterate_times % 1000 == 0:
                    self.jcostlist.append([iterate_times, jcost])
                    # print("JCost:%s"%(self.jcostlist[len(self.jcostlist)-1]))

                # update parameters
                for lyr in self.layerlist:
                    lyr.update_bias_and_weight(learn_rate=learn_rate, lamda=lamda, mini_batch_size=m)
                # qualified trained
                if jcost <= j_mini:
                    break
                jcost = 0.0
            if iterate_times >= max_iterate:
                break
        end_time = time.clock()
        print("Info of JCost:%s" % (self.jcostlist[len(self.jcostlist) - 1]))
        print("Training complete after %f second.\n" % (end_time - start_time))
        # training complete

    #gradient descent algorithm check before fit a trainint_set
    def gradient_check(self, x_train, y_train, epsilon=1e-4, m=10, lamda=0.01):
        if self._check_dataset(x_train, y_train) == False:
            print("Error in check_dataset")
            return
        # random select a bias of a cell and one of its weights
        # i is layer index,input layer not included
        # j is cell index in layer[i]
        # k is cell index in layer[i-1]
        # we observe the weight of cell k in layer[i-1] to cell j in layer[i]
        # if k == size of layer[i-1]
        # we observe the bias of cell j in layer[i]
        # then k refers to the bias of cell of layer[i]
        i = np.random.random_integers(1, len(self.layerlist) - 1)
        j = np.random.random_integers(0, self.layerlist[i].size - 1)
        k = np.random.random_integers(0, self.layerlist[i - 1].size - 1)
        e = epsilon
        jcost_bias_plus = 0.0
        jcost_bias_minus = 0.0
        jcost_w_plus=0.0
        jcost_w_minus=0.0

        # observe1=self.layerlist[i].bias[j]
        # observe2=self.layerlist[i].con.weight[j,k]
        times = 0
        sample_size = x_train.shape[0]
        if m<1:
            m=1
        print("start checking code of gradient decent algorithm...")
        while True:
            index = times % sample_size
            # index=np.random.random_integers(low=0,high=3)
            # compute J(bias+epsilon) and J(bias-epsilon) for bias
            self.layerlist[i].bias[j] += e
            self._forward(x_train[index])
            jcost_bias_plus += self._computejcostxy(self.outputlayer.active, y_train[index])

            self.layerlist[i].bias[j] -= 2 * e
            self._forward(x_train[index])
            jcost_bias_minus += self._computejcostxy(self.outputlayer.active, y_train[index])

            # reset to its original value
            self.layerlist[i].bias[j] += e

            # compute J(weight+epsilon) and J(weight-epsilon) for bias
            self.layerlist[i].con.weight[j,k]+=e
            self._forward(x_train[index])
            jcost_w_plus+=self._computejcostxy(self.outputlayer.active, y_train[index])
            self.layerlist[i].con.weight[j, k] -= 2*e
            self._forward(x_train[index])
            jcost_w_minus+=self._computejcostxy(self.outputlayer.active, y_train[index])
            # reset weight to its original data
            self.layerlist[i].con.weight[j,k]+=e

            self._forward(x_train[index])
            self._backward(y_train[index])
            times += 1
            # give an Jcost and update parameters
            if times == m:
                # give an Jcost
                jcost_bias_plus /= m
                jcost_bias_minus /= m

                dj_over_dbias = (jcost_bias_plus - jcost_bias_minus)/(2*e)

                jcost_w_plus /= m
                jcost_w_minus /= m
                dj_over_dw = (jcost_w_plus-jcost_w_minus)
                weight=self.layerlist[i].con.weight[j,k]
                # only one data in weight matrixes is changed
                dj_over_dw +=0.5*lamda \
                             *(np.power((weight+e),2) - np.power((weight-e),2))
                dj_over_dw /= 2*e

                dj_over_dbias_code=self.layerlist[i].dbias[j]/m
                dj_over_dw_code=self.layerlist[i].con.dweight[j,k]/m+lamda*weight

                print("Result of gradient descent algorithm code check:")
                print("For weight from cell[%d] in layer[%d] to cell[%d] in layer[%d]:"%(k,i-1,j,i))
                print(" dJ/dw numerical = %10f"%(dj_over_dw))
                print(" dJ/dw from code = %10f"%(dj_over_dw_code))
                print("For bias of cell[%d] in layer[%d]:"%(j,i))
                print(" dJ/db numerical = %10f"%(dj_over_dbias))
                print(" dJ/db from code = %10f"%(dj_over_dbias_code))

                break
            pass
        return

    def __str__(self):
        str = "----- network: %s (%d layers)-----\n"%(self.name,len(self.layerlist))
        str += " input layer: %s\n" % (self.inputlayer.name)
        str += "output layer: %s\n" % (self.outputlayer.name)
        for l in self.layerlist:
            str += "%s"%(l)
        str += "----- end of network -----\n"
        return str
