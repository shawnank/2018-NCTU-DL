import numpy as np
import random

class NN:
    def __init__(self, inputs, outputs):
        #3 input nodes(2+1), 3 hidden nodes(2+1), 1 output node
        self.ni=3
        self.nh=3
        self.no=1

        #initialize activation value with 1
        self.ai, self.ah, self.ao=[], [], []
        self.ai=[1.0]*self.ni
        self.ah=[1.0]*self.nh
        self.ao=[1.0]*self.no

        #initialize weights
        self.wih, self.who=[], []
        for i in range(self.ni):
            self.wih.append([0.0]*(self.nh-1))
        for j in range(self.nh):
            self.who.append([0.0]*self.no)

        #random the waight value first
        randomizeMatrix(self.wih,-0.2,0.2)
        randomizeMatrix(self.who,-2.0,2.0)

        #initialize a matrix support for momentum
        self.cih, self.cho=[], []
        for i in range(self.ni):
            self.cih.append([0.0]*self.nh)
        for j in range(self.nh):
            self.cho.append([0.0]*self.no)

    #M for momentum factor and N for learning factor
    def backpropagate(self, inputs, desired, real, N=0.5, M=0.1):
        output_deltas=[0.0]*self.no
        for k in range(self.no):
            error=desired[k]-real[k]
            output_deltas[k]=error*dsigmoid(self.ao[k])

        for j in range(self.nh):
            for k in range(self.no):
                delta_weight=self.ah[j]*output_deltas[k]
                self.who[j][k]+= M*self.cho[j][k]+N*delta_weight
                self.cho[j][k]=delta_weight

        hidden_deltas=[0.0]*self.nh
        for j in range(self.nh):
            error=0.0
            for k in range(self.no):
                error+=self.who[j][k]*output_deltas[k]
            hidden_deltas[j]=error*dsigmoid(self.ah[j])

        for i in range(self.ni):
            for j in range(self.nh-1):
                delta_weight=self.ai[i]*hidden_deltas[j]
                self.wih[i][j]+= M*self.cih[i][j]+N*delta_weight
                self.cih[i][j]=delta_weight

    def forward_propagate(self, inputs):
        
        #set the input value of NN
        for i in range(self.ni-1):
            self.ai[i]=inputs[i]
        
        #calculate the sum of inner product of activation and weight, and then send the value to next layer
        for j in range(self.nh-1):
            sum=0.0
            for i in range(self.ni):
                sum+=self.ai[i]*self.wih[i][j]
            self.ah[j]=sigmoid(sum)

        for k in range(self.no):
            sum=0.0
            for j in range(self.nh):
                sum+=self.ah[j]*self.who[j][k]
            self.ao[k]=sigmoid(sum)

        return self.ao

    def train(self, inputs, outputs, iterations):
        for i in range(iterations):
            if (i%10000)==0:
                print 'epochs: ', i
            #run the network for all inputs to get output, and backpropagate them with the desired output
            for j in range(len(inputs)):
                out = self.forward_propagate(inputs[j])
                desired=[outputs[j]]
                self.backpropagate(inputs[j],desired,out)

    def test(self, inputs, outputs):
        for i in range(len(inputs)):
            print 'Input: ', inputs[i], ' Output: ', self.forward_propagate(inputs[i])
#end of class

def randomizeMatrix(mat, a, b):
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            mat[i][j]=random.uniform(a,b)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

#derivative of sigmoid function
def dsigmoid(y):
    return y * (1-y)

if __name__ == "__main__":
    #X means input and Y means output for a xor gate
    X=np.array([[0,0],[0,1],[1,0],[1,1]])
    Y=np.array([0,1,1,0])
    #create a new neural network and train it for 500 iterations, then test it
    new_NN=NN(X,Y)
    new_NN.train(X,Y,100000)
    new_NN.test(X,Y)
