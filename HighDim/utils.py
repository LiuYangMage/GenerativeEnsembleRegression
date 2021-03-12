import tensorflow as tf
import numpy as np

def swish(x):
    return x * tf.nn.sigmoid(x)

class NeuralNet(object):
    def __init__(self, layer_dims, name, dtype, act):
        self.layer_dims = layer_dims
        self.dtype = dtype
        self.act = act
        self.L = len(layer_dims)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self.W = [tf.get_variable('W_{}'.format(l), [layer_dims[l-1], layer_dims[l]], 
                dtype=dtype, initializer=tf.contrib.layers.xavier_initializer()) for l in range(1, self.L)]
            self.b = [tf.get_variable('b_{}'.format(l), [layer_dims[l]], 
                dtype=dtype, initializer=tf.zeros_initializer()) for l in range(1, self.L)]


    def __call__(self, x):
        A = x
        for i in range(self.L-2):
            A = self.act(tf.matmul(A, self.W[i]) + self.b[i])
        return tf.matmul(A, self.W[-1]) + self.b[-1]

    def trainable_variables(self):
        return self.W + self.b



class ResNet(object):
    def __init__(self, layer_dims, name, dtype, act):
        self.layer_dims = layer_dims
        self.dtype = dtype
        self.act = act

        self.L = len(layer_dims)
        with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
            self.W = [tf.get_variable('W_{}'.format(l), [layer_dims[l-1], layer_dims[l]], dtype=dtype, 
                                      initializer=tf.contrib.layers.xavier_initializer()) for l in range(1,self.L)]
            self.b = [tf.get_variable('b_{}'.format(l), [layer_dims[l]], dtype = dtype, 
                                     initializer=tf.zeros_initializer()) for l in range(1,self.L)]


    def __call__(self, x):
        A = x
        A = self.act(tf.matmul(A, self.W[0]) + self.b[0])

        for i in range(1, self.L-2):
            if np.mod(i,2)==1:
                B = A  # record the value into the residual block
                A = self.act(tf.matmul(A, self.W[i]) + self.b[i])
            else:
                A = self.act(tf.matmul(A, self.W[i]) + self.b[i] + B)
        return tf.matmul(A, self.W[-1]) + self.b[-1]
    
    def trainable_variables(self):
        return self.W + self.b


class ODENet(object):
    def __init__(self, vnet, steps = 20, T = 1):
        self.v = vnet
        self.steps = steps
        self.T = T
        self.dt = T/steps
        self.dims = self.v.layer_dims[-1]

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x, tau = None):
        u = [None for i in range(self.steps + 1)]
        u[0] = x
        tt = tf.ones_like(x)[:,:1]
        taus = [] if (tau is None) else [tau]
        for i in range(self.steps):
            currentv = self.v(tf.concat(taus + [u[i], tt*self.dt*i], axis = -1))
            midu = u[i] + 0.5 * self.dt * currentv
            midv = self.v(tf.concat(taus + [midu, tt*self.dt*(i+0.5)], axis = -1))
            u[i+1] = u[i] + self.dt * midv
            print(i, end = " ", flush = True)
        print(' ', flush = True)
        return u[-1]
    
    def inverse(self, x, tau = None):
        u = [None for i in range(self.steps + 1)]
        u[0] = x
        tt = tf.ones_like(x)[:,:1]
        taus = [] if (tau is None) else [tau]
        for i in range(self.steps):
            currentv = self.v(tf.concat(taus + [u[i], tt*(self.T-self.dt*i)], axis = -1))
            midu = u[i] - 0.5 * self.dt * currentv
            midv = self.v(tf.concat(taus + [midu, tt*(self.T-self.dt*(i+0.5))], axis = -1))
            u[i+1] = u[i] - self.dt * midv
            print(i, end = " ", flush = True)
        print(' ', flush = True)
        return u[-1]
    
    def trainable_variables(self):
        return self.v.trainable_variables()

