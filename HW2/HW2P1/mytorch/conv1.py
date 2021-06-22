# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)
        
        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        
        batch=x.shape[0]
        output_size=((x.shape[2]-self.kernel_size)//self.stride)+1
        out=np.zeros((batch,self.out_channel,output_size))
        Result=np.zeros((self.out_channel,output_size))
        self.b=self.b.reshape(-1,1)
        self.x=x
        self.input_size=x.shape[2]
        print("after reshape bias is:",self.b)
        for b in range(batch):
            for j in range(self.out_channel):
                for i in range(output_size):
                    Result[j][i]=np.multiply(x[b,:,i*self.stride:i*self.stride+self.kernel_size],self.W[j]).sum()
                    
           
            out[b]=Result+self.b
      
       
        
        return out
        

     # res[b]=Resu+lt[j][i]




    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
    
        batch_size, out_channel, output_size = delta.shape
        for batch in range(batch_size):
            for cOut in range(self.out_channel):
                for cIn in range(self.in_channel):
                    for i in range(self.kernel_size):
                        for out in range(output_size):
                            self.dW[cOut, cIn, i] += self.x[batch, cIn, i + self.stride * out] * delta[batch, cOut, out]

        # Calculate db
        self.db = np.sum(delta, axis=(0, 2))

        # Calculate dX
        dX = np.zeros(self.x.shape)
        for batch in range(batch_size):
            for cIn in range(self.in_channel):
                for cOut in range(self.out_channel):
                    for s in range((self.input_size - self.kernel_size)//self.stride + 1):
                        for k in range(self.kernel_size):
                            dX[batch, cIn, self.stride * s + k] += delta[batch, cOut, s] * self.W[cOut, cIn, k]

        return dX


class Conv2D():
    def __init__(self, in_channel, out_channel, kernel_size, stride=1,
                    weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)
        
        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        cache = self.W, self.b, self.stride
        n_filters, d_filter, h_filter, w_filter = W.shape
        #out_channel, in_channel, kernel_size, kernel_size=W.shape
        n_x, d_x, h_x, w_x = x.shape
        h_out = (h_x - h_filter) /self.stride + 1
        w_out = (w_x - w_filter) / self.stride + 1

        h_out, w_out = int(h_out), int(w_out)

        X_col = im2col_indices(x, h_filter, w_filter, stride=self.stride)
        W_col = W.reshape(n_filters, -1)

        out = W_col @ X_col + b
        out = out.reshape(n_filters, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)

        cache = (x, W, b, stride, padding, X_col)

        return out, cache
        
        raise NotImplemented

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        
        raise NotImplemented


class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape
        raise NotImplemented

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        
        raise NotImplemented
