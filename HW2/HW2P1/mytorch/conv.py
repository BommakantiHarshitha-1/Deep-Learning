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
                            # print(self.dW)


        self.db = np.sum(delta, axis=(0, 2))


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
        batch=x.shape[0]
        output_size_w=((x.shape[2]-self.kernel_size)//self.stride)+1
        output_size_h=((x.shape[3]-self.kernel_size)//self.stride)+1
        out=np.zeros((batch,self.out_channel,output_size_w,output_size_h))
        Result=np.zeros((self.out_channel,output_size_w,output_size_h))
        self.b=self.b.reshape(-1,1)
        self.b=np.pad(self.b,pad_width=((0,0),(output_size_h-1,0)),mode="constant",constant_values=0)

        self.b=self.b.reshape(self.out_channel,output_size_h,1)
        self.x=x
        self.input_size=x.shape[2]
        print("after reshape bias is:",self.b)
        for b in range(batch):
            for j in range(self.out_channel):
                for i in range(output_size_w):
                    for k in range(output_size_h):
                        Result[j][i][k]=np.multiply(x[b,:,i*self.stride:i*self.stride+self.kernel_size,k*self.stride:k*self.stride+self.kernel_size],self.W[j]).sum()
                    
            out[b]=Result+self.b
        return out
        raise NotImplemented

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        batch_size, out_channel, output_size_w,output_size_h = delta.shape
        for batch in range(batch_size):
            for cOut in range(self.out_channel):
                for cIn in range(self.in_channel):
                    for i in range(self.kernel_size):
                        for j in range(self.kernel_size):
                            for out in range(output_size_w):
                                for out_h in range(output_size_h):
                                    self.dW[cOut, cIn, i,j] += self.x[batch, cIn, i + self.stride * out,j+self.stride*out_h] * delta[batch, cOut, out,out_h]

        self.db = np.sum(delta, axis=(0, 2,3))
        dX = np.zeros(self.x.shape)
        for batch in range(batch_size):
            for cIn in range(self.in_channel):
                for cOut in range(self.out_channel):
                    for s in range(output_size_w):
                        for ss in range(output_size_h):
                            for k in range(self.kernel_size):
                                for kk in range(self.kernel_size):
                                    dX[batch, cIn, self.stride * s + k,self.stride*ss+kk] += delta[batch, cOut, s,ss] * self.W[cOut, cIn, k,kk]

        return dX

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
        return x.reshape( self.b, self.c* self.w)
        raise NotImplemented

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        dx = np.reshape(delta, (self.b, self.c, self.w))

        return dx
        raise NotImplemented
