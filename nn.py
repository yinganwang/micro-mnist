import random
from micrograd import Value

class Neuron():
    def __init__(self, nin, activation_type='sigmoid'):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.activation_type = activation_type
    
    def __call__(self, x):
        # wx + b
        output = sum([wi * xi for wi, xi in zip(self.w, x)], self.b)
        
        if self.activation_type == 'sigmoid':
            return output.sigmoid()
        elif self.activation_type == 'relu':
            return output.relu()
        elif self.activation_type == 'tanh':
            return output.tanh()
        else:
            raise NotImplementedError
    
    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{self.activation_type} Neuron({len(self.w)})"

class Layer:
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
    
    def __call__(self, x):
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts, **kwargs):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], **kwargs) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layers in self.layers for p in layers.parameters()]
