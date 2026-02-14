import math

class Value():
    '''
    A single scalar value and its gradient. 
    '''
    def __init__(self, data, children=[], label=''):
        self.data = data
        self.label = label

        self.grad = 0
        self._backward = lambda: None
        self._prev = children.copy()
    
    def __rmul__(self, other):
        return self * other
    def __radd__(self, other):
        return self + other
    def __sub__(self, other):
        return self + (-other)
    def __neg__(self):
        return self * -1
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1
    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other * self ** -1
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        output = Value(data=self.data + other.data, children=[self, other], label='+')

        def _backward():
            '''
            z = x + y
            dz/dx = dz/dy = 1 in the addition case

            output.grad = dL/dz
            By the chain rule, we have:
            self.grad = dL/dx = dL/dz * (dz/dx)
            other.grad = dL/dy = dL/dz * (dz/dy)
            '''
            self.grad += output.grad * 1
            other.grad += output.grad * 1
        output._backward = _backward

        return output
    
    def __mul__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        output = Value(data=self.data * other.data, children=[self, other], label='*')

        def _backward():
            '''
            z = x * y
            dz/dx = y, dz/dy = x
            '''
            self.grad += output.grad * other.data
            other.grad += output.grad * self.data
        output._backward = _backward

        return output
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))  # pow value is a scalar (int or float) here.
        output = Value(data=self.data ** other, children=[self], label=f'**{other}')

        def _backward():
            self.grad +=  output.grad * (other * self.data ** (other - 1))
        output._backward = _backward

        return output
    
    def relu(self):
        output = Value(data=0 if self.data < 0 else self.data, children=[self], label='relu')

        def _backward():
            self.grad += output.grad * (self.data > 0)
        output._backward = _backward

        return output

    def tanh(self):
        output = Value(data=math.tanh(self.data), children=[self], label='tanh')

        def _backward():
            self.grad += output.grad * (1 - output.data ** 2)
        output._backward = _backward

        return output

    def sigmoid(self):
        sigmoid = 1 / (1 + math.exp(-self.data))
        output = Value(data=sigmoid, children=[self], label='sigmoid')

        def _backward():
            self.grad += output.grad * (output.data * (1 - output.data))
        output._backward = _backward

        return output

    def log(self):
        output = Value(math.log(self.data), children=[self], label='log')
        def _backward():
            self.grad += output.grad / self.data
        output._backward = _backward
        return output
    
    def exp(self):
        output = Value(math.exp(self.data), children=[self], label='exp')
        def _backward():
            self.grad += output.grad * math.exp(self.data)
        output._backward = _backward
        return output

    def backward(self):
        '''
        topological sort of the DAG nodes
        '''
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()
