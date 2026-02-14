# micro-mnist

A tiny neural network framework built from scratch in pure Python, inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd).

This project trains a fully connected MLP to classify MNIST digits using scalar automatic differentiation — no NumPy, no PyTorch autograd, no GPU.

## Features

- Scalar reverse-mode autodiff (Value class)
- Fully connected MLP (Multi-Layer Perceptron), with implementations of:
    - tanh, relu, softmax, and cross_entropy

## Results

73% test accuracy on 8×8 downsampled MNIST using pure Python.
