from datasets import load_dataset
from nn import MLP
from micrograd import Value
import numpy as np
from tqdm import tqdm
from PIL import Image


def softmax(values):
    exp_vals = [v.exp() for v in values]
    s = sum(exp_vals)
    return [v / s for v in exp_vals]

def cross_entropy(preds, target):
    sm = softmax(preds)
    return -sm[target].log()

def flatten_im(im):
    im_resized = im.resize((8, 8), Image.Resampling.LANCZOS)
    im_arr = np.array(im_resized) / 255.0
    
    return im_arr.flatten()

# exp test
x = [Value(1.0), Value(2.0)]
sm = softmax(x)
loss = -sm[0].log()
loss.backward()
print(x[0].grad, x[1].grad)

ds = load_dataset("ylecun/mnist")
train_data = ds["train"]
test_data = ds["test"]

train_imgs = [flatten_im(train_data[i]["image"]) for i in range(100)]
train_labels = [train_data[i]["label"] for i in range(100)]
test_imgs = [flatten_im(test_data[i]["image"]) for i in range(100)]
test_labels = [test_data[i]["label"] for i in range(100)]

im = train_imgs[0]
print(im.shape[0])
nin_dim = im.shape[0]

mlp = MLP(nin_dim, [32, 10], activation_type='relu')
num_epochs = 100
lr = 0.05

for epoch in tqdm(range(num_epochs)):
    total_loss = 0
    for (x, y) in zip(train_imgs, train_labels):
        y_pred = mlp(x)
        loss = cross_entropy(y_pred, y)

        # zero gradient
        for p in mlp.parameters():
            p.grad = 0
        
        loss.backward()
        for p in mlp.parameters():
            p.data -= lr * p.grad
        
        total_loss += loss.data
    
    tqdm.write(f"total_loss={total_loss}")
    print(f'Epoch {epoch}, Loss {total_loss/len(train_imgs)}')

def predict(x):
    pred = mlp([Value(p.item()) for p in x])
    # argmax
    values = [v.data for v in pred]
    return values.index(max(values))

total_correct = 0
for im, label in zip(test_imgs, test_labels):
    pred_label = predict(im)
    if pred_label == label:
        total_correct += 1

print("Test accuracy:", total_correct / len(test_labels))
