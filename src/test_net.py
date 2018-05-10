import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from Net import Net


# load net
mynet = torch.load('model/net.pth')
print(mynet)

#import ipdb; ipdb.set_trace()

# show me the weight
weight_conv1 = list(mynet.parameters())[0]
weight_conv1 = (weight_conv1-weight_conv1.min())/(weight_conv1.max()-weight_conv1.min())
weight_conv1 = weight_conv1.cpu()
weight_conv1 = torchvision.utils.make_grid(weight_conv1)
weight_conv1_np = weight_conv1.detach().numpy()
weight_conv1_np = weight_conv1_np.transpose(1,2,0)

weight_conv2 = list(mynet.parameters())[2]
weight_conv2 = (weight_conv2-weight_conv2.min())/(weight_conv2.max()-weight_conv2.min())
weight_conv2 = weight_conv2.cpu()
weight_conv2 = weight_conv2.view(16*6,1,5,5)
print(weight_conv2.shape)
weight_conv2 = torchvision.utils.make_grid(weight_conv2)
weight_conv2_np = weight_conv2.detach().numpy()
print(weight_conv2_np.shape)
weight_conv2_np = weight_conv2_np.transpose(1,2,0)
#weight_conv2_np = weight_conv2_np.squeeze(1)

plt.figure()
plt.imshow(weight_conv1_np)
plt.figure()
plt.imshow(weight_conv2_np)
plt.show()


# test on my img
img = plt.imread("myimg/4.jpg")
print(img.shape)
img = img.transpose(2,1,0)
img = torch.unsqueeze(torch.from_numpy(img),0)
print(img.shape)

img = img.type(torch.float)
img = (img-img.min())/(img.max()-img.min())
img = (img-0.5)/0.5
img = img.cuda()

pred = mynet(img)
print(pred)
pred = pred.max(1)[1]

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(classes[pred])