import torch
import torchvision
import torchvision.transforms as transforms

import read_params

params = read_params.params

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# [0,1] -> [-1,1]
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))]
)

trainset = torchvision.datasets.CIFAR10(root=params['root'], train=True, download=True, transform=transform)
#testset = torchvision.datasets.CIFAR10(root=params['root'], train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=int(params['batch_size']),
                                             shuffle=True, num_workers=4, pin_memory=True)

