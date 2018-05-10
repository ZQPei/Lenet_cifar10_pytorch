import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from read_data import params, train_loader
from Net import Net

torch.manual_seed(int(params['seed']))
device = torch.device('cuda' if params['CUDA'] == 'True' and torch.cuda.is_available() else 'cpu')
net = Net()
print(torch.cuda.device_count())

# parallel using 4 gpus at the same time
if torch.cuda.device_count() >1:
    net = nn.DataParallel(net)
net.to(device)
print(net)

#import ipdb; ipdb.set_trace()
# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=float(params['learning_rate']), momentum=float(params['momentum']))

def train(epoch):
    running_loss = 0.
    for i,data in enumerate(train_loader, 1):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i%2000 == 0:
            print('[epoch:{}, round:{:5d}]\tloss:{:.3f}'.format(epoch, i, running_loss/2000))
            running_loss = 0.


if __name__ == '__main__':
    for epoch in range(1, int(params['epochs'])+1):
        train(epoch)
    torch.save(net, 'model/net.pth')
