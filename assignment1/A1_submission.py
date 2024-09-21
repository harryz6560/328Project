"""
TODO: Finish and submit your code for logistic regression and hyperparameter search.

"""
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms

# all of the needed class and helper functions
# Logistic regression
class MNIST_LogisticRegression(nn.Module):
    def __init__(self):
        super(MNIST_LogisticRegression, self).__init__()
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.softmax(x, dim=1)


class CIFAR10_LogisticRegression(nn.Module):
    def __init__(self):
        super(CIFAR10_LogisticRegression, self).__init__()
        self.fc = nn.Linear(3*32*32, 10)
        nn.init.zeros_(self.fc.weight)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return  F.softmax(x, dim=1)

def train(epoch,data_loader,model,optimizer, device, one_hot):
    # get the nessacery variables
    log_interval = 100
    loss = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, one_hot(target,num_classes=10).float())
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(data_loader.dataset),
            100. * batch_idx / len(data_loader), loss.item()))

def eval(data_loader,model,dataset, device, one_hot):
    loss = 0
    correct = 0
    with torch.no_grad(): # notice the use of no_grad
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            loss += F.cross_entropy(output, one_hot(target,num_classes=10).float(), size_average=False).item()
    loss /= len(data_loader.dataset)
    print(loss)
    print(dataset+'set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))

def get_score(data_loader,model,dataset, device, one_hot):
    loss = 0
    correct = 0
    with torch.no_grad(): # notice the use of no_grad
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            loss += F.cross_entropy(output, one_hot(target,num_classes=10).float(), size_average=False).item()
    loss /= len(data_loader.dataset)
    return loss, (100. * correct / len(data_loader.dataset))


def logistic_regression(dataset_name, device):
    # TODO: implement logistic regression here
    if dataset_name == "MNIST":
        n_epochs = 10
        batch_size_train = 100
        batch_size_test = 1000
        learning_rate = 0.001
        weight_decay = 1e-4
        log_interval = 100

        random_seed = 1
        torch.backends.cudnn.enabled = False
        torch.manual_seed(random_seed)

        # create the model
        logistic_model = MNIST_LogisticRegression().to(device)
        # add L1 and L2 regularization, weight decary being L1 and momentum being L2
        optimizer = optim.Adam(logistic_model.parameters(), lr=learning_rate, weight_decay= weight_decay)
        one_hot = torch.nn.functional.one_hot

        # get the dataset
        training = torchvision.datasets.MNIST('/MNIST_dataset/', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

        test_set = torchvision.datasets.MNIST('/MNIST_dataset/', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

        # create a training and a validation set
        training_set, validation_set = random_split(training, [48000, 12000])

    else: # it is CIFAR10
        n_epochs = 5
        batch_size_train = 128
        batch_size_test = 1000
        learning_rate = 1e-3
        weight_decay = 1e-4
        log_interval = 100

        random_seed = 1
        torch.backends.cudnn.enabled = False
        torch.manual_seed(random_seed)

        # create the model
        logistic_model = CIFAR10_LogisticRegression().to(device)
        # add L1 and L2 regularization, weight decary being L1 and momentum being L2
        optimizer = optim.Adam(logistic_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        one_hot = torch.nn.functional.one_hot

        # get the data
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        training = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        # CIFAR-10 test set
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)

        classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # create a training and a validation set
        training_set, validation_set = random_split(training, [38000, 12000])

    print(n_epochs, batch_size_train, learning_rate, weight_decay)
    train_loader = torch.utils.data.DataLoader(training_set,batch_size=batch_size_train, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set,batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size_test, shuffle=True)

    # train the model
    eval(validation_loader, logistic_model,"Validation", device, one_hot)
    for epoch in range(1, n_epochs + 1):
        train(epoch,train_loader,logistic_model,optimizer, device, one_hot)
        eval(validation_loader,logistic_model,"Validation", device, one_hot)

    eval(test_loader,logistic_model,"Test", device, one_hot)

    results = dict(model=logistic_model)

    return results


def tune_hyper_parameter(dataset_name, target_metric, device):
    # TODO: implement logistic regression hyper-parameter tuning here
    best_metric = 0.0
    best_params = {}

    # Define hyperparameter search space
    optimization = ['Adam']
    learning_rates = [0.001, 0.01]
    num_epochs = [5, 10]
    batch_sizes = [128, 64]
    reg = 1e-4
    log_interval = 100
    batch_size_test = 1000

    for optimizer in optimization:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                for epochs in num_epochs:
                    n_epochs = epochs
                    batch_size_train = batch_size
                    batch_size_test = 1000
                    learning_rate = lr
                    weight_decay = reg
                    log_interval = 100

                    if dataset_name == "MNIST":
                        random_seed = 1
                        torch.backends.cudnn.enabled = False
                        torch.manual_seed(random_seed)

                        # create the model
                        logistic_model = MNIST_LogisticRegression().to(device)
                        # add L1 and L2 regularization, weight decary being L1 and momentum being L2
                        optimizer = optim.Adam(logistic_model.parameters(), lr=learning_rate, weight_decay= weight_decay)
                        one_hot = torch.nn.functional.one_hot

                        # get the dataset
                        training = torchvision.datasets.MNIST('/MNIST_dataset/', train=True, download=True,
                                                transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

                        test_set = torchvision.datasets.MNIST('/MNIST_dataset/', train=False, download=True,
                                                transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

                        # create a training and a validation set
                        training_set, validation_set = random_split(training, [48000, 12000])

                    else: # it is CIFAR10
                        random_seed = 1
                        torch.backends.cudnn.enabled = False
                        torch.manual_seed(random_seed)

                        # create the model
                        logistic_model = CIFAR10_LogisticRegression().to(device)
                        # add L1 and L2 regularization, weight decary being L1 and momentum being L2
                        optimizer = optim.Adam(logistic_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                        one_hot = torch.nn.functional.one_hot

                        # get the data
                        transform = transforms.Compose(
                            [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

                        training = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                                download=True, transform=transform)
                        # CIFAR-10 test set
                        test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                              download=True, transform=transform)

                        classes = ('plane', 'car', 'bird', 'cat',
                          'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

                        # create a training and a validation set
                        training_set, validation_set = random_split(training, [38000, 12000])

                    print(n_epochs, batch_size_train, learning_rate, weight_decay)
                    train_loader = torch.utils.data.DataLoader(training_set,batch_size=batch_size_train, shuffle=True)
                    validation_loader = torch.utils.data.DataLoader(validation_set,batch_size=batch_size_train, shuffle=True)
                    test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size_test, shuffle=True)

                    # train the model
                    eval(validation_loader, logistic_model,"Validation", device, one_hot)
                    for epoch in range(1, n_epochs + 1):
                        train(epoch,train_loader,logistic_model,optimizer, device, one_hot)
                        eval(validation_loader,logistic_model,"Validation", device, one_hot)

                    # Evaluate the model on the validation set
                    metric_loss, metric_acc = get_score(validation_loader, logistic_model,"Validation", device, one_hot)
                    if target_metric == "acc":
                        testing_metric = metric_acc
                    else: # is based on val loss
                        testing_metric = metric_loss

                    # Check if the current combination of hyperparameters gives a better accuracy
                    if testing_metric > best_metric:
                        best_metric = testing_metric
                        best_params = {
                            'optimizer': optimizer,
                            'learning_rate': lr,
                            'regularization_strength': reg,
                            'num_epochs': epochs,
                            'batch_size': batch_size}

    print("Best Hyperparameters:")
    print(best_params)
    print("Best best metric:", best_metric)

    return best_params, best_metric