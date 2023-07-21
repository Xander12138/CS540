import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_set = datasets.MNIST('./data', train=True, download=True,
                               transform=transform)
    test_set = datasets.MNIST('./data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=50)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=50)

    if training:
        return train_loader
    else:
        return test_loader

def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model


def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(T):
        # loop over the dataset multiple times
        running_loss = 0.0
        c = 0
        total = 0
        correct = []

        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            _, predicted = torch.max(outputs.data, 1)
            total += len(labels)

            correct.append(sum((predicted == labels)).item())
            c += 1
            running_loss += loss.item()

        # print statistics
        corr_total = sum(correct)
        pcent = (corr_total / total) * 100
        print("Train Epoch: {0} Accuracy: {1}/{2}({3}%) Loss: {4}".format(
            str(epoch),str(sum(correct)),str(total),str(round(pcent,2)),str(round(running_loss / c, 3))))
    print('Finished Training')


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    with torch.no_grad():
        running_loss = 0.0
        c = 0
        total = 0
        correct = []
        for i, data in enumerate(test_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # loss.backward()
            opt.step()

            _, predicted = torch.max(outputs.data, 1)
            total += len(labels)

            correct.append(sum((predicted == labels)).item())
            c += 1
            running_loss += loss.item()

        pcent = (sum(correct) / total) * 100

        if show_loss:
            print("Average loss: {}%".format(str(round(running_loss / c, 4))))
        print("Accuracy: {}%".format(str(round(pcent, 2))))


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    prob = F.softmax(model(test_images), dim=1)
    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

    images = list(prob[index])
    sorted_pred = sorted(images, reverse=True)

    label = [class_names[images.index(sort)] for sort in sorted_pred[:3]]
    penc = ["%.2f%%" % (sort*100) for sort in sorted_pred[:3]]

    for i in range(3):
        print(label[i], ":" ,penc[i])


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)
    model = build_model()
    # criterion = nn.CrossEntropyLoss()
    # train_model(model, train_loader, criterion, T = 5)
    # evaluate_model(model, test_loader, criterion, show_loss=True)
    # evaluate_model(model, test_loader, criterion, show_loss=False)

    # pred_set = []
    # for dat in test_loader:
    #     imgs, labels = dat
    #     pred_set.append(imgs)
    #
    # img_list = []
    # for i, data in enumerate(test_loader, 0):
    #     if i > 9:
    #         break
    #     image, labels = data
    #
    #     img_list.append(image)
    #
    # test_images = torch.cat(img_list, dim=0)
    #
    # predict_label(model, test_images, 1)
    #
    # pred_set, _ = iter(test_loader).next()
    #
    # predict_label(model, pred_set, 0)
    # predict_label(model, pred_set, 1)
    # predict_label(model, pred_set, 2)
    # predict_label(model, pred_set, 3)
    # predict_label(model, pred_set, 4)


