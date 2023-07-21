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
    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=50)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=50)
    if training:
        return train_loader
    else:
        return test_loader


def build_model():
    """
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
        nn.Linear(64, 10),
    )
    return model


def train_model(model, train_loader, criterion, T):
    """
    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    for epoch in range(T):
        running_loss = 0.0
        count = 0
        correct = []
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            predicted = torch.max(outputs.data, 1)[1]
            total += len(labels)
            correct.append(sum(predicted == labels).item())
            count += 1
            running_loss += loss.item()

        percent = (sum(correct) / total) * 100
        print("Train Epoch: " + str(epoch) + " Accuracy: " + str(sum(correct)) + "/" + str(total) + "(" +
              str(round(percent, 2)) + "%)" + " Loss: " + str(round(running_loss / count, 3)))
    print('Finished Training')


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
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
        count = 0
        correct = []
        total = 0
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            opt.step()
            predicted = torch.max(outputs.data, 1)[1]
            total += labels.size(0)
            correct.append(sum(predicted == labels).item())
            running_loss += loss.item()
            count += 1

        percent = round((sum(correct) / total) * 100, 2)

        if show_loss:
            print("Average loss: " + str(round(running_loss / count, 4)))
        print("Accuracy: " + str(percent) + "%")


def predict_label(model, test_images, index):
    """
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
    sorted_pred = sorted(list(prob[index]), reverse = True)

    label = []
    penc = []

    for sort in sorted_pred[:3]:
        label.append(class_names[images.index(sort)])
        penc.append("%.2f%%" % (sort*100))

    for i in range(3):
        print(label[i], ":" ,penc[i])


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    # criterion = nn.CrossEntropyLoss()
    # train_loader = get_data_loader()
    # test_loader = get_data_loader(False)
    # # print(type(train_loader))
    # model = build_model()
    # # print(train_loader.dataset)
    # train_model(model, train_loader, criterion, 5)
    # evaluate_model(model, test_loader, criterion, show_loss=False)
    # evaluate_model(model, test_loader, criterion, show_loss=True)
    #
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


    # criterion = nn.CrossEntropyLoss()
    # train_loader = get_data_loader()
    # test_loader = get_data_loader(False)
    # # print(type(train_loader))
    # # print(train_loader.dataset)
    #
    # model = build_model()
    # # print(model)
    # # train_model(model, train_loader, criterion, T=5)
    # # evaluate_model(model, test_loader, criterion, show_loss=False)
    # # evaluate_model(model, test_loader, criterion, show_loss=True)
    # pred_set = iter(test_loader).next()[0]
    # predict_label(model, pred_set, 1)
