import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        self.convolutional1 = nn.Conv2d(in_channels=3, out_channels=6,
                                             kernel_size=5, stride=1, padding=0, bias=True)
        self.convolutional2 = nn.Conv2d(in_channels=6, out_channels=16,
                                             kernel_size=5, stride=1, padding=0, bias=True)
        self.maxPooling1 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        self.maxPooling2 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)

        self.linearLayer1 = nn.Linear(400, 256)
        self.linearLayer2 = nn.Linear(256, 128)
        self.linearLayer3 = nn.Linear(128, num_classes)

    def forward(self, x):
        shape_dict = {}
        # certain operations
        x = self.maxPooling1(nn.functional.relu(self.convolutional1(x)))
        shape_dict['1'] = list(x.shape)

        x = self.maxPooling2(nn.functional.relu(self.convolutional2(x)))
        shape_dict['2'] = list(x.shape)

        x = x.view(-1, 400)
        shape_dict['3'] = list(x.shape)

        x = nn.functional.relu(self.linearLayer1(x))
        shape_dict['4'] = list(x.shape)

        x = nn.functional.relu(self.linearLayer2(x))
        shape_dict['5'] = list(x.shape)

        x = self.linearLayer3(x)
        shape_dict['6'] = list(x.shape)

        out = x

        return out, shape_dict


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    model_params = 0.0
    for name, parameters in model.named_parameters():
        paraNumber = 1
        for i in range(len(parameters.size())):
            paraNumber *= parameters.size()[i]
        model_params += paraNumber
    return model_params


print(count_model_params()/1e6)