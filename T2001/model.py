import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from facenet_pytorch import InceptionResnetV1

def multi_sample_dropout(in_feature, out_feature, p=0.5, bias=True):
    return nn.Sequential(
        nn.Dropout(p),
        nn.Linear(in_feature, out_feature, bias)
    )

def multi_sample_dropout_forward(x, dropout_layer, hidden_size=2):
    return torch.mean(torch.stack([
        dropout_layer(x) for _ in range(hidden_size)], dim=0), dim=0)

class InceptionResnetV2(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        self.net = timm.create_model('inception_resnet_v2', pretrained=True)
        self.net.classif = nn.Linear(1536, num_classes)
    
    def forward(self, x):
        return self.net(x)

class MyModelBaseIRV2(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        self.net = timm.create_model('inception_resnet_v2', pretrained=True)
        for param in self.net.parameters():
            param.requires_grad_(False)
        self.logits = nn.Sequential(
            nn.Linear(1000, 4000),
            nn.ReLU(),
            nn.Linear(4000, 2000),
            nn.ReLU()
        )
        self.mldr = multi_sample_dropout(2000, num_classes, 0.25)

    def forward(self, x):
        x = self.net(x)
        x = self.logits(x)
        return multi_sample_dropout_forward(x, self.mldr, 4)

class multilabel_dropout_IR(nn.Module):
    def __init__(self, num_classes = 18):
        super().__init__()
        self.net = InceptionResnetV1(pretrained='casia-webface', classify=True)
        self.classifier1 = nn.Sequential(
            nn.Linear(10575, 8192),
            nn.ReLU(True),
            nn.Linear(8192, 8192),
            nn.ReLU(True)
        )
        self.msdo1 = multi_sample_dropout(8192, 4096, 0.5)
        self.classifier2 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
        )
        self.msdo2 = multi_sample_dropout(2048, num_classes, 0.5)

    def forward(self, x):
        x = self.net(x)
        x = self.classifier1(x)
        x = multi_sample_dropout_forward(x, self.msdo1, 2)
        x = self.classifier2(x)
        x = multi_sample_dropout_forward(x, self.msdo2, 2)
        return x
class MyModel(nn.Module):
    def __init__(self, num_classes):

        super().__init__()
        def residual_layer(input_channel, count):
            res_layer = nn.Sequential(
                nn.Conv2d(input_channel, input_channel/2, kernel_size=1),
                nn.ReLU(),
                nn.Dropout(p=0.15),
                nn.Conv2d(input_channel/2, input_channel, kernel_size=3, padding=1),
            )
            def residual_func(input_value):
                for _ in range(count):
                    input_value = F.relu(input_value + res_layer(input_value))
                return input_value
            return residual_func

        def net_layer(output_channel):
            return nn.Sequential(
                nn.Conv2d(output_channel/2, output_channel, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            )
        
        count_num = [1,2,8,8,4]
        channel = 32
        layers = []
        for count in count_num:
            channel = channel * 2
            layers.append(net_layer(channel))
            layers.append(residual_layer(channel, count))

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            *layers,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)

