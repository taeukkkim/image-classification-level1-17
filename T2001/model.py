import torch
import torch.nn as nn
import torch.nn.functional as F

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

