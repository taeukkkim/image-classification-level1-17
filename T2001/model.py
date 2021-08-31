import torch
import torch.nn as nn
import torch.nn.functional as F

from facenet_pytorch import InceptionResnetV1

class InceptionResnet(InceptionResnetV1):
    def __init__(self, num_classes=18, pretrained='vggface2', classify=True):
        super().__init__()

class multilabel_dropout_IR(InceptionResnetV1):
    def __init__(self, hidden_size = 2, num_classes = 18, pretrained = 'vggface2', classify = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.high_dropout = torch.nn.Dropout(1 / hidden_size)
        self.logits = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        """Calculate embeddings or logits given a batch of input image tensors.
        Arguments:
            x {torch.tensor} -- Batch of image tensors representing faces.
        Returns:
            torch.tensor -- Batch of embedding vectors or multinomial logits.
        """
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)
        if self.classify:
            x = torch.mean(torch.stack([
            self.logits(self.high_dropout(x))
            for _ in range(self.hidden_size)
        ], dim=0), dim=0)
        else:
            x = F.normalize(x, p=2, dim=1)
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

