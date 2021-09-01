import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# install facenet_pytorch
from facenet_pytorch import InceptionResnetV1

def multilabel_dropout(in_feature, out_feature, p=0.5, bias=True):
    return nn.Sequential(
        nn.Dropout(p),
        nn.Linear(in_feature, out_feature, bias)
    )

def multilabel_dropout_forward(x, dropout_layer, hidden_size=2):
    return torch.mean(torch.stack([
        dropout_layer(x) for _ in range(hidden_size)]), dim=0)

class InceptionResnetV2(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        self.net = timm.create_model('inception_resnet_v2', pretrained=True)
        self.net.classif = nn.Linear(1536, num_classes)
    
    def forward(self, x):
        return self.net(x)

class MyModelBaseIRV2(InceptionResnetV2):
    def __init__(self, num_classes=18):
        super().__init__()
        for param in self.net.parameters():
            param.requires_grad_(False)
        self.logits = nn.Sequential(
            nn.Linear(1000, 4000),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4000, 2000),
            nn.ReLU()
        )
        self.mldr = multilabel_dropout(2000, num_classes, 0.25)
    def forward(self, x):
        x = self.net(x)
        x = self.logits(x)
        return multilabel_dropout_forward(x, self.mldr, 4)

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

class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = timm.create_model('resnet18', pretrained=True)
        self.net.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.net(x)


class VGG19(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = timm.create_model('vgg19', pretrained=True)
        self.net.head.fc = nn.Linear(in_features=4096, out_features=num_classes, bias=True)


    def forward(self, x):
        return self.net(x)

class Xception(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = timm.create_model('xception', pretrained=True)
        self.net.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)


    def forward(self, x):
        return self.net(x)

class EfficientNet(nn.Module):
    def __init__(self, num_classes, version):
        '''
        verson: b0, b1, b1_pruned, b2, b2_pruned, b3, b3_pruned, b4...
        '''
        super().__init__()
        self.net = timm.create_model(f'efficientnet_{version}', pretrained=True)
        if version in ['b0','b1','b1_pruned']:
            in_features = 1280
        elif version in ['b2','b2_pruned']:
            in_features = 1408
        elif version in ['b3','b3_pruned']:
            in_features = 1536
        elif version in ['b4']:
            in_features = 1792
            
        assert in_features is not None, "version에 올바른 EfficientNet version을 입력해주세요."     

        self.net.classifier = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.net(x)

class EfficientNet_v2(nn.Module):
    def __init__(self, num_classes, version):
        '''
        verson: rw_m, rw_s
        '''
        super().__init__()
        self.net = timm.create_model(f'efficientnetv2_{version}', pretrained=True)
        if version == 'rw_s':
            in_features = 1792
        elif version == 'rw_m':
            in_features = 2152
        assert in_features is not None, "version에 올바른 EfficientNet_v2 version을 입력해주세요."   
            
        self.net.classifier = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.net(x)

class ViT(nn.Module):
    def __init__(self, num_classes, version):
        '''
        verson: 'visformer_small', 'vit_base_patch16_224', 'vit_base_patch16_224_in21k', 'vit_base_patch16_224_miil',
                'vit_base_patch16_224_miil_in21k', 'vit_base_patch16_384', 'vit_base_patch32_224', 'vit_base_patch32_224_in21k',
                'vit_base_patch32_384', 'vit_base_r50_s16_224_in21k', 'vit_base_r50_s16_384', 'vit_huge_patch14_224_in21k',
                'vit_large_patch16_224', 'vit_large_patch16_224_in21k', 'vit_large_patch16_384', 'vit_large_patch32_224_in21k', 
                'vit_large_patch32_384', 'vit_large_r50_s32_224', 'vit_large_r50_s32_224_in21k', 'vit_large_r50_s32_384',
                'vit_small_patch16_224', 'vit_small_patch16_224_in21k', 'vit_small_patch16_384', 'vit_small_patch32_224',
                'vit_small_patch32_224_in21k', 'vit_small_patch32_384', 'vit_small_r26_s32_224', 'vit_small_r26_s32_224_in21k',
                'vit_small_r26_s32_384'
        '''
        super().__init__()
        self.net = timm.create_model(f'{version}', pretrained=True)   
        self.net.head = nn.Linear(in_features=self.net.head.in_features, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.net(x)

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

# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x

