import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# install facenet_pytorch
from facenet_pytorch import InceptionResnetV1

def multi_sample_dropout(in_feature, out_feature, p=0.5, bias=True):
    return nn.Sequential(
        nn.Dropout(p),
        nn.Linear(in_feature, out_feature, bias)
    )

def multi_sample_dropout_forward(x, dropout_layer, hidden_size=2):
    return torch.mean(torch.stack([
        dropout_layer(x) for _ in range(hidden_size)]), dim=0)

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
            nn.Dropout(),
            nn.Linear(4000, 2000),
            nn.ReLU()
        )
        self.mldr = multi_sample_dropout(2000, num_classes, 0.25)
    def forward(self, x):
        x = self.net(x)
        x = self.logits(x)
        return multi_sample_dropout_forward(x, self.mldr, 4)

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

class InceptionResnet(nn.Module):
    def __init__(self, num_classes=18, pretrained='vggface2'):
        super().__init__()
        self.net = InceptionResnetV1(pretrained=pretrained)
        self.net.logits = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.net(x)

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
class Resnet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('resnet50', pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


class Resnet18_multi(nn.Module):
    '''
    multi-label classification model 
    mask, gender, age =  Resnet18_multi(x)
    https://discuss.pytorch.org/t/modify-resnet50-to-give-multiple-outputs/46905 참고
    '''
    def __init__(self, mask_num=3, gender_num=2, age_num=3):
        super().__init__()
        self.model = timm.create_model('resnet18', pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.model.fc1 = nn.Linear(num_ftrs, mask_num)
        self.model.fc2 = nn.Linear(num_ftrs, gender_num)
        self.model.fc3 = nn.Linear(num_ftrs, age_num)

    def forward(self, x):
        x = self.model(x)
        mask_out = self.model.fc1(x)
        gender_out = self.model.fc2(x)
        age_out = self.model.fc3(x)
        return mask_out, gender_out, age_out



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

