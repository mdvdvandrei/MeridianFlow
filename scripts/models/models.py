import torchvision.models as models
import torch
from torch import nn
import torch.nn as nn
import timm



def replace_activations(model, old_act, new_act):
    for name, module in model.named_children():
        if isinstance(module, old_act):
            setattr(model, name, new_act())
        else:
            replace_activations(module, old_act, new_act)

def resnext50_32x4d():
    model = models.resnext50_32x4d(pretrained=False).float()
    model.conv1 = nn.Conv2d(18, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=1)
    replace_activations(model, nn.ReLU, nn.GELU)
    return model 

def mobilenet_v2():
    model = models.mobilenet_v2(pretrained=False).float()
    model.features[0][0] = nn.Conv2d(18, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=1)
    replace_activations(model, nn.ReLU, nn.GELU)
    return model

def mobilenet_v3_large():
    model = timm.create_model('mobilenetv3_large_100', pretrained=False)
    model.blocks[0][0] = nn.Sequential(
        nn.Conv2d(18, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        nn.BatchNorm2d(32),
        nn.GELU()  
    )
    model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=1)
    return model

def mobilenet_v3_small():
    model = timm.create_model('mobilenetv3_rw', pretrained=False)
    model.conv_stem = nn.Sequential(
        nn.Conv2d(18, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        nn.BatchNorm2d(32),
        )    
    model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=1)
    replace_activations(model, nn.ReLU, nn.GELU)
    return model

def convnext_base():
    model = timm.create_model('convnext_base', pretrained=False)
    model.features[0][0] = nn.Conv2d(18, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=1)
    replace_activations(model, nn.ReLU, nn.GELU)
    return model
    
def convnext_small():
    model = timm.create_model('convnext_small', pretrained=False)
    model.features[0][0] = nn.Conv2d(18, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=1)
    replace_activations(model, nn.ReLU, nn.GELU)
    return model


def shufflenet_small():
    model = models.shufflenet_v2_x1_0(pretrained=False)
    if model is None:
        print("Failed to create model")
        return None

    model.conv1[0] = nn.Conv2d(18, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=1)
    
    if model is None:
        print("Failed to modify model")
        return None

    return model


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(18*402*934, 1)  

    def forward(self, x):
        x = self.flatten(x)
        out = self.linear(x)
        return out



if __name__ == "__main__":
    print(timm.list_models('conv*'))