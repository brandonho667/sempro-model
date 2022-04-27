import torch
import torchvision.models as models

class SEMPro(torch.nn.Module):
    def __init__(self):
        super(SEMPro,self).__init__()
        self.cnn_block = torch.nn.Sequential(
            torch.nn.Conv2d(1,16,kernel_size=3,stride=2),
            torch.nn.ELU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16,8,kernel_size=3,stride=2),
            torch.nn.ELU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(8)         
        )
        
        self.linear_layer = torch.nn.Sequential(
            torch.nn.LazyLinear(64),
            torch.nn.ELU(inplace=True),
            torch.nn.BatchNorm1d(64),
            torch.nn.Linear(64,32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(32,1)
        )
    
    def forward(self, x):
        x = self.cnn_block(x)
#         print(x.shape)
        x = x.view(x.size(0),-1)
#         print(x.shape)
        x = self.linear_layer(x)
        return x
    
    
class SEMPro_resNext(torch.nn.Module):
    def __init__(self, fc_size=2048, large = False, pretrained = False):
        super(SEMPro_resNext,self).__init__()
        if large:
            self.model = models.resnext101_32x8d(pretrained=pretrained)
        else:
            self.model = models.resnext50_32x4d(pretrained=pretrained)
        self.model.fc = torch.nn.Linear(fc_size,1)
        # Change relu to elu
        self.replace_layers(self.model, torch.nn.ReLU, torch.nn.ELU()) # Replace with ELU activations
    def forward(self,x):
        x = self.model(x)
        return x
    ## From https://stackoverflow.com/questions/58297197/how-to-change-activation-layer-in-pytorch-pretrained-module
    def replace_layers(self, model, old, new):
        for n, module in model.named_children():
            if len(list(module.children())) > 0:
                ## compound module, go inside it
                self.replace_layers(module, old, new)
            if isinstance(module, old):
                ## simple module
                setattr(model, n, new)
                
class SEMPro_denseNet(torch.nn.Module):
    def __init__(self, fc_size=2048, size = 0,  pretrained = False):
        super(SEMPro_resNext,self).__init__()
        if size == 0:
            self.model = models.densenet121(pretrained=pretrained)
        elif size == 1:
            self.model = models.densenet169(pretrained=pretrained)
        elif size == 2:
            self.model = models.densenet201(pretrained=pretrained)
        elif size == 3:
            self.model = models.densenet161(pretrained=pretrained)
        else:
            print("Invalid size specified, defaulting to densenet121")
            self.model = models.densenet121(pretrained=pretrained)
        self.model.fc = torch.nn.Linear(fc_size,1)
        # Change relu to elu
        self.replace_layers(self.model, torch.nn.ReLU, torch.nn.ELU()) # Replace with ELU activations
    def forward(self,x):
        x = self.model(x)
        return x
    ## From https://stackoverflow.com/questions/58297197/how-to-change-activation-layer-in-pytorch-pretrained-module
    def replace_layers(self, model, old, new):
        for n, module in model.named_children():
            if len(list(module.children())) > 0:
                ## compound module, go inside it
                self.replace_layers(module, old, new)
            if isinstance(module, old):
                ## simple module
                setattr(model, n, new)
                
class SEMPro_ConvNext(torch.nn.Module):
    def __init__(self, fc_size=2048, size = 0, pretrained = False):
        super(SEMPro_resNext,self).__init__()
        if size == 0:
            self.model = models.convnext_tiny(pretrained=pretrained)
        elif size == 1:
            self.model = models.convnext_small(pretrained=pretrained)
        elif size == 2:
            self.model = models.convnext_base(pretrained=pretrained)
        elif size == 3:
            self.model = models.convnext_large(pretrained=pretrained)
        else:
            print("Invalid size specified, defaulting to convnext_tiny")
            self.model = models.convnext_tiny(pretrained=pretrained)
        self.model.fc = torch.nn.Linear(fc_size,1)
    def forward(self,x):
        x = self.model(x)
        return x