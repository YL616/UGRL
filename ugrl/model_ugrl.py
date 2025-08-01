import torch
import torch.nn as nn
import torch.nn.functional as F

from .segformer_head import SegFormerHead
from . import mix_transformer

class UGRL(nn.Module):
    def __init__(self, backbone, num_classes=None, embedding_dim=256, stride=None, pretrained=None, pooling=None, vicreg_cfg=None):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride
        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        if pretrained:
            state_dict = torch.load('pretrained/'+backbone+'.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict,)

        if pooling=="gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling=="gap":
            self.pooling = F.adaptive_avg_pool2d

        self.dropout = torch.nn.Dropout2d(0.5)
        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels, embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        
        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes-1, kernel_size=1, bias=False)

        projector_dim = vicreg_cfg.projector_dim
        self.projector = nn.Sequential(
            nn.Linear(self.embedding_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_dim, projector_dim),
        )
        

    def get_param_groups(self):

        param_groups = [[], [], [], [], []]
        
        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        for param in list(self.projector.parameters()):
                param_groups[4].append(param)
                
        return param_groups


    def forward(self, x, cam_only=False, seg_detach=True,):
        _x, _ = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x
        
        seg, fused_features = self.decoder(_x)
        
        if cam_only:
            cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
            return cam_s4, None

        cls_x4 = self.pooling(_x4,(1,1))
        cls_x4 = self.classifier(cls_x4)
        cls_x4 = cls_x4.view(-1, self.num_classes-1)
 
        pooled_features = F.adaptive_avg_pool2d(fused_features, (1, 1)).flatten(1)
        z = self.projector(pooled_features)
        
        return cls_x4, seg, _x4, z
    

if __name__=="__main__":
    pretrained_weights = torch.load('pretrained/mit_b1.pth')
    ugrl = UGRL('mit_b1', num_classes=20, embedding_dim=256, pretrained=True)
    ugrl._param_groups()
    dummy_input = torch.rand(2,3,512,512)
    cls, seg, attns = ugrl(dummy_input)
    print(cls.shape, seg.shape)