import timm
import torch.nn as nn
import torch
from meta_data import disease_encoding


class BaseModel(nn.Module):
    def __init__(self, num_classes=1000, drop_p=0., pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.model = timm.create_model('resnetv2_50x3_bitm_in21k', pretrained=pretrained, num_classes=num_classes, drop_rate=drop_p,)
            
    @torch.cuda.amp.autocast()
    def forward(self, img):
        return self.model(img)


class CropHeadClassifier(nn.Module):
    def __init__(self, num_base_features=1000, num_classes=6, drop_p=0.1):
        super().__init__()
        self.classifier = nn.Linear(num_base_features, num_classes)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop_p)
    
    @torch.cuda.amp.autocast()
    def forward(self, x):
        return self.classifier(self.act(self.drop(x)))


class CropClassifier(nn.Module):
    def __init__(self, num_base_features=1000,num_classes=6, drop_p=0.1):
        super().__init__()
        self.base_model = BaseModel(num_classes=num_base_features, drop_p=drop_p)
        self.classifier = CropHeadClassifier(num_base_features=num_base_features, num_classes=num_classes, drop_p=drop_p)
        
    def forward(self, x, not_use=None):
        return self.classifier(self.base_model(x))
    

class DiseaseHeadClassifier(nn.Module):
    def __init__(self, num_base_features=1000, num_classes=13, drop_p=0.1):
        super().__init__()
        self.classifier = nn.Linear(num_base_features, num_classes)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop_p)
        
    @torch.cuda.amp.autocast()
    def forward(self, x, mask):
        x = self.classifier(self.act(self.drop(x)))
        x.masked_fill_(mask, -10000.)
        return x


class DiseaseClassifier(nn.Module):
    def __init__(self, num_base_features=1000, num_classes=13, drop_p=0.1):
        super().__init__()
        self.base_model = BaseModel(num_classes=num_base_features, drop_p=drop_p)
        self.classifier = DiseaseHeadClassifier(num_base_features=num_base_features, num_classes=num_classes, drop_p=drop_p)
        
    def forward(self, x, mask):
        return self.classifier(self.base_model(x), mask)
     

class RiskHeadClassifier(nn.Module):
    def __init__(self, num_base_features=1000, hidden_feature=200, num_classes=4, drop_p=0.1, embedding_dim=10):
        super().__init__()
        self.linear = nn.Linear(num_base_features + embedding_dim, hidden_feature)
        self.classifier = nn.Linear(hidden_feature, num_classes)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop_p)
        self.embedding = nn.Embedding(num_embeddings=len(disease_encoding), embedding_dim=embedding_dim)
    
    @torch.cuda.amp.autocast()
    def forward(self, x, disease_code, mask):
        x = self.act(self.drop(x))
        y = self.drop(self.embedding(disease_code))
        concat = torch.cat([x, y], dim=1)
        out = self.act(self.drop(self.linear(concat)))
        out = self.classifier(out)
        out.masked_fill_(mask, -10000.)
        return out

        
class RiskClassifier(nn.Module):
    def __init__(self, num_base_features=1000, hidden_feature=200, num_classes=4, drop_p=0.1, embedding_dim=100):
        super().__init__()
        self.base_model = BaseModel(num_classes=num_base_features, drop_p=drop_p)
        self.classifier = RiskHeadClassifier(num_base_features=num_base_features, hidden_feature=hidden_feature, num_classes=num_classes, drop_p=drop_p, embedding_dim=embedding_dim)
        
    def forward(self, x, disease_code, mask):
        return self.classifier(self.base_model(x), disease_code, mask)