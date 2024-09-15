import torch
import torch.nn as nn
import torchvision.models as models

class EntityValueExtractor(nn.Module):
    def __init__(self, num_classes, num_units):
        super(EntityValueExtractor, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        
        self.fc1 = nn.Linear(in_features, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_value = nn.Linear(512, 1)  # Regression for the value
        self.fc_unit = nn.Linear(512, num_units)  # Classification for the unit
        
    def forward(self, x):
        x = self.base_model(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        value = self.fc_value(x)
        unit = self.fc_unit(x)
        return value, unit