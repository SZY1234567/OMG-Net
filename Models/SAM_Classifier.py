import lightning as L
import torch
import torchvision
from torchvision.transforms.functional import pil_to_tensor
from torch.nn.functional import softmax, interpolate
from torchmetrics.functional import accuracy, precision, recall, f1_score
from torchmetrics import AUROC, Accuracy, Precision, Recall, F1Score
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay, precision_recall_fscore_support, roc_auc_score, accuracy_score
from PIL import Image

class Classifier(L.LightningModule):
    def __init__(self, config, label_encoder=None):
        super().__init__()
        self.config = config
        self.loss_fcn = getattr(torch.nn, self.config["BASEMODEL"]["Loss_Function"])()
        if self.config['BASEMODEL']['Loss_Function'] == 'CrossEntropyLoss':
            if "weights" in self.config['DATA']:
                w = torch.tensor(self.config['DATA']['weights'], dtype=torch.float32)
            else:
                w = torch.ones(self.config['DATA']['Num_of_Classes'], dtype=torch.float32)
            self.loss_fcn = torch.nn.CrossEntropyLoss(weight=w,
                                                      label_smoothing=self.config['REGULARIZATION']['Label_Smoothing'])

        self.LabelEncoder = label_encoder
        self.activation = getattr(torch.nn, self.config["BASEMODEL"]["Activation"])()
        backbone = getattr(torchvision.models, self.config['BASEMODEL']['Backbone'])
        self.backbone = backbone(weights='DEFAULT')
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, self.config['DATA']['Num_of_Classes'])
        self.mask_encoder = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=True)
        #self.mask_encoder.bias.data = torch.ones(self.mask_encoder.bias.data.shape)
        #self.mask_encoder.weight.data = torch.ones(self.mask_encoder.weight.data.shape)
        self.encoder_4d = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.save_hyperparameters()

    def forward(self, data):
        x = self.backbone.conv1(data['img'])
        if self.config['BASEMODEL']['Mask_Input']:
            if self.config['BASEMODEL']['Input_Type'] == "3_Channel":
                x = x + self.mask_encoder(data['msk'])
            elif self.config['BASEMODEL']['Input_Type'] == "4_Channel":
                x = self.encoder_4d(torch.cat([data['img'], data['msk']], dim=1))

        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = torch.mean(torch.mean(x, dim=2), dim=2)
        # x = x.squeeze()
        x = self.backbone.fc(x)
        x = self.activation(x)

        return x

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output = softmax(self(batch), dim=1)
        return output, batch['coords'], batch['id']
        # return self.all_gather(output), self.all_gather(batch['coords']), self.all_gather(batch['id'])



