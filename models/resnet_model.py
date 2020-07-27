import pretrainedmodels
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    """
    Model using ResNet for the classification.
    """

    def __init__(self, nb_metafeatures, pretrained='imagenet'):
        super(ResNet, self).__init__()
        self.res_model = pretrainedmodels.__dict__[
            'resnet50'](pretrained=pretrained)
        self.res_fc = nn.Linear(2048, 512)
        self.dropout = nn.Dropout(.1)
        self.meta_model = nn.Sequential(nn.Linear(nb_metafeatures, 512),
                                        nn.LayerNorm(512),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1),
                                        # FC layer output will have 250 features
                                        nn.Linear(512, 256),
                                        nn.LayerNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))

        self.cat_ouput = nn.Linear(512 + 256, 1)

    def forward(self, image, metadata):
        x = self.res_model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(image.shape[0], -1)
        image_embeddings = self.dropout(self.res_fc(x))

        metafeatures_embeddings = self.meta_model(metadata)

        features = torch.cat(
            (image_embeddings, metafeatures_embeddings), dim=1)
        output = self.cat_ouput(features)
        return output
