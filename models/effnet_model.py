import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

# efficientnet-b0-224 - 1280
# efficientnet-b1-240 - 1280
# efficientnet-b2-260 - 1408
# efficientnet-b3-300 - 1536
# efficientnet-b4-380 - 1792
# efficientnet-b5-456 - 2048
# efficientnet-b6-528 - 2304
# efficientnet-b7-600 - 2560


class EffNet(nn.Module):
    def __init__(self, nb_metafeatures):
        super(EffNet, self).__init__()
        self.eff_model = EfficientNet.from_pretrained(
            'efficientnet-b1', advprop=False)
        self.eff_model._fc = nn.Linear(1280, 512)
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

        image_embeddings = self.dropout(self.eff_model(image))

        metafeatures_embeddings = self.meta_model(metadata)

        features = torch.cat(
            (image_embeddings, metafeatures_embeddings), dim=1)
        output = self.cat_ouput(features)
        return output
