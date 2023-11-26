from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torch import nn

def get_efficient_net_v2_model():

    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    # Modify the first convolutional layer to accept 1-channel input
    first_conv_layer = model.features[0][0]
    model.features[0][0] = nn.Conv2d(1, first_conv_layer.out_channels, 
                                    kernel_size=first_conv_layer.kernel_size, 
                                    stride=first_conv_layer.stride, 
                                    padding=first_conv_layer.padding, 
                                    bias=False)

    # Modify the classifier for your specific number of classes (13 in your case)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 13)

    return model
