import torch
import segmentation_models_pytorch as smp

def build_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,   # we're loading OUR fine-tuned weights, not ImageNet again
        in_channels=12,
        classes=1,
        activation=None,
    )
    return model

def load_model(weights_path, device):
    model = build_model()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model