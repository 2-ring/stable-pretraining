"""IWM ViT-S/16 on ImageNet-10 (Imagenette). 20 epochs, 1 GPU, no W&B."""

import stable_pretraining as spt
from stable_pretraining.data import transforms


def iwm_train_transform():
    """Source/target views from Garrido et al. 2024, Appendix A.1.

    A shared crop and flip, then independent color jitters on each view, with
    the destructive augmentations on the source only, and finally the action
    between them. Source order follows BYOL (jitter, grayscale, blur, solarize);
    the paper doesn't give one.
    """
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((224, 224), scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToImage(),  # [0, 1] floats so solarize's 0.5 threshold is half brightness
        transforms.MultiViewTransform(
            {
                "target": transforms.Compose(
                    transforms.ColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8),
                    transforms.ToImage(**spt.data.static.ImageNet),
                ),
                "source": transforms.Compose(
                    transforms.ColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0), p=0.2),
                    transforms.RandomSolarize(threshold=0.5, p=0.2),
                    transforms.ToImage(**spt.data.static.ImageNet),
                ),
            }
        ),
        transforms.IWMAction(),
    )
