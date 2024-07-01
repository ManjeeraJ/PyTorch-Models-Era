from albumentations import (
    Compose,
    Normalize,
    RandomCrop,
    PadIfNeeded,
    CoarseDropout,
    Sequential,
    OneOf
)
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np


class Transforms:
    def __init__(self, means, stds, train=True):
        if train:
            self.transformations = Compose(
                [
                    OneOf(
                        [
                            Sequential(
                                [
                                    PadIfNeeded(
                                        min_height=40, min_width=40, always_apply=True
                                    ),
                                    RandomCrop(height=32, width=32, always_apply=True),
                                ],
                                p=1,  # The OneOf block normalizes the probabilities of all augmentations inside it, so their probabilities sum up to 1. In this case, the probability of each of the sequential blocks is 0.5
                            ),
                            Sequential(
                                [
                                    CoarseDropout(
                                        max_height=16,
                                        max_width=16,
                                        min_height=16,
                                        min_width=16,
                                        min_holes=1,
                                        max_holes=1,
                                        fill_value=means,
                                        always_apply=True,
                                    ),
                                ],
                                p=1,  # The OneOf block normalizes the probabilities of all augmentations inside it, so their probabilities sum up to 1. In this case, the probability of each of the sequential blocks is 0.5
                            ),
                        ],
                        p=1,  # Probability of applying the OneOf block
                    ),
                    Normalize(mean=means, std=stds, always_apply=True),
                    ToTensorV2(),
                ]
            )
        else:
            self.transformations = Compose(
                [
                    Normalize(mean=means, std=stds, always_apply=True),
                    ToTensorV2(),
                ]
            )

    def __call__(self, img):
        return self.transformations(image=np.array(img))["image"]
