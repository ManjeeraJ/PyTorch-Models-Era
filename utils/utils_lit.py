import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from functools import reduce
from typing import Union
from torch import nn

def get_misclassified_images(model, device):
    
    # set model to evaluation mode
    print ("made a change")
    model.model.eval()

    images = []
    predictions = []
    labels = []

    with torch.no_grad():
        for data, target in model.test_dataloader():
            data, target = data.to(device), target.to(device)

            output = model(data)

            _, preds = torch.max(output, 1)  # Perform torch.max along dimension 1 

            for i in range(len(preds)):
                if preds[i] != target[i]:
                    images.append(data[i])
                    predictions.append(preds[i])
                    labels.append(target[i])

    return images, predictions, labels

"""
Declared here and in utils.py, need to think over it 
"""
def denormalize(img):
    """
    The transpose operation np.transpose(img, (1, 2, 0)) is used to reorder the axes of the image array from (C, H, W) to (H, W, C).
    This is necessary for compatibility with image display and processing functions that expect the latter format.
    """
    channel_means = (0.4914, 0.4822, 0.4465)
    channel_stdevs = (0.2470, 0.2435, 0.2616)
    img = img.astype(dtype=np.float32)
    # print (img.shape)  # (3, 32, 32)

    for i in range(img.shape[0]):
        img[i] = (img[i] * channel_stdevs[i]) + channel_means[i]

    return np.transpose(img, (1, 2, 0))

"""
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5)
        )
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

get_module_by_name(model, "layer1") ===> Sequential(
  (0): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
  (1): ReLU()
  (2): Conv2d(10, 20, kernel_size=(5, 5), stride=(1, 1))
)
"""
def get_module_by_name(module: Union[torch.Tensor, nn.Module], access_string: str):
    """Retrieve a module nested in another by its access string.

    Works even when there is a Sequential in the module.
    """
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)

"""
https://github.com/jacobgil/pytorch-grad-cam?tab=readme-ov-file
1. How to get target layer(s)'s string repre - https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/4
"""
def show_grad_cam(
    model,
    images,
    labels,
    predictions,
    target_layer,
    classes,
    device,
    use_cuda=True,
):
    """
    model = model,
    device = device,
    images = input images
    labels = correct classes for the images
    predictions = predictions for the images. If the desired gradcam is for the correct classes, pass labels here.
    target_layer = string representation of layer e.g. "layer3.1.conv2"
    classes = list of class labels
    """
    if isinstance(target_layer, str):
    # If you pass a list with several layers, the CAM will be averaged accross them. This can be useful if you're not sure what layer will perform best.
        target_layers = [get_module_by_name(model, target_layer)]
    else:
        target_layers = [model.layer2[target_layer]]

    # Construct the CAM object once, and then re-use it on many images
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)

    fig = plt.figure(figsize=(32, 32))

    plot_idx = 1
    for i in range(len(images)):
        input_tensor = images[i].unsqueeze(0).to(device)
        # We have to specify the target we want to generate
        # the Class Activation Maps for.
        # If targets is None, the highest scoring category
        # will be used for every image in the batch.
        # Here we use ClassifierOutputTarget, but you can define your own custom targets
        # That are, for example, combinations of categories, or specific outputs in a non 
        targets = [ClassifierOutputTarget(predictions[i])]
        rgb_img = denormalize(images[i].cpu().numpy().squeeze())
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Layout = 6 images per row - 2 * (original image, gradcam and visualization)
        ax = fig.add_subplot(len(images) // 2, 6, plot_idx, xticks=[], yticks=[])
        ax.imshow(rgb_img, cmap="gray")
        ax.set_title("True class: {}".format(classes[labels[i]]))
        plot_idx += 1

        ax = fig.add_subplot(len(images) // 2, 6, plot_idx, xticks=[], yticks=[])
        ax.imshow(grayscale_cam, cmap="gray")
        ax.set_title("GradCAM Output\nTarget class: {}".format(classes[predictions[i]]))
        plot_idx += 1

        ax = fig.add_subplot(len(images) // 2, 6, plot_idx, xticks=[], yticks=[])
        ax.imshow(visualization, cmap="gray")
        ax.set_title("Visualization\nTarget class: {}".format(classes[predictions[i]]))
        plot_idx += 1

    plt.tight_layout()
    plt.show()