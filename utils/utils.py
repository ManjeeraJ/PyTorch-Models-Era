import torch
import matplotlib.pyplot as plt
import numpy as np

def get_device():

    is_cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda_available else "cpu")
    return is_cuda_available, device

def show_lr_history(lr_history, epochs):
    fig, ax = plt.subplots()

    linspace = np.linspace(0, epochs, len(lr_history))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.plot(linspace, lr_history)
    ax.tick_params(axis="y", labelleft=True, labelright=True)

    # fig.set_size_inches(30, 10)
    plt.tight_layout()
    plt.show()

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

# Another way of describing function. Docstring can be added as well. 
def show_sample_images(
    loader,
    classes,
    num_images: int=12,
    label: str=""
  ) -> None:

    images, labels = next(iter(loader))
    images = images[0:num_images]
    labels = labels[0:num_images]

    fig = plt.figure(figsize=(20, 10))
    for i in range(num_images):
        sub = fig.add_subplot(count // 5, 5, i + 1)
        # print (batch_data[i].shape)  # torch.Size([3, 32, 32])
        npimg = denormalize(images[i].cpu().numpy().squeeze())
        # print (batch_label[i])  # tensor(3), no need to add .item()
        plt.imshow(npimg, cmap="gray")  # The cmap parameter is not required for RGB images since the color channels provide the necessary color information.
        sub.set_title("Correct class: {}".format(classes[labels[i]]))
    plt.title(label)
    plt.tight_layout()
    plt.show()

def show_misclassified_images(images, predictions, labels, classes):
    fig = plt.figure(figsize=(20, 10))
    for i in range(len(images)):
        sub = fig.add_subplot(len(images) // 5, 5, i + 1)
        image = images[i]
        npimg = denormalize(image.cpu().numpy().squeeze())
        plt.imshow(npimg, cmap="gray")
        predicted = classes[predictions[i]]
        correct = classes[labels[i]]
        sub.set_title(
            "Correct class: {}\nPredicted class: {}".format(correct, predicted)
        )
    plt.tight_layout()
    plt.show()

"""
axis="y": This specifies that the customization should apply to the y-axis ticks. If you wanted to customize the x-axis ticks, you would use axis="x".

labelleft=True: This parameter ensures that the tick labels are displayed on the left side of the y-axis.

labelright=True: This parameter ensures that the tick labels are also displayed on the right side of the y-axis.
"""
def show_losses_and_accuracies(trainer, tester, epochs):
    fig, ax = plt.subplots(2, 2)

    train_epoch_loss_linspace = np.linspace(0, epochs, len(trainer.train_losses))
    test_epoch_loss_linspace = np.linspace(0, epochs, len(tester.test_losses))
    train_epoch_acc_linspace = np.linspace(0, epochs, len(trainer.train_accuracies))
    test_epoch_acc_linspace = np.linspace(0, epochs, len(tester.test_accuracies))

    ax[0][0].set_xlabel("Epoch")
    ax[0][0].set_ylabel("Train Loss")
    ax[0][0].plot(train_epoch_loss_linspace, trainer.train_losses)
    ax[0][0].tick_params(axis="y", labelleft=True, labelright=True)

    ax[0][1].set_xlabel("Epoch")
    ax[0][1].set_ylabel("Test Loss")
    ax[0][1].plot(test_epoch_loss_linspace, tester.test_losses)
    ax[0][1].tick_params(axis="y", labelleft=True, labelright=True)

    ax[1][0].set_xlabel("Epoch")
    ax[1][0].set_ylabel("Train Accuracy")
    ax[1][0].plot(train_epoch_acc_linspace, trainer.train_accuracies)
    ax[1][0].tick_params(axis="y", labelleft=True, labelright=True)
    ax[1][0].yaxis.set_ticks(np.arange(0, 101, 5))

    ax[1][1].set_xlabel("Epoch")
    ax[1][1].set_ylabel("Test Accuracy")
    ax[1][1].plot(test_epoch_acc_linspace, tester.test_accuracies)
    ax[1][1].tick_params(axis="y", labelleft=True, labelright=True)
    ax[1][1].yaxis.set_ticks(np.arange(0, 101, 5))

    fig.set_size_inches(30, 10)
    plt.tight_layout()
    plt.show()