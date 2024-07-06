'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
"""
PATH_DATASETS is a variable that specifies the directory where your dataset is stored or should be downloaded to. 
os.environ.get ("PATH_DATASETS", ".") - This line retrieves the value of the environment variable PATH_DATASETS.
If PATH_DATASETS is not set in the environment, it defaults to the current directory (".").
"""
import os
PATH_DATASETS = os.environ.get ("PATH_DATASETS", ".")
from torchmetrics.functional import accuracy
from torchvision import datasets
from torch.utils.data import random_split
from utils.transforms import Transforms
from torch.utils.data import DataLoader


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

"""
1.     def forward(self,x):
        return self.model(x)
       forward is a method in the class - self.model. How is it invoked automatically when you pass a tensor i.e x it it?
       1. In PyTorch, when you create a subclass of nn.Module (like Net in this case), you define a forward method that specifies the forward pass of the network. PyTorch uses some special methods to ensure that the forward method is automatically invoked when you call the model instance with an input tensor.
       2. __call__ method: In Python, the __call__ method allows an instance of a class to be called as a function. The nn.Module class defines a __call__ method that does several things, including calling the forward method.
       3. Calling the model instance: When you call model(dummy_input), you are actually invoking the __call__ method of the nn.Module class. This method handles pre- and post-processing tasks, such as hooks and registering layers, and ultimately calls the forward method.
       4. Similary thing is happening in 'logits = self(x)'

1. In PyTorch, the nn.Module class (which Net and LitResnet both inherit from) has a built-in mechanism to recursively collect all the parameters from all submodules when you call self.parameters(). This includes not just the immediate parameters defined in the class, but also those of any submodules defined within the class.
2. When you call self.parameters() in a PyTorch nn.Module (or a subclass of it, like LightningModule), it recursively iterates over all the submodules and collects their parameters.
3. In the first example, LitResnet has a submodule self.model which is an instance of Net. Here's how PyTorch collects the parameters:
    1. Submodules in nn.Module: When you define a submodule like self.model = Net() inside a nn.Module or its subclass, nn.Module's internal mechanisms register this submodule.
    2. Recursive Collection: The parameters() method in nn.Module recursively looks for all nn.Module instances that are attributes of the current module. When it finds self.model, it calls self.model.parameters() internally to collect the parameters of Net.
    3. Combining Parameters: The parameters() method then combines these parameters with any parameters defined directly in the parent module (in this case, LitResnet).
        1. Module Initialization: self.model = Net() inside LitCustomResnet.__init__(). Net is a subclass of nn.Module, so it has its own parameters. These parameters are registered with self.model.
        2. Calling self.parameters(): Inside configure_optimizers: self.parameters() in LitCustomResnet triggers a recursive search. It finds self.model, which is an instance of Net. It calls self.model.parameters(), which collects all parameters of Net.

1. The self.hparams.lr in your LitResnet class refers to the hyperparameters that are saved when you call self.save_hyperparameters() in the __init__ method. Here’s how it works:
2. The save_hyperparameters() method in PyTorch Lightning automatically saves the hyperparameters passed to the __init__ method into a hparams attribute. When you call self.save_hyperparameters(), PyTorch Lightning inspects the __init__ method and saves all the arguments (except self) into self.hparams.
3. In your case, lr and batch_size are saved into self.hparams.
4. In your LitResnet class, it is somewhat redundant to save self.BATCH_SIZE separately if you are already using self.save_hyperparameters(). You can simply use self.hparams.batch_size instead of self.BATCH_SIZE
"""
class Lit_CIFAR10_Resnet18(LightningModule):
    def __init__(self, data_dir = PATH_DATASETS, type_transforms = Transforms, lr = 0.05, batch_size = 64):
        super().__init__()
        self.data_dir = data_dir
        self.classes = (
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
        self.transforms = type_transforms


        self.model = ResNet18(len(self.classes))
        self.save_hyperparameters()
        self.BATCH_SIZE=batch_size

    def forward(self,x):
        return self.model(x)
    
    def training_step(self,batch,batch_id):
        x,y = batch
        logits = self(x)
        loss = F.cross_entropy(logits,y)
        self.log("training loss", loss)
        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def evaluate(self, batch, stage=None):
        x,y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        # print(preds.shape,y.shape)
        acc = accuracy(preds,y, task = "multiclass", num_classes=len(self.classes))

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
        # return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            # momentum=0.9,
            # weight_decay=5e-4,
        )
        # steps_per_epoch = 45000 // self.BATCH_SIZE
        # scheduler_dict = {
        #     "scheduler": OneCycleLR(
        #         optimizer,
        #         0.003,
        #         epochs=self.trainer.max_epochs,
        #         steps_per_epoch=steps_per_epoch,
        #     ),
        #     "interval": "step",
        # }
        # return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        return {"optimizer": optimizer}

    ##################
    # Data related hooks
    ##################
    """
    Another way to get data instead of the below methods : 
    The CIFAR10DataModule is a part of the pl_bolts library, which provides various pre-built data modules, models, and utilities to facilitate research and development with PyTorch Lightning. The CIFAR10DataModule specifically is designed to handle the CIFAR-10 dataset, making it easy to load, preprocess, and manage data for training and evaluation.
    cifar10_dm = CIFAR10DataModule(
    data_dir=PATH_DATASETS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    train_transforms=album_Compose_train(),
    test_transforms=album_Compose_test(),
    val_transforms=album_Compose_test(),
)
    cifar10_dm.prepare_data()  # downloads the dataset if necessary.
    cifar10_dm.setup()  # prepares the data loaders. Automatic Calling: When you pass the datamodule to the Trainer.fit method, PyTorch Lightning takes care of calling prepare_data and setup at the appropriate times. Customization: If your data module’s prepare_data and setup methods are well-defined to handle everything internally, you might not need to call them explicitly.

    trainer.fit(model, cifar10_dm)
    trainer.test(model, datamodule=cifar10_dm)

    You can fetch cifar10_dm.test_dataloader(), cifar10_dm.dataset_train.dataset.classes
   
    1. But where is validation used??
    2. and what kind of transform needs to be applied on it? train or test ka?

    """

    def prepare_data(self):
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None) :
    # Assign train/val datasets for use in dataloaders. How does None work for both if conditions
        if stage == "fit" or stage is None:
            cifar10_full = datasets.CIFAR10(self.data_dir, train=True, transform=self.transforms([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616], train=True))

            # Calculate the number of validation samples
            val_size = int(len(cifar10_full) * 0.1)
            train_size = len(cifar10_full) - val_size

            self.cifar10_train, self.cifar10_val = random_split(cifar10_full, [train_size, val_size])
    # Assign test dataset for use in dataloader (s)
        if stage == "test" or stage is None:
            self.cifar10_test = datasets.CIFAR10(self.data_dir, train=False, transform=self.transforms([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616], train=False))

    def train_dataloader(self) :
        return DataLoader(self.cifar10_train, batch_size=self.BATCH_SIZE, num_workers=os. cpu_count ( ))

    def val_dataloader (self) : 
        return DataLoader(self.cifar10_val, batch_size=self.BATCH_SIZE, num_workers=os. cpu_count ())

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.BATCH_SIZE, num_workers=os. cpu_count())

    
