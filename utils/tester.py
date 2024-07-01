import torch

"""
1. 100.0 * correct / len(self.test_loader.dataset) commented out; not used anywhere
2. test_loss used for save_best_model and reduce lr on plateau
"""
class Tester:
    def __init__(self, model, test_loader, criterion, device) -> None:
        self.test_losses = []
        self.test_accuracies = []
        self.model = model.to(device)
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device

    def test(self):
        self.model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):  # Verify : for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output = self.model(inputs)
                loss = self.criterion(output, targets)  # .item() throws error 'float' object has no attribute 'item'

                test_loss += loss.item()

                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(targets.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        self.test_losses.append(test_loss)

        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                test_loss,
                correct,
                len(self.test_loader.dataset),
                100.0 * correct / len(self.test_loader.dataset),
            )
        )

        self.test_accuracies.append(100.0 * correct / len(self.test_loader.dataset))

        # return 100.0 * correct / len(self.test_loader.dataset), test_loss
        return test_loss

    def get_misclassified_images(self):
        
        # set model to evaluation mode
        self.model.eval()

        images = []
        predictions = []
        labels = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                _, preds = torch.max(output, 1)  # Perform torch.max along dimension 1 

                for i in range(len(preds)):
                    if preds[i] != target[i]:
                        images.append(data[i])
                        predictions.append(preds[i])
                        labels.append(target[i])

        return images, predictions, labels
