import torch

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