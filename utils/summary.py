from torchinfo import summary

def getModelSummary(model, device, input_size):
    """
    Prints a detailed summary of a PyTorch model.

    Parameters:
    model (torch.nn.Module): The PyTorch model to summarize.

    Returns:
    None
    """
    # Print the model summary
    summary(model.to(device), input_size=input_size)