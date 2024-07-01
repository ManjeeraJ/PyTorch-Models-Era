from torchinfo import summary

def getModelSummary(model, input_size):
    """
    Prints a detailed summary of a PyTorch model.

    Parameters:
    model (torch.nn.Module): The PyTorch model to summarize.

    Returns:
    None
    """
    # Print the model summary
    summary(model, input_size=input_size)