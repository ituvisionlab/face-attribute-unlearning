import matplotlib.pyplot as plt

def save_loss_plot(train_loss, val_loss, file_path):
    """
    Save the training and validation loss plot as an image.

    Parameters:
    - train_loss (list): List of training loss values for each epoch.
    - val_loss (list): List of validation loss values for each epoch.
    - file_path (str): File path to save the plot image. Default is 'loss_plot.png'.
    """    
    epochs = range(1, len(train_loss) + 1)

    plt.plot(epochs, train_loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(file_path)
    print(f"Loss plot saved as {file_path}")