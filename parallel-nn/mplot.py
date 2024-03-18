import matplotlib.pyplot as plt
from matplotlib import rc

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 22
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def plot_losses(train_loss, train_accuracy, validation_accuracy, save_path, epochs, batch_size):

    # Create two separate figures
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    train_loss = extract_epoch_losses(train_loss, epochs)
    train_accuracy = [tensor.item() for tensor in train_accuracy]
    validation_accuracy = [tensor.item() for tensor in validation_accuracy]

    # Plot Training Loss over Epochs
    ax1.plot(train_loss, marker='o')
        
    ax1.set_xlabel(r'Epochs')
    ax1.set_ylabel(r'Loss')

    # Plot Training and Validation Accuracy over Epochs
    ax2.plot(train_accuracy, marker='o', label=r'Training Accuracy')
    ax2.plot(validation_accuracy, marker='o', label=r'Validation Accuracy')
    ax2.set_xlabel(r'Epochs')
    ax2.set_ylabel(r'Accuracy [\%]')
    ax2.legend()

    # Save the figures to a given path
    fig1.savefig(save_path + "/training_loss_plot.png")
    fig2.savefig(save_path + "/accuracy_plot.png")
    print("Plots saved")

def extract_epoch_losses(losses, epochs):
    epoch_losses = []
    total_batches = len(losses)
    batches_per_epoch = total_batches // epochs

    for i in range(epochs):
        epoch_loss = losses[(i+1)*batches_per_epoch-1]
        epoch_losses.append(epoch_loss)

    return epoch_losses
