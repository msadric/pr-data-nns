# Plot Amdahl-scalability (number of cores vs. speedup) -> see lecture 4, slide 18

# Plot runtime [s] vs number of nodes

# Plot speedup vs single-node numpy implementation (see lecture 3, slide 19)

import matplotlib.pyplot as plt

def plot_losses(train_loss, train_accuracy, validation_accuracy, save_path, epochs, batch_size):
        # Create subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        train_loss = extract_epoch_losses(train_loss, epochs)
        train_accuracy = [tensor.item() for tensor in train_accuracy]
        validation_accuracy = [tensor.item() for tensor in validation_accuracy]

        # Plot Training Loss over Epochs (Left subplot)
    
        axs[0].plot(train_loss, marker='o')
        axs[0].set_title('Train Loss over Epochs')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Training Loss')

        # Plot Training and Validation Accuracy over Epochs (Right subplot)
        axs[1].plot(train_accuracy, marker='o', label='Training Accuracy')
        axs[1].plot(validation_accuracy, marker='o', label='Validation Accuracy')
        axs[1].set_title('Accuracy over Epochs')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy (%)')
        axs[1].legend()

        # Adjust layout to prevent clipping of ylabel
        plt.tight_layout()

        # Save the figures to a given path
        plt.savefig(save_path + "/training_plot.png")
        print("plot saved")

def extract_epoch_losses(losses, epochs):
    epoch_losses = []
    total_batches = len(losses)
    batches_per_epoch = total_batches // epochs

    for i in range(epochs):
        epoch_loss = losses[(i+1)*batches_per_epoch-1]
        epoch_losses.append(epoch_loss)

    return epoch_losses