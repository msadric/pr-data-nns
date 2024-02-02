# Plot Amdahl-scalability (number of cores vs. speedup) -> see lecture 4, slide 18

# Plot runtime [s] vs number of nodes

# Plot speedup vs single-node numpy implementation (see lecture 3, slide 19)

import matplotlib.pyplot as plt

def plot_losses(train_loss, train_accuracy, validation_accuracy, save_path):
        # Create subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Plot Training Loss over Epochs (Left subplot)
        epochs = range(1, len(train_loss) + 1)
        
        axs[0].plot(epochs, train_loss, marker='o', linestyle='-', color='b')
        axs[0].set_title('Train Loss over Epochs')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Training Loss')

        # Plot Training and Validation Accuracy over Epochs (Right subplot)
        axs[1].plot(epochs, train_accuracy, marker='o', linestyle='-', color='g', label='Training Accuracy')
        axs[1].plot(epochs, validation_accuracy, marker='o', linestyle='-', color='r', label='Validation Accuracy')
        axs[1].set_title('Accuracy over Epochs')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy (%)')
        axs[1].legend()

        # Adjust layout to prevent clipping of ylabel
        plt.tight_layout()

        # Save the figures to a given path
        plt.savefig(save_path + "training_plot.png")