import os
import json
import matplotlib.pyplot as plt
import numpy as np


def compare_model_results():
    # Load results from JSON files
    results_dir = './model_results'

    # Read comprehensive results
    with open(os.path.join(results_dir, 'comprehensive_results.json'), 'r') as f:
        comprehensive_results = json.load(f)
    all_results = comprehensive_results['models']
    class_counts = comprehensive_results['class_counts']

    # Prepare data for comparison
    model_names = list(all_results.keys())
    final_train_accuracies = [results['final_train_accuracy'] for results in all_results.values()]
    final_val_accuracies = [results['final_val_accuracy'] for results in all_results.values()]

    # First Figure: Accuracies and Learning Curves
    plt.figure(figsize=(15, 10))

    # Final Accuracies Comparison
    plt.subplot(2, 1, 1)
    x = np.arange(len(model_names))
    width = 0.35
    plt.bar(x - width / 2, final_train_accuracies, width, label='Train Accuracy')
    plt.bar(x + width / 2, final_val_accuracies, width, label='Validation Accuracy')

    # Add numbers to the bars
    for i in range(len(model_names)):
        plt.text(x[i] - width / 2, final_train_accuracies[i] + 0.5, f'{final_train_accuracies[i]:.2f}%', ha='center',
                 va='bottom')
        plt.text(x[i] + width / 2, final_val_accuracies[i] + 0.5, f'{final_val_accuracies[i]:.2f}%', ha='center',
                 va='bottom')

    plt.title('Final Accuracies Comparison')
    plt.xlabel('Models')
    plt.ylabel('Accuracy (%)')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()

    # Per-class Accuracy Comparison
    plt.subplot(2, 1, 2)
    width = 0.2  # Width of each bar
    x = np.arange(len(next(iter(all_results.values()))['class_accuracies']))  # Class indices
    for i, (name, results) in enumerate(all_results.items()):
        bars = plt.bar(
            x + i * width,
            results['class_accuracies'],
            width,
            label=name
        )
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom'
            )
    plt.title('Per-Class Accuracy Comparison')
    plt.xlabel('Model and Class Index')
    plt.ylabel('Accuracy (%)')
    plt.xticks(x + width * (len(all_results) - 1) / 2, range(len(x)))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_accuracies_and_learning_curves.png'), bbox_inches='tight')
    plt.close()



    # Second Figure: Loss Curves and Per-class Accuracy
    plt.figure(figsize=(15, 10))

    # Loss Curves Comparison
    plt.subplot(2, 1, 1)
    for name, results in all_results.items():
        plt.plot(results['train_losses'], label=f'{name} - Train Loss')
        plt.plot(results['val_losses'], label=f'{name} - Val Loss', linestyle='--')
    plt.title('Loss Curves Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Learning Curves Comparison
    plt.subplot(2, 1, 2)
    for name, results in all_results.items():
        plt.plot(results['train_accuracies'], label=f'{name} - Train')
        plt.plot(results['val_accuracies'], label=f'{name} - Val', linestyle='--')
    plt.title('Learning Curves Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_losses_and_class_accuracies.png'), bbox_inches='tight')
    plt.close()

    # Class Distribution Histogram (as a separate figure)
    plt.figure(figsize=(15, 5))
    width = 0.2  # Width of each bar
    x = np.arange(len(next(iter(class_counts.values()))))  # Class indices
    for i, (model_name, counts) in enumerate(class_counts.items()):
        plt.bar(x + i * width, counts, width, label=model_name)
    plt.title('Class Distribution in Training Dataset')
    plt.xlabel('Class Index')
    plt.ylabel('Number of Images')
    plt.xticks(x + width * (len(class_counts) - 1) / 2, range(len(counts)))
    plt.legend()

    # Save class distribution plot
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'class_distribution.png'), bbox_inches='tight')
    plt.close()

    # Print summary
    print("Model Comparison Summary:")
    for name, results in all_results.items():
        print(f"\n{name}:")
        print(f"  Final Train Accuracy: {results['final_train_accuracy']:.2f}%")
        print(f"  Final Validation Accuracy: {results['final_val_accuracy']:.2f}%")
        print("  Per-Class Accuracies:",
              ", ".join([f"{i}: {acc:.2f}%" for i, acc in enumerate(results['class_accuracies'])]))


def main():
    compare_model_results()


if __name__ == '__main__':
    main()