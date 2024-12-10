import os
import json
import matplotlib.pyplot as plt
import numpy as np


def compare_model_results():
    # Load results from JSON files
    results_dir = './model_results'

    # Read comprehensive results
    with open(os.path.join(results_dir, 'comprehensive_results.json'), 'r') as f:
        all_results = json.load(f)

    # Prepare data for comparison
    model_names = list(all_results.keys())
    final_train_accuracies = [results['final_train_accuracy'] for results in all_results.values()]
    final_val_accuracies = [results['final_val_accuracy'] for results in all_results.values()]

    # Create comparison plots
    plt.figure(figsize=(15, 10))

    # Final Accuracies Comparison
    plt.subplot(2, 2, 1)
    x = np.arange(len(model_names))
    width = 0.35
    plt.bar(x - width / 2, final_train_accuracies, width, label='Train Accuracy')
    plt.bar(x + width / 2, final_val_accuracies, width, label='Validation Accuracy')
    plt.title('Final Accuracies Comparison')
    plt.xlabel('Models')
    plt.ylabel('Accuracy (%)')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()

    # Learning Curves Comparison
    plt.subplot(2, 2, 2)
    for name, results in all_results.items():
        plt.plot(results['train_accuracies'], label=f'{name} - Train')
        plt.plot(results['val_accuracies'], label=f'{name} - Val', linestyle='--')
    plt.title('Learning Curves Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Loss Curves Comparison
    plt.subplot(2, 2, 3)
    for name, results in all_results.items():
        plt.plot(results['train_losses'], label=f'{name} - Train Loss')
        plt.plot(results['val_losses'], label=f'{name} - Val Loss', linestyle='--')
    plt.title('Loss Curves Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Per-class Accuracy Comparison
    plt.subplot(2, 2, 4)
    for name, results in all_results.items():
        plt.bar(
            [f"{name}_{i}" for i in range(len(results['class_accuracies']))],
            results['class_accuracies'],
            label=name
        )
    plt.title('Per-Class Accuracy Comparison')
    plt.xlabel('Model and Class Index')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_comparison.png'), bbox_inches='tight')
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