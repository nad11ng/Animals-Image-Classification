import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import AnimalsDataset,create_evaluation_transform
from src.model import AnimalCNN


def get_arguments():
    parser = argparse.ArgumentParser(description=("Evaluate AnimalCNN on the independent test dataset."))

    parser.add_argument("--data-dir", type=str, default="image_data/test", help="Path to the test dataset.")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt", help="Path to the trained model checkpoint.")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation", help="Directory for evaluation results.")
    parser.add_argument("--batch-size", type=int, default=64, help="Number of test images in one batch.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")

    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    required_keys = [
        "model_state_dict",
        "class_to_idx",
        "model_config",
        "preprocessing_config",
    ]

    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(f"Checkpoint does not contain required key: {key}")

    class_to_idx = checkpoint["class_to_idx"]
    model_config = checkpoint["model_config"]
    preprocessing_config = checkpoint["preprocessing_config"]

    model = AnimalCNN(num_classes=model_config["num_classes"])

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    return model, checkpoint, class_to_idx, preprocessing_config


def create_test_loader(
    data_dir,
    image_size,
    batch_size,
    num_workers,
    pin_memory,
    checkpoint_class_to_idx,
):
    evaluation_transform = create_evaluation_transform(image_size=image_size)

    test_dataset = AnimalsDataset(data_dir=data_dir, transform=evaluation_transform)

    if (test_dataset.class_to_idx != checkpoint_class_to_idx):
        raise ValueError(
            "Test dataset class mapping does not match "
            "the checkpoint class mapping.\n"
            f"Test mapping: {test_dataset.class_to_idx}\n"
            f"Checkpoint mapping: {checkpoint_class_to_idx}"
        )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return test_dataset, test_loader


def evaluate_model(model, data_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    number_of_processed_samples = 0

    true_labels = []
    predicted_labels = []

    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)

    with torch.inference_mode():
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            batch_size = images.size(0)
            
            running_loss += (loss.item() * batch_size)
            number_of_processed_samples += batch_size

            predictions = logits.argmax(dim=1)

            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(predictions.cpu().tolist())

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    if number_of_processed_samples == 0:
        raise ValueError("The test dataset contains no samples.")

    test_loss = (running_loss / number_of_processed_samples)

    return test_loss, true_labels, predicted_labels


def calculate_metrics(true_labels, predicted_labels, class_names):
    class_indices = list(range(len(class_names)))

    test_accuracy = accuracy_score(true_labels, predicted_labels)

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=class_indices,
        average="macro",
        zero_division=0,
    )

    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _,
    ) = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=class_indices,
        average="weighted",
        zero_division=0,
    )

    report_text = classification_report(
        true_labels,
        predicted_labels,
        labels=class_indices,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    report_dictionary = classification_report(
        true_labels,
        predicted_labels,
        labels=class_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    confusion_matrix_values = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=class_indices,
    )

    metrics = {
        "test_accuracy": float(test_accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }

    return  metrics, report_text, report_dictionary, confusion_matrix_values,


def save_classification_report(
    output_path,
    checkpoint_path,
    checkpoint_epoch,
    best_val_accuracy,
    test_loss,
    metrics,
    report_text,
):
    report_lines = [
        "ANIMAL IMAGE CLASSIFICATION - TEST EVALUATION",
        "=" * 50,
        "",
        f"Checkpoint: {checkpoint_path}",
        f"Checkpoint epoch: {checkpoint_epoch}",
        (
            "Best validation accuracy: "
            f"{best_val_accuracy:.2%}"
        ),
        "",
        f"Test loss: {test_loss:.4f}",
        (
            "Test accuracy: "
            f"{metrics['test_accuracy']:.2%}"
        ),
        (
            "Macro precision: "
            f"{metrics['macro_precision']:.4f}"
        ),
        (
            "Macro recall: "
            f"{metrics['macro_recall']:.4f}"
        ),
        (
            "Macro F1-score: "
            f"{metrics['macro_f1']:.4f}"
        ),
        (
            "Weighted precision: "
            f"{metrics['weighted_precision']:.4f}"
        ),
        (
            "Weighted recall: "
            f"{metrics['weighted_recall']:.4f}"
        ),
        (
            "Weighted F1-score: "
            f"{metrics['weighted_f1']:.4f}"
        ),
        "",
        "CLASSIFICATION REPORT",
        "=" * 50,
        report_text,
    ]

    output_path.write_text("\n".join(report_lines), encoding="utf-8")


def save_metrics_json(
    output_path,
    checkpoint_path,
    checkpoint_epoch,
    best_val_accuracy,
    test_loss,
    metrics,
    report_dictionary,
):
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(
            checkpoint_epoch
        ),
        "best_validation_accuracy": float(
            best_val_accuracy
        ),
        "test_loss": float(test_loss),
        **metrics,
        "classification_report": (
            report_dictionary
        ),
    }

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            result,
            json_file,
            indent=4,
            ensure_ascii=False,
        )


def save_confusion_matrix(confusion_matrix_values, class_names, output_path):
    figure, axis = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        confusion_matrix_values,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=axis,
    )

    axis.set_title("Confusion Matrix - Test Dataset")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight",)
    plt.close(figure)


def main():
    arguments = get_arguments()

    if arguments.batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    if arguments.num_workers < 0:
        raise ValueError("Number of workers cannot be negative.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    output_directory = Path(arguments.output_dir)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("===== EVALUATION CONFIGURATION =====")
    print(f"Device: {device}")
    print(f"Test data: {arguments.data_dir}")
    print(f"Checkpoint: {arguments.checkpoint}")
    print(f"Batch size: {arguments.batch_size}")

    (
        model,
        checkpoint,
        class_to_idx,
        preprocessing_config,
    ) = load_model_from_checkpoint(
        checkpoint_path=arguments.checkpoint,
        device=device,
    )

    image_size = preprocessing_config["image_size"]

    normalize = preprocessing_config.get("normalize",False,)

    if normalize:
        raise ValueError(
            "The checkpoint expects normalized inputs, "
            "but the current evaluation transform does "
            "not use normalization."
        )

    number_of_classes = len(class_to_idx)

    expected_number_of_classes = checkpoint["model_config"]["num_classes"]

    if number_of_classes != expected_number_of_classes:
        raise ValueError(
            "The number of classes in class_to_idx "
            "does not match model_config."
        )

    idx_to_class = {}

    for class_name, class_index in (class_to_idx.items()):
        idx_to_class[class_index] = class_name

    class_names = []

    for class_index in range(number_of_classes):
        if class_index not in idx_to_class:
            raise ValueError(
                f"Class index {class_index} is missing "
                "from the checkpoint class mapping."
            )

        class_names.append(idx_to_class[class_index])

    test_dataset, test_loader = (
        create_test_loader(
            data_dir=arguments.data_dir,
            image_size=image_size,
            batch_size=arguments.batch_size,
            num_workers=arguments.num_workers,
            pin_memory=device.type == "cuda",
            checkpoint_class_to_idx=class_to_idx,
        )
    )

    print("\n===== CHECKPOINT INFORMATION =====")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Best validation accuracy: {checkpoint['best_val_accuracy']:.2%}")
    print(f"Image size: {image_size}")
    print(f"Normalize: {normalize}")

    print("\n===== TEST DATASET INFORMATION =====")
    print(f"Number of test images: {len(test_dataset)}")
    print(f"Number of classes: {number_of_classes}")
    print(f"Classes: {class_names}")

    criterion = nn.CrossEntropyLoss()

    (
        test_loss,
        true_labels,
        predicted_labels,
    ) = evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    (
        metrics,
        report_text,
        report_dictionary,
        confusion_matrix_values,
    ) = calculate_metrics(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        class_names=class_names,
    )

    print("\n===== TEST RESULTS =====")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.2%}")
    print(f"Macro precision: {metrics['macro_precision']:.4f}")
    print(f"Macro recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1-score: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1-score: {metrics['weighted_f1']:.4f}")

    print("\n===== CLASSIFICATION REPORT =====")
    print(report_text)

    report_path = (output_directory / "classification_report.txt")

    metrics_path = (output_directory / "evaluation_metrics.json")

    confusion_matrix_path = (output_directory / "confusion_matrix.png")

    save_classification_report(
        output_path=report_path,
        checkpoint_path=arguments.checkpoint,
        checkpoint_epoch=checkpoint["epoch"],
        best_val_accuracy=checkpoint[
            "best_val_accuracy"
        ],
        test_loss=test_loss,
        metrics=metrics,
        report_text=report_text,
    )

    save_metrics_json(
        output_path=metrics_path,
        checkpoint_path=arguments.checkpoint,
        checkpoint_epoch=checkpoint["epoch"],
        best_val_accuracy=checkpoint[
            "best_val_accuracy"
        ],
        test_loss=test_loss,
        metrics=metrics,
        report_dictionary=report_dictionary,
    )

    save_confusion_matrix(
        confusion_matrix_values=(
            confusion_matrix_values
        ),
        class_names=class_names,
        output_path=confusion_matrix_path,
    )

    print("\n===== EVALUATION COMPLETED =====")
    print(f"Report: {report_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Confusion matrix: {confusion_matrix_path}")


if __name__ == "__main__":
    main()