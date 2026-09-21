import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import AnimalsDataset, create_evaluation_transform, create_train_transform
from src.model import AnimalCNN
DATA_DIRECTORY = "image_data/train"

def get_arguments():
    parser = argparse.ArgumentParser(description="Train AnimalCNN for animal image classification.")

    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,help="Path to the training dataset.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for checkpoints and TensorBoard logs.")
    parser.add_argument("--epochs", type=int, default=25,help="Total number of training epochs.")
    parser.add_argument("--image-size",type=int,default=224,help="Input image height and width.")
    parser.add_argument("--batch-size",type=int,default=64,help="Number of images in one batch.")
    parser.add_argument("--val-ratio",type=float,default=0.2,help="Fraction of training data used for validation.")
    parser.add_argument("--learning-rate",type=float,default=0.01,help="Learning rate for SGD.")
    parser.add_argument("--momentum",type=float,default=0.9,help="Momentum for SGD.")
    parser.add_argument("--weight-decay",type=float,default=0.0001, help="Weight decay for SGD.")
    parser.add_argument("--seed",type=int,default=42,help="Random seed.")
    parser.add_argument("--num-workers",type=int,default=0,help="Number of DataLoader workers.")
    parser.add_argument("--resume",type=str,default=None,help="Path to a checkpoint for resuming training.")

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_data_loaders(
    data_dir,
    image_size,
    batch_size,
    val_ratio,
    seed,
    num_workers,
    pin_memory,
):
    train_transform = create_train_transform(image_size=image_size)
    evaluation_transform = create_evaluation_transform(image_size=image_size)

    train_dataset_full = AnimalsDataset(data_dir=data_dir, transform=train_transform)
    val_dataset_full = AnimalsDataset(data_dir=data_dir, transform=evaluation_transform)

    if train_dataset_full.class_to_idx != val_dataset_full.class_to_idx:
        raise ValueError("Training and validation class mappings are different.")

    sample_indices = list(range(len(train_dataset_full)))
    sample_class_index = []

    for image_path, class_index in train_dataset_full.samples:
        sample_class_index.append(class_index)

    train_indices, val_indices = train_test_split(
        sample_indices,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=sample_class_index,
    )

    train_dataset = Subset(train_dataset_full,train_indices,)
    val_dataset = Subset(val_dataset_full,val_indices,)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, train_dataset_full.class_to_idx


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    number_of_correct_predictions = 0
    number_of_processed_samples = 0

    progress_bar = tqdm(data_loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        # input tensor shape is [batch_size, channels, height, width], and image must be a tensor to be process
        batch_size = images.size(0) # image.shape = (64, 3, 224, 224), it follows batch size = image.shape[0] = 64

        running_loss += (loss.item() * batch_size)

        predictions = logits.argmax(dim=1) # looking for the max prediction every row of logits, return class of max point

        number_of_correct_predictions += (predictions == labels).sum().item()
        number_of_processed_samples += batch_size

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = (running_loss/ number_of_processed_samples)
    epoch_accuracy = (number_of_correct_predictions/ number_of_processed_samples)

    return epoch_loss, epoch_accuracy


def validate_one_epoch(model, data_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    number_of_correct_predictions = 0
    number_of_processed_samples = 0

    progress_bar = tqdm(data_loader, desc="Validation", leave=False)

    with torch.inference_mode():
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)

            running_loss += (loss.item() * batch_size)

            predictions = logits.argmax(dim=1)

            number_of_correct_predictions += (predictions == labels).sum().item()
            number_of_processed_samples += batch_size

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")


    epoch_loss = (running_loss / number_of_processed_samples)
    epoch_accuracy = (number_of_correct_predictions / number_of_processed_samples)

    return epoch_loss, epoch_accuracy


def create_checkpoint(
    epoch,
    model,
    optimizer,
    best_val_accuracy,
    class_to_idx,
    arguments,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_accuracy": best_val_accuracy,
        "class_to_idx": class_to_idx,
        "model_config": {"num_classes": len(class_to_idx)},
        "preprocessing_config": {
            "image_size": arguments.image_size,
            "normalize": False,
        },
        "training_config": {
            "batch_size": arguments.batch_size,
            "validation_ratio": arguments.val_ratio,
            "learning_rate": arguments.learning_rate,
            "momentum": arguments.momentum,
            "weight_decay": arguments.weight_decay,
            "seed": arguments.seed,
        },
    }

    return checkpoint


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    device,
    current_class_to_idx,
):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    checkpoint_class_to_idx = checkpoint.get("class_to_idx")

    if checkpoint_class_to_idx != current_class_to_idx:
        raise ValueError("Checkpoint class mapping does not match the current dataset.")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_accuracy = checkpoint["best_val_accuracy"]

    return start_epoch, best_val_accuracy


def main():
    arguments = get_arguments()

    if arguments.epochs < 1:
        raise ValueError("The number of epochs must be at least 1.")

    if arguments.batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    if not 0.0 < arguments.val_ratio < 1.0:
        raise ValueError("Validation ratio must be between 0 and 1.")

    set_seed(arguments.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("===== TRAINING CONFIGURATION =====")
    print(f"Device: {device}")
    print(f"Data directory: {arguments.data_dir}")
    print(f"Epochs: {arguments.epochs}")
    print(f"Batch size: {arguments.batch_size}")
    print(f"Image size: {arguments.image_size}")
    print(f"Validation ratio: {arguments.val_ratio}")
    print(f"Learning rate: {arguments.learning_rate}")

    output_directory = Path(arguments.output_dir)

    checkpoint_directory = (output_directory / "checkpoints")
    tensorboard_directory = (output_directory / "tensorboard")

    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    tensorboard_directory.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_to_idx = (
        create_data_loaders(
            data_dir=arguments.data_dir,
            image_size=arguments.image_size,
            batch_size=arguments.batch_size,
            val_ratio=arguments.val_ratio,
            seed=arguments.seed,
            num_workers=arguments.num_workers,
            pin_memory=device.type == "cuda",
        )
    )

    print("\n===== DATASET INFORMATION =====")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Number of classes: {len(class_to_idx)}")
    print(f"Class mapping: {class_to_idx}")

    model = AnimalCNN(num_classes=len(class_to_idx))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=arguments.learning_rate,
        momentum=arguments.momentum,
        weight_decay=arguments.weight_decay,
    )

    start_epoch = 1
    best_val_accuracy = 0.0

    if arguments.resume is not None:
        start_epoch, best_val_accuracy = (
            load_checkpoint(
                checkpoint_path=arguments.resume,
                model=model,
                optimizer=optimizer,
                device=device,
                current_class_to_idx=class_to_idx,
            )
        )

        print("\n===== RESUME TRAINING =====")
        print(f"Checkpoint: {arguments.resume}")
        print(f"Start epoch: {start_epoch}")
        print(f"Best validation accuracy: {best_val_accuracy:.2%}")

    if start_epoch > arguments.epochs:
        raise ValueError(
            f"Checkpoint already completed epoch "
            f"{start_epoch - 1}, but --epochs is "
            f"{arguments.epochs}."
        )

    writer = SummaryWriter(
        log_dir=str(tensorboard_directory),
        purge_step=start_epoch,
    )

    print("\n===== START TRAINING =====")

    try:
        for epoch in range(
            start_epoch,
            arguments.epochs + 1,
        ):
            print(f"\nEpoch {epoch}/{arguments.epochs}")

            train_loss, train_accuracy = (
                train_one_epoch(
                    model=model,
                    data_loader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device
                )
            )

            val_loss, val_accuracy = (
                validate_one_epoch(
                    model=model,
                    data_loader=val_loader,
                    criterion=criterion,
                    device=device
                )
            )

            print(
                f"Train loss: {train_loss:.4f} | "
                f"Train accuracy: {train_accuracy:.2%}"
            )

            print(
                f"Validation loss: {val_loss:.4f} | "
                f"Validation accuracy: {val_accuracy:.2%}"
            )

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_accuracy, epoch)
            writer.add_scalar("Accuracy/validation", val_accuracy, epoch)
            is_best_model = (val_accuracy > best_val_accuracy)

            if is_best_model:
                best_val_accuracy = val_accuracy

            checkpoint = create_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_val_accuracy=best_val_accuracy,
                class_to_idx=class_to_idx,
                arguments=arguments,
            )

            last_checkpoint_path = (checkpoint_directory / "last.pt")

            torch.save(checkpoint, last_checkpoint_path)

            if is_best_model:
                best_checkpoint_path = (
                    checkpoint_directory / "best.pt"
                )

                torch.save(checkpoint,best_checkpoint_path,)

                print(
                    "New best model saved with "
                    f"validation accuracy "
                    f"{best_val_accuracy:.2%}."
                )

    finally:
        writer.close()

    print("\n===== TRAINING COMPLETED =====")
    print(f"Best validation accuracy: {best_val_accuracy:.2%}")
    print(f"Checkpoints: {checkpoint_directory}")
    print(f"TensorBoard logs: {tensorboard_directory}")


if __name__ == "__main__":
    main()