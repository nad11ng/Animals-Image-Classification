import argparse
from pathlib import Path

import torch
from PIL import Image

from src.dataset import create_evaluation_transform
from src.model import AnimalCNN


DEFAULT_CHECKPOINT_PATH = "outputs/checkpoints/best.pt"


class InferenceEngine:
    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT_PATH, device=None):
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.checkpoint_path}")

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self._validate_checkpoint(checkpoint)

        self.class_to_idx = checkpoint["class_to_idx"]

        self.idx_to_class = {
            class_index: class_name
            for class_name, class_index
            in self.class_to_idx.items()
        }

        model_config = checkpoint["model_config"]
        preprocessing_config = checkpoint["preprocessing_config"]

        self.num_classes = model_config["num_classes"]
        self.image_size = preprocessing_config["image_size"]

        normalize = preprocessing_config.get("normalize", False)

        if normalize:
            raise ValueError(
                "This checkpoint requires normalization, "
                "but the current evaluation transform does "
                "not support normalization."
            )

        self._validate_class_mapping()

        self.transform = create_evaluation_transform(image_size=self.image_size)

        self.model = AnimalCNN(num_classes=self.num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.checkpoint_epoch = checkpoint.get("epoch")

        self.best_validation_accuracy = checkpoint.get("best_val_accuracy")

    @staticmethod
    def _validate_checkpoint(checkpoint):
        required_keys = [
            "model_state_dict",
            "class_to_idx",
            "model_config",
            "preprocessing_config",
        ]

        missing_keys = []

        for key in required_keys:
            if key not in checkpoint:
                missing_keys.append(key)

        if missing_keys:
            raise KeyError(f"Checkpoint is missing required keys: {missing_keys}")

    def _validate_class_mapping(self):
        if len(self.class_to_idx) != self.num_classes:
            raise ValueError("Number of classes in class_to_idx does not match model_config['num_classes'].")

        expected_indices = set(range(self.num_classes))

        actual_indices = set(self.idx_to_class.keys())

        if actual_indices != expected_indices:
            raise ValueError("Class indices must start at 0 and be continuous.")

    def predict(self, image, top_k=3):
        if image is None:
            raise ValueError("No image was provided.")

        if not isinstance(image, Image.Image):
            raise TypeError("The input image must be a PIL image.")

        if not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")

        top_k = max(1, min(top_k, self.num_classes))

        image = image.convert("RGB")

        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        with torch.inference_mode():
            logits = self.model(image_tensor)

            probabilities = torch.softmax(logits, dim=1)

            top_probabilities, top_indices = (
                probabilities.topk(
                    k=top_k,
                    dim=1,
                )
            )

        top_probabilities = (top_probabilities[0].cpu().tolist())

        top_indices = (top_indices[0].cpu().tolist())

        predictions = {}

        for probability, class_index in zip(top_probabilities, top_indices):
            class_name = self.idx_to_class[class_index]

            predictions[class_name] = probability

        return predictions

    def predict_path(self, image_path, top_k=3):
        image_path = Path(image_path)

        if not image_path.is_file():
            raise FileNotFoundError(f"Image does not exist: {image_path}")

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")

        return self.predict(image=rgb_image, top_k=top_k)


def get_arguments():
    parser = argparse.ArgumentParser(description=("Classify an animal image using the trained AnimalCNN model."))
    parser.add_argument("--image-path", type=str, required=True, help="Path to the image to classify.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_PATH, help="Path to the trained checkpoint.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of predictions to display.")

    return parser.parse_args()


def main():
    args = get_arguments()

    inference_engine = InferenceEngine(checkpoint_path=args.checkpoint)

    predictions = inference_engine.predict_path(image_path=args.image_path, top_k=args.top_k)

    print("===== INFERENCE CONFIGURATION =====")
    print(f"Device: {inference_engine.device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Image: {args.image_path}")
    print(f"Training epoch: {inference_engine.checkpoint_epoch}")

    if (inference_engine.best_validation_accuracy is not None):
        print(f"Best validation accuracy: {inference_engine.best_validation_accuracy:.2f}%")

    print("\n===== TOP PREDICTIONS =====")

    for rank, (class_name, probability) in enumerate(predictions.items(), start=1,):
        print(f"{rank}. {class_name}: {probability * 100:.2f}%")


if __name__ == "__main__":
    main()