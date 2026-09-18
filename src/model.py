import torch
import torch.nn as nn


def create_convolution_block(in_channels, out_channels):
    convolution_block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),

        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=2, stride=2,)
    )

    return convolution_block


class AnimalCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            create_convolution_block(in_channels=3, out_channels=8),
            create_convolution_block(in_channels=8, out_channels=16),
            create_convolution_block(in_channels=16, out_channels=32),
            create_convolution_block(in_channels=32, out_channels=64),
            create_convolution_block(in_channels=64, out_channels=64),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(4, 4))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64 * 4 * 4, out_features=256),
            nn.ReLU(),

            nn.Dropout(p=0.3),
            nn.Linear(in_features=256, out_features=num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)

        return x

def test_forward_pass(
    model,
    batch_size,
    image_size,
    num_classes=10,
):
    dummy_images = torch.randn(
        batch_size,
        3,
        image_size,
        image_size,
    )

    with torch.inference_mode():
        outputs = model(dummy_images)

    expected_shape = (
        batch_size,
        num_classes,
    )

    actual_shape = tuple(outputs.shape)

    print(
        f"Input: {tuple(dummy_images.shape)} "
        f"→ Output: {actual_shape}"
    )

    assert actual_shape == expected_shape, (
        f"Expected output shape {expected_shape}, "
        f"but received {actual_shape}."
    )

    assert torch.isfinite(outputs).all(), (
        "Output contains NaN or infinite values."
    )

    print("Forward pass successful.\n")

if __name__ == "__main__":
    model = AnimalCNN(num_classes=10)

    # Switch off Dropout and use BatchNorm in evaluation mode
    model.eval()

    # Create a dummy batch containing four RGB images
    dummy_images = torch.randn(
        4,
        3,
        224,
        224,
    )

    print("===== SHAPE TEST =====")
    print(f"Input shape: {dummy_images.shape}")

    x = dummy_images

    with torch.inference_mode():
        # Run through each convolution block separately
        for block_index, conv_block in enumerate(
            model.features,
            start=1,
        ):
            x = conv_block(x)

            print(
                f"After convolution block {block_index}: "
                f"{x.shape}"
            )

        # Adaptive pooling
        x = model.adaptive_pool(x)

        print(
            f"After adaptive pooling: {x.shape}"
        )

        # Flatten
        x = torch.flatten(
            x,
            start_dim=1,
        )

        print(
            f"After flatten: {x.shape}"
        )

        # Classifier
        outputs = model.classifier(x)

        print(
            f"Output shape: {outputs.shape}"
        )

    # Count model parameters
    total_parameters = 0
    trainable_parameters = 0

    for parameter in model.parameters():
        total_parameters += parameter.numel()

        if parameter.requires_grad:
            trainable_parameters += parameter.numel()

    print("\n===== MODEL PARAMETERS =====")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    
    # Use evaluation mode for model testing
    print("===== FORWARD PASS TESTS =====\n")

    test_cases = [
        (1, 128),
        (4, 224),
        (8, 256),
    ]

    for batch_size, image_size in test_cases:
        test_forward_pass(
            model=model,
            batch_size=batch_size,
            image_size=image_size,
            num_classes=10,
        )

    print("All forward pass tests completed successfully.")