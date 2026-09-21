from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}

class AnimalsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Get and sort all class folder names after alphabets
        self.classes = sorted([
            folder.name
            for folder in self.data_dir.iterdir()
            if folder.is_dir()
        ])
        

        # Create a dictionary class mapping to index
        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(self.classes)
        }

        # Save image paths and corresponding labels
        self.samples = []

        for class_name in self.classes: #duyệt từng class trong list classes đã được sorted
            class_folder = self.data_dir / class_name
            class_index = self.class_to_idx[class_name]
            
            #Duyệt từng ảnh trong cái class folder hiện tại
            for image_path in sorted(class_folder.iterdir()):
                if image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    self.samples.append((image_path, class_index)) #

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        # Convert every image to RGB because AnimalCNN expects 3 input channels.
        image = Image.open(image_path).convert("RGB") #đọc ảnh bằng thư viện Pillow => PIL image


        if self.transform:
            image = self.transform(image)

        return image, label

# Preprocessing for training
def create_train_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])


def create_evaluation_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    
def main():
    train_data_path = "image_data/train"
    test_data_path = "image_data/test"

    # Use evaluation transform for testing
    evaluation_transform = create_evaluation_transform()

    # Create datasets
    train_dataset = AnimalsDataset(data_dir=train_data_path, transform=evaluation_transform)

    test_dataset = AnimalsDataset(data_dir=test_data_path, transform=evaluation_transform)

    # Print general dataset information
    print("===== TRAIN DATASET =====")
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Class mapping: {train_dataset.class_to_idx}")

    print("\n===== TEST DATASET =====")
    print(f"Number of test images: {len(test_dataset)}")
    print(f"Number of classes: {len(test_dataset.classes)}")
    print(f"Classes: {test_dataset.classes}")
    print(f"Class mapping: {test_dataset.class_to_idx}")

    # Check whether train and test use the same mapping
    print("\n===== CLASS MAPPING CHECK =====")

    if train_dataset.class_to_idx == test_dataset.class_to_idx:
        print("Train and test class mappings are identical.")
    else:
        print("Warning: Train and test class mappings are different.")

    # Load and inspect the first training image
    image, label = train_dataset[0]

    print("\n===== FIRST SAMPLE =====")
    print(f"Image tensor shape: {image.shape}")
    print(f"Label index: {label}")
    print(f"Class name: {train_dataset.classes[label]}")
    print(f"Image data type: {image.dtype}")

    # Count images in every training class
    class_counts = {
        class_name: 0
        for class_name in train_dataset.classes
    }

    for _, class_index in train_dataset.samples:
        class_name = train_dataset.classes[class_index]
        class_counts[class_name] += 1

    print("\n===== TRAINING IMAGES PER CLASS =====")

    for class_name, number_of_images in class_counts.items():
        print(f"{class_name}: {number_of_images}")
        
if __name__ == "__main__":
    main()