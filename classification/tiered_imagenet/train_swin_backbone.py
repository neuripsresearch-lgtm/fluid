import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import timm 
import argparse
import os

def get_data_loaders(data_root, batch_size, image_size=224):
    """
    Returns the training and validation data loaders for the Tiered ImageNet dataset.
    Assumes data_root contains 'train' and 'val' subdirectories.
    """
    # --- Standard ImageNet Normalization ---
    # Since this is a subset of ImageNet, we use standard ImageNet stats
    # instead of CIFAR stats.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)), # Resize slightly larger (approx 256 for 224 input)
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_path = os.path.join(data_root, 'train')
    val_path = os.path.join(data_root, 'val')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train folder not found at: {train_path}")
    if not os.path.exists(val_path):
        print(f"Warning: Validation folder not found at {val_path}. Checking for 'test' folder...")
        val_path = os.path.join(data_root, 'test')
        if not os.path.exists(val_path):
             raise FileNotFoundError(f"Neither 'val' nor 'test' folders found in {data_root}")

    # using ImageFolder since data is organized in class folders
    trainset = torchvision.datasets.ImageFolder(root=train_path, transform=transform_train)
    valset = torchvision.datasets.ImageFolder(root=val_path, transform=transform_val)

    # Detect number of classes automatically
    num_classes = len(trainset.classes)
    print(f"Detected {num_classes} classes in the dataset.")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                                              num_workers=4, pin_memory=True)
    
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, 
                                            num_workers=4, pin_memory=True)

    return trainloader, valloader, num_classes

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data loaders and number of classes
    trainloader, valloader, num_classes = get_data_loaders(args.data_path, args.batch_size, image_size=224)

    # --- Model Definition ---
    print(f"Loading Swin Transformer model for {num_classes} classes...")
    # Initialize model with the correct number of classes for this dataset
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_classes)
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training Loop ---
    best_accuracy = 0.0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(trainloader)
        print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {avg_train_loss:.4f}')

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f} %')
        
        scheduler.step()

        # Save Best Backbone
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"New best model found! Accuracy: {best_accuracy:.2f}%. Saving backbone weights...")
            
            # Remove the head (classifier) before saving so it's ready for downstream tasks
            backbone_state_dict = {k: v for k, v in model.state_dict().items() if not k.startswith('head.')}
            torch.save(backbone_state_dict, args.save_path)

    print('Finished Training')
    print(f'Best validation accuracy: {best_accuracy:.2f}%')
    print(f'Backbone saved to: {args.save_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Swin Backbone on Tiered ImageNet')
    
    # Point this to the NEW folder created by the previous script
    parser.add_argument('--data-path', default='./data/tiered_imagenet_standard', type=str, 
                        help='Path to the dataset root (containing train/val folders)')
    
    parser.add_argument('--learning-rate', default=5e-5, type=float, help='LR')
    parser.add_argument('--weight-decay', default=5e-2, type=float, help='Weight decay')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size')
    parser.add_argument('--epochs', default=20, type=int, help='Epochs')
    parser.add_argument('--save-path', default='tiered_imagenet_swin_backbone.pth', type=str, 
                        help='Save path for .pth file')
    
    args = parser.parse_args()
    main(args)