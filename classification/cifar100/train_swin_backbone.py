# train_swin_backbone.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import timm # <-- Import the timm library
import argparse
import os

def get_cifar100_loaders(data_path, batch_size, image_size=224):
    """
    Returns the training and validation data loaders for CIFAR-100.
    Images are resized and augmented for the Swin Transformer.
    """
    # --- Updated Transforms for a Vision Transformer ---
    transform_train = transforms.Compose([
        # Transformers expect a larger image size, e.g., 224x224
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.TrivialAugmentWide(), # A strong, modern augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(image_size), # Just resize for validation
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return trainloader, testloader

def main(args):
    """
    Main function to run the training, validation, and saving of the backbone.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data loaders
    # The default image size for this swin model is 224
    trainloader, testloader = get_cifar100_loaders(args.data_path, args.batch_size, image_size=224)

    # --- Model Definition (Using Swin Transformer from timm) ---
    # 1. Load pre-trained Swin Transformer model
    print("Loading Swin Transformer model...")
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=100)
    
    # Note: For timm models, you can often directly set num_classes on creation.
    # The head layer is automatically replaced.
    # The equivalent manual replacement would be:
    # model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
    # num_ftrs = model.head.in_features
    # model.head = nn.Linear(num_ftrs, 100)
    
    # Move model to device
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # A smaller learning rate is often better for fine-tuning transformers
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training and Validation Loop ---
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
        
        print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {running_loss / len(trainloader):.4f}')

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Accuracy on test set: {accuracy:.2f} %')
        
        # Update learning rate
        scheduler.step()

        # Check if this is the best model so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"New best model found! Accuracy: {best_accuracy:.2f}%. Saving backbone weights...")
            
            # --- Save the Backbone (without the final 'head' layer) ---
            backbone_state_dict = {k: v for k, v in model.state_dict().items() if not k.startswith('head.')}
            torch.save(backbone_state_dict, args.save_path)

    print('Finished Training')
    print(f'Best accuracy achieved: {best_accuracy:.2f}%')
    print(f'The backbone of the best model has been saved to: {args.save_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Swin Transformer backbone on CIFAR-100 and save it.')
    parser.add_argument('--data-path', default='./data', type=str, 
                        help='Path to CIFAR-100 dataset directory.')
    parser.add_argument('--learning-rate', default=5e-5, type=float, 
                        help='Learning rate for the optimizer.')
    parser.add_argument('--weight-decay', default=5e-2, type=float, 
                        help='Weight decay for the optimizer.')
    parser.add_argument('--batch-size', default=64, type=int, # Reduced batch size as transformers are more memory intensive
                        help='Batch size for training and validation.')
    parser.add_argument('--epochs', default=25, type=int, # Fewer epochs needed for fine-tuning
                        help='Number of epochs to train.')
    parser.add_argument('--save-path', default='cifar100_swin_tiny_backbone.pth', type=str, 
                        help='Path to save the trained backbone weights.')
    
    args = parser.parse_args()
    main(args)