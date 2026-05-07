# This script trains a standard Swin-T classifier and saves the *full model*.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import timm
import argparse
import os

def get_cifar100_loaders(data_path, batch_size, image_size=224):
    """
    Returns the training and validation data loaders for CIFAR-100.
    """
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(image_size),
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainloader, testloader = get_cifar100_loaders(args.data_path, args.batch_size, image_size=224)

    # --- Model Definition (Using Swin Transformer from timm) ---
    print("Loading Swin Transformer model with 100 classes...")
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=100)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training and Validation Loop ---
    best_accuracy = 0.0
    for epoch in range(args.epochs):
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
        
        scheduler.step()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"New best model found! Accuracy: {best_accuracy:.2f}%. Saving FULL model...")
            
            # --- MODIFICATION: Save the FULL Model State Dict ---
            # Instead of saving the backbone, we save the entire model
            torch.save(model.state_dict(), args.save_path)

    print('Finished Training')
    print(f'Best accuracy achieved: {best_accuracy:.2f}%')
    print(f'The FULL model has been saved to: {args.save_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Swin Transformer (full model) on CIFAR-100 and save it.')
    parser.add_argument('--data-path', default='./data', type=str, 
                        help='Path to CIFAR-100 dataset directory.')
    parser.add_argument('--learning-rate', default=5e-5, type=float, 
                        help='Learning rate for the optimizer.')
    parser.add_argument('--weight-decay', default=5e-2, type=float, 
                        help='Weight decay for the optimizer.')
    parser.add_argument('--batch-size', default=64, type=int, 
                        help='Batch size for training and validation.')
    parser.add_argument('--epochs', default=20, type=int, 
                        help='Number of epochs to train.')
    # MODIFICATION: Changed default save path
    parser.add_argument('--save-path', default='cifar100_flat_xe_swin.pth', type=str, 
                        help='Path to save the trained FULL model weights.')
    
    args = parser.parse_args()
    main(args)