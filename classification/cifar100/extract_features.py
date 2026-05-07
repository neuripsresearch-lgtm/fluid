# gemini/python_scripts/extract_features.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm
import os
from tqdm import tqdm

def extract_and_save(loader, model, device, save_path_features, save_path_labels):
    features_list = []
    labels_list = []
    
    print(f"Extracting features...")
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            # Extract features (batch_size, 768)
            output = model(images)
            features_list.append(output.cpu())
            labels_list.append(labels)
            
    features_all = torch.cat(features_list)
    labels_all = torch.cat(labels_list)
    
    print(f"Saving {features_all.shape} to {save_path_features}")
    torch.save(features_all, save_path_features)
    torch.save(labels_all, save_path_labels)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Swin Backbone ---
    print("Loading Swin Transformer...")
    # Create model without classifier (num_classes=0)
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0)
    
    # Load your trained weights
    weights_path = './weights/cifar100_swin_tiny_backbone.pth'
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at {weights_path}")
        
    # Check if the saved weights contain 'head' or 'fc' layers and remove them if necessary
    state_dict = torch.load(weights_path, map_location=device)
    # Remove head keys if they exist to match num_classes=0 model
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # --- Data Loaders ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)

    # --- Extract and Save ---
    os.makedirs('./data/features', exist_ok=True)
    
    extract_and_save(train_loader, model, device, 
                     './data/features/train_features.pt', 
                     './data/features/train_labels.pt')
                     
    extract_and_save(test_loader, model, device, 
                     './data/features/test_features.pt', 
                     './data/features/test_labels.pt')

if __name__ == '__main__':
    main()