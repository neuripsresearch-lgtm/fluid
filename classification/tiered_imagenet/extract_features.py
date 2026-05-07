import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import timm
import os
import argparse
from tqdm import tqdm

def extract_and_save(loader, model, device, save_path_features, save_path_labels):
    features_list = []
    labels_list = []
    
    print(f"Extracting features to {save_path_features}...")
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            # Extract features (batch_size, 768)
            output = model(images)
            features_list.append(output.cpu())
            labels_list.append(labels)
            
    features_all = torch.cat(features_list)
    labels_all = torch.cat(labels_list)
    
    print(f"Saving {features_all.shape} features.")
    torch.save(features_all, save_path_features)
    torch.save(labels_all, save_path_labels)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Swin Backbone ---
    print("Loading Swin Transformer backbone...")
    # Create model without classifier (num_classes=0)
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0)
    
    if not os.path.exists(args.backbone_path):
        raise FileNotFoundError(f"Backbone weights not found at {args.backbone_path}")
        
    state_dict = torch.load(args.backbone_path, map_location=device)
    # Remove head keys if they exist
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # --- Data Loaders (ImageFolder) ---
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dir = os.path.join(args.data_path, 'train')
    test_dir = os.path.join(args.data_path, 'test')

    train_set = ImageFolder(train_dir, transform=transform)
    test_set = ImageFolder(test_dir, transform=transform)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True, type=str, help="Root dataset path")
    parser.add_argument('--backbone-path', required=True, type=str, help="Path to .pth backbone")
    args = parser.parse_args()
    main(args)