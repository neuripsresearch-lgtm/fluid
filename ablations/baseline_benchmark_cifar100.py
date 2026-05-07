import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import timm
import argparse
import json
import numpy as np
import random
import os
import csv
import copy
import math
from tqdm import tqdm
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform, pdist
from torch.utils.data import Subset

# ==========================================
# 1. UTILITIES & TREE LOGIC
# ==========================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class Node:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.height = 0
        self.id = None 

    def add_child(self, child_node):
        self.children.append(child_node)

def build_tree(data, parent=None):
    if isinstance(data, str):
        node = Node(data.replace(' ', '_'), parent)
        return node
    node = Node(data.get('name', 'Unnamed').replace(' ', '_'), parent)
    if 'children' in data and data['children']:
        for child_data in data['children']:
            child_node = build_tree(child_data, node)
            node.add_child(child_node)
    return node

def index_tree_nodes(root):
    nodes = []
    def _traverse(n):
        n.id = len(nodes)
        nodes.append(n)
        for c in n.children: _traverse(c)
    _traverse(root)
    return nodes

def find_node(root, name):
    if root.name == name: return root
    for child in root.children:
        res = find_node(child, name)
        if res: return res
    return None

def get_ancestors(node):
    path = []
    curr = node
    while curr:
        path.append(curr)
        curr = curr.parent
    return path

def get_depth(node):
    d = 0
    curr = node
    while curr.parent:
        d += 1
        curr = curr.parent
    return d

def find_lca(node1, node2):
    path1 = set(get_ancestors(node1))
    curr = node2
    while curr:
        if curr in path1: return curr
        curr = curr.parent
    return None

def compute_node_heights(node):
    if not node.children:
        node.height = 0
        return 0
    max_h = 0
    for c in node.children:
        max_h = max(max_h, compute_node_heights(c))
    node.height = max_h + 1
    return node.height

def get_average_leaf_depth(root):
    leaves = [n for n in index_tree_nodes(root) if not n.children]
    if not leaves: return 1.0
    return sum(get_depth(l) for l in leaves) / len(leaves)

def get_hierarchy_layers(root, classes):
    """
    Analyzes the tree to create layers for BiLT.
    Returns:
        layers: List of Lists [ [Level0_Nodes], [Level1_Nodes], ... ]
    """
    level_0_nodes = [find_node(root, c.replace(' ', '_')) for c in classes]
    layers = [level_0_nodes]
    
    curr_nodes = level_0_nodes
    while True:
        parents = set()
        for n in curr_nodes:
            if n and n.parent:
                parents.add(n.parent)
        
        if not parents:
            break
            
        sorted_parents = sorted(list(parents), key=lambda x: x.name)
        
        # Stop if we reached the absolute root if it's a single dummy root
        if len(sorted_parents) == 1 and len(curr_nodes) == 1:
            break
            
        layers.append(sorted_parents)
        curr_nodes = sorted_parents
        if len(sorted_parents) == 1:
            break
            
    return layers

def compute_layer_distance_matrices(layers):
    """
    Computes static distance matrix D for each layer.
    Returns list of Tensors.
    """
    matrices = []
    for nodes in layers:
        n = len(nodes)
        mat = torch.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    mat[i, j] = 0
                else:
                    lca = find_lca(nodes[i], nodes[j])
                    # Standard tree distance
                    d = get_depth(nodes[i]) + get_depth(nodes[j]) - 2 * get_depth(lca)
                    mat[i, j] = d
        matrices.append(mat)
    return matrices

# ==========================================
# 2. LOSS FUNCTIONS
# ==========================================

class SoftLabelLoss(nn.Module):
    def __init__(self, root, classes, beta, device):
        super().__init__()
        self.beta = beta
        self.num_classes = len(classes)
        nodes = [find_node(root, c.replace(' ', '_')) for c in classes]
        dist_matrix = np.zeros((self.num_classes, self.num_classes))
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i == j: dist_matrix[i, j] = 0
                elif nodes[i] and nodes[j]:
                    lca = find_lca(nodes[i], nodes[j])
                    dist = get_depth(nodes[i]) + get_depth(nodes[j]) - 2 * get_depth(lca)
                    dist_matrix[i, j] = dist
                else: 
                    dist_matrix[i, j] = 100 
        neg_dist = -self.beta * dist_matrix
        neg_dist = neg_dist - np.max(neg_dist, axis=1, keepdims=True)
        exp_dist = np.exp(neg_dist)
        self.soft_labels = torch.tensor(exp_dist / np.sum(exp_dist, axis=1, keepdims=True), dtype=torch.float32).to(device)

    def forward(self, logits, targets):
        target_dist = self.soft_labels[targets]
        log_probs = F.log_softmax(logits, dim=1)
        return F.kl_div(log_probs, target_dist, reduction='batchmean')

class HXELoss(nn.Module):
    def __init__(self, root, classes, device, alpha=0.5):
        super().__init__()
        self.device = device
        self.alpha = alpha
        self.classes = classes
        self.num_classes = len(classes)
        self.all_nodes = index_tree_nodes(root)
        num_total_nodes = len(self.all_nodes)
        
        self.membership_matrix = torch.zeros((num_total_nodes, self.num_classes), device=device)
        def get_leaves_names(n):
            if not n.children: return [n.name]
            l = []
            for c in n.children: l.extend(get_leaves_names(c))
            return l

        for n in self.all_nodes:
            leaves = get_leaves_names(n)
            for l_name in leaves:
                if l_name in classes:
                    c_idx = classes.index(l_name)
                    self.membership_matrix[n.id, c_idx] = 1.0

        self.path_info = [] 
        class_nodes = [find_node(root, c.replace(' ', '_')) for c in classes]
        
        for c_node in class_nodes:
            if not c_node:
                self.path_info.append([])
                continue
            ancestors = get_ancestors(c_node)
            step_data = []
            for i in range(len(ancestors) - 1):
                child = ancestors[i]
                parent = ancestors[i+1]
                d = get_depth(child)
                w = np.exp(-self.alpha * d)
                step_data.append((child.id, parent.id, w))
            self.path_info.append(step_data)

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets)
        probs = F.softmax(logits, dim=1)
        node_probs = torch.matmul(probs, self.membership_matrix.t())
        node_probs = torch.clamp(node_probs, min=1e-9) 
        hxe_loss = 0.0
        batch_size = logits.size(0)
        for b in range(batch_size):
            t = targets[b].item()
            path_steps = self.path_info[t]
            for (child_id, parent_id, weight) in path_steps:
                p_child = node_probs[b, child_id]
                p_parent = node_probs[b, parent_id]
                cond_prob = p_child / p_parent
                hxe_loss -= weight * torch.log(cond_prob + 1e-9)
        return ce_loss + (hxe_loss / batch_size)

class HIELoss(nn.Module):
    def __init__(self, fine_to_coarse_map, device):
        super().__init__()
        self.fine_to_coarse = torch.tensor(fine_to_coarse_map, device=device)
        self.ce = nn.CrossEntropyLoss()
    def forward(self, logits_tuple, targets_fine):
        logits_fine, logits_coarse = logits_tuple
        targets_coarse = self.fine_to_coarse[targets_fine]
        loss_f = self.ce(logits_fine, targets_fine)
        loss_c = self.ce(logits_coarse, targets_coarse)
        return loss_f + loss_c

class BiLTLoss(nn.Module):
    def __init__(self, num_levels, alpha=0.1):
        super().__init__()
        self.weights = [math.exp(-alpha * i) for i in range(num_levels)]
        max_w = max(self.weights)
        self.weights = [w / max_w for w in self.weights]
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits_list, targets_list):
        total_loss = 0.0
        for i, (logits, targets) in enumerate(zip(logits_list, targets_list)):
            layer_loss = self.ce(logits, targets)
            total_loss += self.weights[i] * layer_loss
        return total_loss

class BiLTAIGDLLoss(nn.Module):
    def __init__(self, num_levels, model_deltas, model_fixed_dists, alpha=0.1, beta=1.0, gamma=1.0, epsilon=0.1):
        super().__init__()
        self.weights = [math.exp(-alpha * i) for i in range(num_levels)]
        max_w = max(self.weights)
        self.weights = [w / max_w for w in self.weights]
        
        self.deltas = model_deltas # Reference to learnable params in model
        self.fixed_dists = model_fixed_dists # Reference to buffers in model
        
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, logits_list, targets_list):
        total_loss = 0.0
        for i, (logits, targets) in enumerate(zip(logits_list, targets_list)):
            # 1. Standard CE (Hard Labels)
            ce_loss = F.cross_entropy(logits, targets)
            
            # 2. AIGDL Adaptive Smoothing (Soft Labels)
            # Utility = beta * Delta - D
            delta = self.deltas[i]
            dist = self.fixed_dists[i]
            utility = self.beta * delta - dist
            
            # Soft targets: softmax(gamma * utility[target_idx])
            target_utilities = utility[targets] 
            soft_targets = F.softmax(self.gamma * target_utilities, dim=1)
            
            # KL Divergence
            log_probs = F.log_softmax(logits, dim=1)
            aigdl_loss = F.kl_div(log_probs, soft_targets, reduction='batchmean')
            
            # Combined Loss
            layer_loss = (1 - self.epsilon) * ce_loss + self.epsilon * aigdl_loss
            
            total_loss += self.weights[i] * layer_loss
            
        return total_loss

# ==========================================
# 3. MODELS
# ==========================================

class BiLTModel(nn.Module):
    def __init__(self, backbone, dim, layer_dims, use_aigdl=False, fixed_dists=None):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.layer_dims = layer_dims
        self.use_aigdl = use_aigdl
        
        # Finest level head
        self.head_fine = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Linear(dim, layer_dims[0]),
            nn.BatchNorm1d(layer_dims[0]),
            nn.ELU(inplace=True)
        )
        self.final_fine_proj = nn.Linear(layer_dims[0], layer_dims[0])
        
        # Coarse heads
        self.coarse_heads = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i+1]
            block = nn.Sequential(
                nn.BatchNorm1d(in_dim),
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ELU(inplace=True),
                nn.Linear(out_dim, out_dim)
            )
            self.coarse_heads.append(block)
            
        # AIGDL Components (Parameters & Buffers)
        if self.use_aigdl:
            self.deltas = nn.ParameterList()
            self.fixed_dists = []
            for i, dim in enumerate(layer_dims):
                # Learnable Delta (initialized to 0)
                delta = nn.Parameter(torch.zeros(dim, dim))
                self.deltas.append(delta)
                
                # Fixed Dist (Buffer)
                if fixed_dists is not None:
                    self.register_buffer(f'fixed_dist_{i}', fixed_dists[i])
                    self.fixed_dists.append(getattr(self, f'fixed_dist_{i}'))

    def forward(self, x):
        features = self.backbone.forward_features(x)
        if features.ndim == 4: 
            features = features.mean(dim=(1, 2))
        elif features.ndim == 3:
            features = features.mean(dim=1)
            
        # Level 0 (Fine)
        logits_0_feat = self.head_fine(features)
        logits_0 = self.final_fine_proj(logits_0_feat)
        
        outputs = [logits_0]
        curr_logits = logits_0
        
        for head in self.coarse_heads:
            next_logits = head(curr_logits)
            outputs.append(next_logits)
            curr_logits = next_logits
            
        # Standard inference for all methods (No CRM)
        return outputs

def get_model(method, num_classes, num_coarse=0, bilt_layers=None, fixed_dists=None):
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_classes)
    
    if method == 'HIE':
        model.reset_classifier(0) 
        dim = model.num_features 
        
        class MultiHeadSwin(nn.Module):
            def __init__(self, backbone, dim, n_fine, n_coarse):
                super().__init__()
                self.backbone = backbone
                self.head_fine = nn.Linear(dim, n_fine)
                self.head_coarse = nn.Linear(dim, n_coarse)
            def forward(self, x):
                features = self.backbone.forward_features(x)
                if features.ndim == 4: features = features.mean(dim=(1, 2))
                elif features.ndim == 3: features = features.mean(dim=1)
                return self.head_fine(features), self.head_coarse(features)
            def forward_features(self, x): return self.backbone.forward_features(x)
                
        return MultiHeadSwin(model, dim, num_classes, num_coarse)
    
    elif method == 'BiLT' or method == 'BiLT+AIGDL':
        model.reset_classifier(0)
        dim = model.num_features
        use_aigdl = (method == 'BiLT+AIGDL')
        return BiLTModel(model, dim, bilt_layers, use_aigdl, fixed_dists)
    
    return model

# ==========================================
# 4. METRIC CALCULATION
# ==========================================

def calculate_alignment(model, loader, device, idx_to_class, tree_root, classes):
    model.eval()
    class_features = {c: [] for c in classes}
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            if hasattr(model, 'backbone'): features = model.backbone.forward_features(images)
            else: features = model.forward_features(images)
            if features.ndim == 4: features = features.mean(dim=(1, 2))
            elif features.ndim == 3: features = features.mean(dim=1)
            for i in range(len(labels)):
                lbl = labels[i].item()
                c_name = idx_to_class[lbl]
                class_features[c_name].append(features[i].cpu().numpy())

    centroids = []
    valid_classes = [] 
    for c in classes:
        if class_features[c]:
            centroids.append(np.mean(np.vstack(class_features[c]), axis=0))
            valid_classes.append(c)
    
    centroids = np.array(centroids)
    if len(centroids) < 2: return 0.0

    vis_dists = pdist(centroids, metric='cosine')
    n = len(valid_classes)
    tree_dists = np.zeros((n, n))
    nodes = [find_node(tree_root, c) for c in valid_classes]
    for i in range(n):
        for j in range(i + 1, n):
            d = get_depth(nodes[i]) + get_depth(nodes[j]) - 2 * get_depth(find_lca(nodes[i], nodes[j]))
            tree_dists[i, j] = d
            tree_dists[j, i] = d
    return spearmanr(vis_dists, squareform(tree_dists))[0]

def evaluate_metrics(model, loader, device, root, classes, fine_to_coarse_idx=None):
    model.eval()
    idx_to_class = {i: c.replace(' ', '_') for i, c in enumerate(classes)}
    avg_leaf_depth = get_average_leaf_depth(root)
    compute_node_heights(root)

    total = 0
    correct = 0
    mistake_stats = {'lca_depth': 0, 'rel_depth': 0, 'dist': 0, 'count': 0}
    rel_depth_all = 0.0 
    mistake_height_sum = 0.0
    topk_height_sums = {1: 0.0, 5: 0.0, 20: 0.0}
    
    fine_to_coarse_tensor = None
    if fine_to_coarse_idx is not None and not isinstance(fine_to_coarse_idx, list): 
        fine_to_coarse_tensor = torch.tensor(fine_to_coarse_idx, device=device)
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            output = model(images)
            logits = None
            if isinstance(output, list): 
                # BiLT/BiLT+AIGDL: Always use Fine Logits (Index 0) for evaluation
                logits = output[0] 
            elif isinstance(output, tuple): 
                # HIE
                logits_fine, logits_coarse = output
                if fine_to_coarse_tensor is not None:
                    p_fine = F.softmax(logits_fine, dim=1)
                    p_coarse = F.softmax(logits_coarse, dim=1)
                    multiplier = p_coarse[:, fine_to_coarse_tensor] 
                    logits = p_fine * multiplier
                else: logits = logits_fine
            else: logits = output

            _, topk_preds = logits.topk(20, dim=1)
            
            for i in range(len(labels)):
                total += 1
                t_idx = labels[i].item()
                p_idx = topk_preds[i, 0].item()
                node_t = find_node(root, idx_to_class[t_idx])
                node_p = find_node(root, idx_to_class[p_idx])
                
                if t_idx == p_idx: correct += 1
                
                if node_t and node_p:
                    lca = find_lca(node_t, node_p)
                    lca_d = get_depth(lca) if lca else 0
                    rel_d = lca_d / avg_leaf_depth if avg_leaf_depth > 0 else 0
                    rel_depth_all += rel_d
                    if t_idx != p_idx:
                        mistake_stats['count'] += 1
                        mistake_stats['lca_depth'] += lca_d
                        mistake_stats['rel_depth'] += rel_d
                        d_t = get_depth(node_t)
                        d_p = get_depth(node_p)
                        mistake_stats['dist'] += (d_t + d_p - 2*lca_d)
                        mistake_height_sum += lca.height if lca else 0
                if node_t:
                    curr_sample_k_sum = 0.0
                    for k in range(20):
                        curr_p_idx = topk_preds[i, k].item()
                        node_curr_p = find_node(root, idx_to_class[curr_p_idx])
                        if node_curr_p:
                            curr_lca = find_lca(node_t, node_curr_p)
                            if curr_lca: curr_sample_k_sum += curr_lca.height
                        rank = k + 1
                        if rank in topk_height_sums:
                            topk_height_sums[rank] += (curr_sample_k_sum / rank)

    cnt = mistake_stats['count'] if mistake_stats['count'] > 0 else 1
    accuracy = 100 * correct / total
    mistake_only_rel_depth = mistake_stats['rel_depth'] / cnt
    master_metric = (accuracy / 100.0) * mistake_only_rel_depth

    metrics = {
        'Accuracy': accuracy,
        'Avg LCA Depth (Mistake)': mistake_stats['lca_depth'] / cnt,
        'Avg Dist to LCA': (mistake_stats['dist'] / cnt) / 2.0,
        'Rel. LCA Depth (All)': rel_depth_all / total,
        'Hierarchical Dist (Mistake)': mistake_height_sum / cnt,
        'Avg Hierarchical Dist @ K=1': topk_height_sums[1] / total,
        'Avg Hierarchical Dist @ K=5': topk_height_sums[5] / total,
        'Avg Hierarchical Dist @ K=20': topk_height_sums[20] / total,
        'Mistake-Only Rel Depth': mistake_only_rel_depth,
        'Master Metric': master_metric
    }
    return metrics

# ==========================================
# 5. MAIN BENCHMARK LOOP
# ==========================================

def run_training_and_eval(tree_path, tree_name, method, args, device, train_loader, val_loader, classes):
    print(f"\n--- Running: {tree_name} | {method} ---")
    with open(tree_path, 'r') as f:
        tree_data = json.load(f)
    root = build_tree(tree_data)
    
    eval_map = None
    fine_to_coarse_idx = None
    bilt_maps = None
    bilt_layers_nodes = None
    fixed_dists_tensors = None
    
    if method == 'HIE':
        coarse_nodes = root.children
        num_coarse = len(coarse_nodes)
        fine_to_coarse_idx = []
        class_nodes = [find_node(root, c.replace(' ', '_')) for c in classes]
        for c_node in class_nodes:
            ancestors = get_ancestors(c_node)
            found_coarse = -1
            for i, coarse_n in enumerate(coarse_nodes):
                if coarse_n in ancestors:
                    found_coarse = i
                    break
            fine_to_coarse_idx.append(max(0, found_coarse))
        eval_map = fine_to_coarse_idx
        model = get_model(method, len(classes), num_coarse).to(device)

    elif 'BiLT' in method:
        layers = get_hierarchy_layers(root, classes)
        bilt_layers_nodes = layers
        layer_dims = [len(l) for l in layers]
        
        # Maps for targets
        bilt_maps = []
        for i in range(len(layers) - 1):
            curr_layer = layers[i]
            next_layer = layers[i+1]
            next_layer_dict = {n: idx for idx, n in enumerate(next_layer)}
            map_tensor = torch.zeros(len(curr_layer), dtype=torch.long)
            for idx, node in enumerate(curr_layer):
                if node and node.parent and node.parent in next_layer_dict:
                    map_tensor[idx] = next_layer_dict[node.parent]
                else: map_tensor[idx] = 0
            bilt_maps.append(map_tensor.to(device))
            
        # Fixed distances for AIGDL
        fixed_dists_list = compute_layer_distance_matrices(layers)
        fixed_dists_tensors = [t.to(device) for t in fixed_dists_list]
            
        model = get_model(method, len(classes), bilt_layers=layer_dims, fixed_dists=fixed_dists_tensors).to(device)

    else:
        model = get_model(method, len(classes)).to(device)
    
    # --- LOSS SETUP ---
    if method == 'Standard': criterion = nn.CrossEntropyLoss()
    elif method == 'HXE': criterion = HXELoss(root, classes, device, alpha=0.5)
    elif method == 'HIE': criterion = HIELoss(fine_to_coarse_idx, device)
    elif method == 'BiLT': criterion = BiLTLoss(len(bilt_layers_nodes), alpha=0.1)
    elif method == 'BiLT+AIGDL': 
        criterion = BiLTAIGDLLoss(len(bilt_layers_nodes), model.deltas, model.fixed_dists, alpha=0.1, beta=1.0, gamma=1.0, epsilon=0.1)
    elif method.startswith('Soft-'):
        beta = int(method.split('-')[1])
        criterion = SoftLabelLoss(root, classes, beta, device)
    else: raise ValueError(f"Unknown method {method}")

    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_master_metric = -1.0
    best_model_state = None
    best_epoch = -1
    patience_counter = 0
    patience_limit = 5 

    for epoch in range(args.epochs):
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            
            loss = None
            if 'BiLT' in method:
                targets_list = [labels]
                curr_targets = labels
                for m_map in bilt_maps:
                    curr_targets = m_map[curr_targets]
                    targets_list.append(curr_targets)
                loss = criterion(output, targets_list)
            else:
                loss = criterion(output, labels)
                
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_metrics = evaluate_metrics(model, val_loader, device, root, classes, eval_map)
        current_master = val_metrics['Master Metric']
        print(f"  Ep {epoch+1}: Acc={val_metrics['Accuracy']:.2f}%, MistakeRel={val_metrics['Mistake-Only Rel Depth']:.4f}, Master={current_master:.4f}")

        if current_master > best_master_metric:
            best_master_metric = current_master
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_model_state is not None:
        print(f"Reloading best weights from Epoch {best_epoch} (Master Metric: {best_master_metric:.4f})")
        model.load_state_dict(best_model_state)

    print("Generating Final Metrics...")
    final_metrics = evaluate_metrics(model, val_loader, device, root, classes, eval_map)
    idx_to_class = {i: c.replace(' ', '_') for i, c in enumerate(classes)}
    alignment = calculate_alignment(model, val_loader, device, idx_to_class, root, classes)
    final_metrics['Tree-Visual Alignment'] = alignment
    final_metrics['Tree'] = tree_name
    final_metrics['Method'] = method
    return final_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wordnet-path', required=True)
    parser.add_argument('--mytree-path', required=True)
    parser.add_argument('--data-path', default='./data')
    parser.add_argument('--output-csv', default='benchmark_results.csv')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Preparing Data...")
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    full_set_train_aug = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
    full_set_val_aug = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_val)
    classes = full_set_train_aug.classes
    num_train = len(full_set_train_aug)
    indices = list(range(num_train))
    np.random.shuffle(indices) 
    split = int(np.floor(0.1 * num_train))
    train_idx, val_idx = indices[split:], indices[:split]
    train_loader = torch.utils.data.DataLoader(Subset(full_set_train_aug, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(Subset(full_set_val_aug, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=4)

    methods = ['Standard', 'Soft-1', 'Soft-5', 'Soft-10', 'HXE', 'HIE', 'BiLT', 'BiLT+AIGDL']
    trees = [('WordNet', args.wordnet_path), ('MyTree', args.mytree_path)]
    
    completed_runs = set()
    if os.path.exists(args.output_csv):
        print(f"Checking existing results in {args.output_csv}...")
        try:
            with open(args.output_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'Tree' in row and 'Method' in row:
                        completed_runs.add((row['Tree'], row['Method']))
        except Exception as e:
            print(f"Warning: Could not read existing CSV ({e}). Starting fresh.")

    for tree_name, tree_path in trees:
        for method in methods:
            if (tree_name, method) in completed_runs:
                print(f"Skipping {tree_name} - {method} (Found in {args.output_csv})")
                continue
            try:
                res = run_training_and_eval(tree_path, tree_name, method, args, device, train_loader, val_loader, classes)
                file_exists = os.path.exists(args.output_csv)
                keys = ['Tree', 'Method'] + sorted([k for k in res.keys() if k not in ['Tree', 'Method']])
                with open(args.output_csv, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    if not file_exists: writer.writeheader()
                    writer.writerow(res)
                completed_runs.add((tree_name, method))
                print(f"Saved results for {tree_name} - {method}")
            except Exception as e:
                print(f"FAILED {tree_name} - {method}: {e}")
                import traceback
                traceback.print_exc()
    print(f"\nBenchmark Complete. Results saved to {args.output_csv}")

if __name__ == '__main__':
    main()