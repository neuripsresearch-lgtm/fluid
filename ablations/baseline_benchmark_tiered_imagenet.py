"""
baseline_benchmark_tiered_imagenet.py
======================================
Unified baseline comparison for Tiered-ImageNet (Table 1 / Table 3).
Mirrors baseline_benchmark_cifar100.py but uses:
  - torchvision.datasets.ImageFolder  (WNID-based folder structure)
  - NLTK WordNet for WNID → human name mapping
  - ImageNet normalisation statistics
  - 90/10 train/val split carved from the training set

Methods: Standard | Soft-1 | Soft-5 | Soft-10 | HXE | HIE | BiLT | BiLT+AIGDL

Usage
-----
    python baseline_benchmark_tiered_imagenet.py \\
        --wordnet-path ../../assets/tiered_imagenet/wordnet_tree.json \\
        --mytree-path  ../../assets/tiered_imagenet/cifar100_best_tree.json \\
        --data-path    /path/to/tiered_imagenet_standard \\
        --output-csv   ../../logs/benchmark_tiered_results.csv \\
        --epochs       20

Resume an interrupted sweep by passing the same --output-csv path;
already-completed (tree, method) rows are skipped automatically.

Dataset layout expected
-----------------------
    <data-path>/
        train/
            <wnid>/   (e.g., n01440764/)
                *.JPEG
        test/          (or val/)
            <wnid>/
"""

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

import nltk
from nltk.corpus import wordnet as wn


# ---------------------------------------------------------------------------
# WNID → human name
# ---------------------------------------------------------------------------

def _ensure_nltk():
    for corpus in ('wordnet', 'omw-1.4'):
        try:
            nltk.data.find(f'corpora/{corpus}.zip')
        except LookupError:
            nltk.download(corpus, quiet=True)


def wnid_to_name(wnid: str) -> str:
    try:
        synset = wn.synset_from_pos_and_offset('n', int(wnid[1:]))
        return synset.lemmas()[0].name()
    except Exception:
        return wnid


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ---------------------------------------------------------------------------
# Tree utilities  (self-contained, no external deps)
# ---------------------------------------------------------------------------

class Node:
    def __init__(self, name, parent=None):
        self.name = name; self.parent = parent
        self.children = []; self.height = 0; self.id = None

    def add_child(self, c): self.children.append(c)


def build_tree(data, parent=None):
    if isinstance(data, str):
        return Node(data.replace(' ', '_'), parent)
    node = Node(data.get('name', 'Unnamed').replace(' ', '_'), parent)
    for c in (data.get('children') or []):
        node.add_child(build_tree(c, node))
    return node


def index_tree_nodes(root):
    nodes = []
    def _t(n):
        n.id = len(nodes); nodes.append(n)
        for c in n.children: _t(c)
    _t(root); return nodes


def find_node(root, name):
    if root.name == name: return root
    for c in root.children:
        r = find_node(c, name)
        if r: return r
    return None


def get_ancestors(node):
    path, curr = [], node
    while curr: path.append(curr); curr = curr.parent
    return path


def get_depth(node):
    d, curr = 0, node
    while curr.parent: d += 1; curr = curr.parent
    return d


def find_lca(n1, n2):
    path1 = set(get_ancestors(n1)); curr = n2
    while curr:
        if curr in path1: return curr
        curr = curr.parent
    return None


def compute_node_heights(node):
    if not node.children: node.height = 0; return 0
    node.height = 1 + max(compute_node_heights(c) for c in node.children)
    return node.height


def get_average_leaf_depth(root):
    leaves = [n for n in index_tree_nodes(root) if not n.children]
    return sum(get_depth(l) for l in leaves) / len(leaves) if leaves else 1.0


def get_hierarchy_layers(root, classes):
    level_0 = [find_node(root, c) for c in classes]
    layers, curr = [level_0], level_0
    while True:
        parents = sorted({n.parent for n in curr if n and n.parent}, key=lambda x: x.name)
        if not parents: break
        layers.append(parents); curr = parents
        if len(parents) == 1: break
    return layers


def compute_layer_distance_matrices(layers):
    mats = []
    for nodes in layers:
        n = len(nodes); mat = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    lca = find_lca(nodes[i], nodes[j])
                    mat[i, j] = get_depth(nodes[i]) + get_depth(nodes[j]) - 2 * get_depth(lca)
        mats.append(mat)
    return mats


# ---------------------------------------------------------------------------
# Loss functions  (identical to CIFAR-100 version)
# ---------------------------------------------------------------------------

class SoftLabelLoss(nn.Module):
    def __init__(self, root, classes, beta, device):
        super().__init__()
        nodes = [find_node(root, c) for c in classes]
        N = len(classes)
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j and nodes[i] and nodes[j]:
                    lca = find_lca(nodes[i], nodes[j])
                    D[i, j] = get_depth(nodes[i]) + get_depth(nodes[j]) - 2*get_depth(lca)
                elif not (nodes[i] and nodes[j]):
                    D[i, j] = 100
        neg = -beta * D - (-beta * D).max(axis=1, keepdims=True)
        exp = np.exp(neg)
        self.soft_labels = torch.tensor(exp / exp.sum(axis=1, keepdims=True), dtype=torch.float32).to(device)

    def forward(self, logits, targets):
        return F.kl_div(F.log_softmax(logits, dim=1), self.soft_labels[targets], reduction='batchmean')


class HXELoss(nn.Module):
    def __init__(self, root, classes, device, alpha=0.5):
        super().__init__()
        all_nodes = index_tree_nodes(root)
        N, M = len(classes), len(all_nodes)
        mem = torch.zeros(M, N, device=device)
        def leaves(n):
            if not n.children: return [n.name]
            return [l for c in n.children for l in leaves(c)]
        for n in all_nodes:
            for ln in leaves(n):
                if ln in classes: mem[n.id, classes.index(ln)] = 1.0
        self.membership_matrix = mem
        self.path_info = []
        for cn in [find_node(root, c) for c in classes]:
            if not cn: self.path_info.append([]); continue
            ancs = get_ancestors(cn)
            self.path_info.append([(ancs[i].id, ancs[i+1].id, np.exp(-alpha*get_depth(ancs[i])))
                                    for i in range(len(ancs)-1)])

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets)
        probs = F.softmax(logits, dim=1)
        node_probs = torch.clamp(probs @ self.membership_matrix.t(), min=1e-9)
        hxe = sum(
            -w * torch.log(node_probs[b, ci] / node_probs[b, pi] + 1e-9)
            for b in range(logits.size(0))
            for ci, pi, w in self.path_info[targets[b].item()]
        )
        return ce + hxe / logits.size(0)


class HIELoss(nn.Module):
    def __init__(self, fine_to_coarse, device):
        super().__init__()
        self.f2c = torch.tensor(fine_to_coarse, device=device)
        self.ce = nn.CrossEntropyLoss()
    def forward(self, out, targets):
        lf, lc = out
        return self.ce(lf, targets) + self.ce(lc, self.f2c[targets])


class BiLTLoss(nn.Module):
    def __init__(self, num_levels, alpha=0.1):
        super().__init__()
        w = [math.exp(-alpha*i) for i in range(num_levels)]
        m = max(w); self.weights = [x/m for x in w]
        self.ce = nn.CrossEntropyLoss()
    def forward(self, logits_list, targets_list):
        return sum(self.weights[i]*self.ce(l, t)
                   for i,(l,t) in enumerate(zip(logits_list, targets_list)))


class BiLTAIGDLLoss(nn.Module):
    def __init__(self, num_levels, deltas, fixed_dists, alpha=0.1, beta=1.0, gamma=1.0, eps=0.1):
        super().__init__()
        w = [math.exp(-alpha*i) for i in range(num_levels)]
        m = max(w); self.weights = [x/m for x in w]
        self.deltas = deltas; self.fixed_dists = fixed_dists
        self.beta = beta; self.gamma = gamma; self.eps = eps
    def forward(self, logits_list, targets_list):
        total = 0.0
        for i,(logits,targets) in enumerate(zip(logits_list, targets_list)):
            ce = F.cross_entropy(logits, targets)
            util = self.beta * self.deltas[i] - self.fixed_dists[i]
            soft = F.softmax(self.gamma * util[targets], dim=1)
            kl = F.kl_div(F.log_softmax(logits, dim=1), soft, reduction='batchmean')
            total += self.weights[i] * ((1-self.eps)*ce + self.eps*kl)
        return total


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class BiLTModel(nn.Module):
    def __init__(self, backbone, dim, layer_dims, use_aigdl=False, fixed_dists=None):
        super().__init__()
        self.backbone = backbone
        self.head_fine = nn.Sequential(nn.BatchNorm1d(dim), nn.Linear(dim, layer_dims[0]),
                                       nn.BatchNorm1d(layer_dims[0]), nn.ELU(inplace=True))
        self.final_fine_proj = nn.Linear(layer_dims[0], layer_dims[0])
        self.coarse_heads = nn.ModuleList([
            nn.Sequential(nn.BatchNorm1d(layer_dims[i]), nn.Linear(layer_dims[i], layer_dims[i+1]),
                          nn.BatchNorm1d(layer_dims[i+1]), nn.ELU(inplace=True),
                          nn.Linear(layer_dims[i+1], layer_dims[i+1]))
            for i in range(len(layer_dims)-1)
        ])
        if use_aigdl:
            self.deltas = nn.ParameterList([nn.Parameter(torch.zeros(d,d)) for d in layer_dims])
            self.fixed_dists = []
            for i,d in enumerate(layer_dims):
                if fixed_dists: self.register_buffer(f'fd_{i}', fixed_dists[i]); self.fixed_dists.append(getattr(self,f'fd_{i}'))

    def forward(self, x):
        f = self.backbone.forward_features(x)
        if f.ndim > 2: f = f.mean(dim=list(range(1, f.ndim-1)))
        l0 = self.final_fine_proj(self.head_fine(f))
        outs = [l0]; curr = l0
        for h in self.coarse_heads: curr = h(curr); outs.append(curr)
        return outs

    def forward_features(self, x): return self.backbone.forward_features(x)


def get_model(method, num_classes, num_coarse=0, bilt_layers=None, fixed_dists=None):
    base = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_classes)
    if method == 'HIE':
        base.reset_classifier(0); dim = base.num_features
        class MH(nn.Module):
            def __init__(self, b, d, nf, nc):
                super().__init__(); self.backbone=b; self.hf=nn.Linear(d,nf); self.hc=nn.Linear(d,nc)
            def forward(self, x):
                f = self.backbone.forward_features(x)
                if f.ndim>2: f=f.mean(dim=list(range(1,f.ndim-1)))
                return self.hf(f), self.hc(f)
            def forward_features(self, x): return self.backbone.forward_features(x)
        return MH(base, dim, num_classes, num_coarse)
    elif 'BiLT' in method:
        base.reset_classifier(0)
        return BiLTModel(base, base.num_features, bilt_layers, method=='BiLT+AIGDL', fixed_dists)
    return base


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_alignment(model, loader, device, idx_to_class, tree_root, classes):
    model.eval(); feats = {c:[] for c in classes}
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            f = model.forward_features(imgs) if hasattr(model,'forward_features') else model.backbone.forward_features(imgs)
            if f.ndim>2: f=f.mean(dim=list(range(1,f.ndim-1)))
            for i,l in enumerate(lbls): feats[idx_to_class[l.item()]].append(f[i].cpu().numpy())
    centroids, valid = [], []
    for c in classes:
        if feats[c]: centroids.append(np.mean(np.vstack(feats[c]),axis=0)); valid.append(c)
    if len(centroids)<2: return 0.0
    vis = pdist(np.array(centroids), metric='cosine')
    n = len(valid); nodes = [find_node(tree_root, c) for c in valid]
    td = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            lca = find_lca(nodes[i],nodes[j])
            d = get_depth(nodes[i])+get_depth(nodes[j])-2*(get_depth(lca) if lca else 0)
            td[i,j]=td[j,i]=d
    return spearmanr(vis, squareform(td))[0]


def evaluate_metrics(model, loader, device, root, classes, fine_to_coarse_idx=None):
    model.eval()
    idx_to_class = {i:c for i,c in enumerate(classes)}
    avg_ld = get_average_leaf_depth(root); compute_node_heights(root)
    total=correct=0; mk={'lca_depth':0,'rel_depth':0,'dist':0,'count':0}
    rda=mhs=0.0; tkhs={1:0.,5:0.,20:0.}
    f2c = torch.tensor(fine_to_coarse_idx, device=device) if fine_to_coarse_idx and not isinstance(fine_to_coarse_idx,list) else None
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs=imgs.to(device); out=model(imgs)
            if isinstance(out,list): logits=out[0]
            elif isinstance(out,tuple):
                lf,lc=out
                logits = F.softmax(lf,dim=1)*(F.softmax(lc,dim=1)[:,f2c] if f2c is not None else 1)
            else: logits=out
            _,topk=logits.topk(20,dim=1)
            for i in range(len(lbls)):
                total+=1; t=lbls[i].item(); p=topk[i,0].item()
                nt=find_node(root,idx_to_class[t]); np_=find_node(root,idx_to_class[p])
                if t==p: correct+=1
                if nt and np_:
                    lca=find_lca(nt,np_); ld=get_depth(lca) if lca else 0
                    rd=ld/avg_ld if avg_ld>0 else 0; rda+=rd
                    if t!=p:
                        mk['count']+=1; mk['lca_depth']+=ld; mk['rel_depth']+=rd
                        mk['dist']+=get_depth(nt)+get_depth(np_)-2*ld
                        mhs+=lca.height if lca else 0
                if nt:
                    ks=0.0
                    for k in range(20):
                        nk=find_node(root,idx_to_class[topk[i,k].item()])
                        if nk:
                            lk=find_lca(nt,nk)
                            if lk: ks+=lk.height
                        if k+1 in tkhs: tkhs[k+1]+=ks/(k+1)
    cnt=max(mk['count'],1); acc=100*correct/total
    mrd=mk['rel_depth']/cnt
    return {'Accuracy':acc,'Avg LCA Depth (Mistake)':mk['lca_depth']/cnt,
            'Avg Dist to LCA':(mk['dist']/cnt)/2.,'Rel. LCA Depth (All)':rda/total,
            'Hierarchical Dist (Mistake)':mhs/cnt,
            'Avg Hierarchical Dist @ K=1':tkhs[1]/total,'Avg Hierarchical Dist @ K=5':tkhs[5]/total,
            'Avg Hierarchical Dist @ K=20':tkhs[20]/total,
            'Mistake-Only Rel Depth':mrd,'Master Metric':(acc/100.)*mrd}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training_and_eval(tree_path, tree_name, method, args, device,
                           train_loader, val_loader, classes):
    print(f"\n--- {tree_name} | {method} ---")
    with open(tree_path) as f: root = build_tree(json.load(f))
    eval_map=fine_to_coarse=bilt_maps=bilt_nodes=fd_tensors=None

    if method=='HIE':
        coarse=root.children; fine_to_coarse=[]
        cnodes=[find_node(root,c) for c in classes]
        for cn in cnodes:
            ancs=get_ancestors(cn)
            fc=next((i for i,co in enumerate(coarse) if co in ancs),-1)
            fine_to_coarse.append(max(0,fc))
        eval_map=fine_to_coarse
        model=get_model(method,len(classes),len(coarse)).to(device)
    elif 'BiLT' in method:
        layers=get_hierarchy_layers(root,classes); bilt_nodes=layers
        dims=[len(l) for l in layers]
        bilt_maps=[]
        for i in range(len(layers)-1):
            nd={n:idx for idx,n in enumerate(layers[i+1])}
            m=torch.zeros(len(layers[i]),dtype=torch.long)
            for idx,n in enumerate(layers[i]):
                if n and n.parent and n.parent in nd: m[idx]=nd[n.parent]
            bilt_maps.append(m.to(device))
        fd_tensors=[t.to(device) for t in compute_layer_distance_matrices(layers)]
        model=get_model(method,len(classes),bilt_layers=dims,fixed_dists=fd_tensors).to(device)
    else:
        model=get_model(method,len(classes)).to(device)

    if method=='Standard': criterion=nn.CrossEntropyLoss()
    elif method=='HXE':    criterion=HXELoss(root,classes,device)
    elif method=='HIE':    criterion=HIELoss(fine_to_coarse,device)
    elif method=='BiLT':   criterion=BiLTLoss(len(bilt_nodes))
    elif method=='BiLT+AIGDL': criterion=BiLTAIGDLLoss(len(bilt_nodes),model.deltas,model.fixed_dists)
    elif method.startswith('Soft-'): criterion=SoftLabelLoss(root,classes,int(method.split('-')[1]),device)
    else: raise ValueError(f"Unknown method {method}")

    opt=optim.AdamW(model.parameters(),lr=5e-5,weight_decay=0.05)
    sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    best_mm=-1.; best_state=None; patience=0

    for epoch in range(args.epochs):
        model.train()
        for imgs,lbls in tqdm(train_loader,desc=f"Ep {epoch+1}/{args.epochs}",leave=False):
            imgs,lbls=imgs.to(device),lbls.to(device); opt.zero_grad()
            out=model(imgs)
            if 'BiLT' in method:
                tl=[lbls]; curr=lbls
                for m in bilt_maps: curr=m[curr]; tl.append(curr)
                loss=criterion(out,tl)
            else: loss=criterion(out,lbls)
            loss.backward(); opt.step()
        sched.step()
        vm=evaluate_metrics(model,val_loader,device,root,classes,eval_map)
        mm=vm['Master Metric']
        print(f"  Ep {epoch+1}: Acc={vm['Accuracy']:.2f}% MRD={vm['Mistake-Only Rel Depth']:.4f} MM={mm:.4f}")
        if mm>best_mm: best_mm=mm; best_state=copy.deepcopy(model.state_dict()); patience=0
        else:
            patience+=1
            if patience>=5: print("  Early stop."); break

    model.load_state_dict(best_state)
    res=evaluate_metrics(model,val_loader,device,root,classes,eval_map)
    idx2c={i:c for i,c in enumerate(classes)}
    res['Tree-Visual Alignment']=calculate_alignment(model,val_loader,device,idx2c,root,classes)
    res['Tree']=tree_name; res['Method']=method
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MEAN, STD = [0.485,0.456,0.406], [0.229,0.224,0.225]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wordnet-path', required=True)
    parser.add_argument('--mytree-path',  required=True)
    parser.add_argument('--data-path', default='./data/tiered_imagenet_standard')
    parser.add_argument('--output-csv', default='benchmark_tiered_results.csv')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args=parser.parse_args()

    set_seed(args.seed); _ensure_nltk()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tf_train=transforms.Compose([transforms.RandomResizedCrop(224,scale=(0.8,1.0)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    tf_val=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                                transforms.ToTensor(),transforms.Normalize(MEAN,STD)])

    train_dir=os.path.join(args.data_path,'train')
    test_dir=os.path.join(args.data_path,'test')
    if not os.path.exists(test_dir):
        test_dir=os.path.join(args.data_path,'val')

    # Carve 90/10 val split from training set
    full_aug=torchvision.datasets.ImageFolder(train_dir, transform=tf_train)
    full_val=torchvision.datasets.ImageFolder(train_dir, transform=tf_val)
    n=len(full_aug); idx=list(range(n)); np.random.shuffle(idx)
    sp=int(args.val_split*n)
    train_loader=torch.utils.data.DataLoader(Subset(full_aug,idx[sp:]),
                                              batch_size=args.batch_size,shuffle=True,num_workers=4)
    val_loader  =torch.utils.data.DataLoader(Subset(full_val,idx[:sp]),
                                              batch_size=args.batch_size,shuffle=False,num_workers=4)

    # WNID → human name mapping (must match hierarchy JSON leaf names)
    classes=[wnid_to_name(w).replace(' ','_') for w in full_aug.classes]
    print(f"Detected {len(classes)} classes. Examples: {classes[:5]}")

    methods=['Standard','Soft-1','Soft-5','Soft-10','HXE','HIE','BiLT','BiLT+AIGDL']
    trees=[('WordNet',args.wordnet_path),('MyTree',args.mytree_path)]

    done=set()
    if os.path.exists(args.output_csv):
        with open(args.output_csv) as f:
            for row in csv.DictReader(f):
                if 'Tree' in row and 'Method' in row:
                    done.add((row['Tree'],row['Method']))

    for tname,tpath in trees:
        for method in methods:
            if (tname,method) in done:
                print(f"Skipping {tname} - {method}"); continue
            try:
                res=run_training_and_eval(tpath,tname,method,args,device,
                                          train_loader,val_loader,classes)
                exists=os.path.exists(args.output_csv)
                keys=['Tree','Method']+sorted(k for k in res if k not in ('Tree','Method'))
                with open(args.output_csv,'a',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=keys)
                    if not exists: w.writeheader()
                    w.writerow(res)
                done.add((tname,method))
                print(f"Saved {tname} - {method}")
            except Exception as e:
                import traceback; print(f"FAILED {tname}-{method}: {e}"); traceback.print_exc()

    print(f"\nDone. Results → {args.output_csv}")

if __name__=='__main__':
    main()
