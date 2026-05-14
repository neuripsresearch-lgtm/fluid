# Fluid Hierarchies: Mistake-guided Neuro-Symbolic Semantic Alignment of Class Hierarchies

Anonymous NeurIPS 2026 submission.

---

## Repository Structure

```
fluid-hierarchies-submission/
│
├── README.md
├── requirements.txt
│
├── assets/                          ← All data assets (classes, trees, prompts)
│   ├── cifar100/
│   │   ├── cifar100_classes.txt
│   │   ├── wordnet_tree.json        ← Static WordNet baseline hierarchy
│   │   ├── cifar100_best_tree.json  ← Best learned Fluid Hierarchy tree
│   │   ├── tree_haf.json            ← HAF baseline tree
│   │   └── prompts/
│   │       ├── prompt_generate.txt
│   │       ├── prompt_critic.txt
│   │       ├── prompt_edit.txt
│   │       └── prompt_single_llm_refine.txt  ← Single-LLM ablation prompt
│   ├── fer2013/
│   │   ├── fer2013_classes.txt
│   │   └── prompts/
│   │       ├── prompt_generate.txt  ← Ab initio induction prompt
│   │       ├── prompt_critic.txt
│   │       └── prompt_edit.txt
│   └── tiered_imagenet/
│       ├── tiered_imagenet_classes.txt
│       ├── wordnet_tree.json
│       ├── tree_haf.json
│       ├── tree_imagenet_og.json
│       └── prompts/
│           ├── prompt_generate.txt
│           ├── prompt_critic.txt
│           └── prompt_edit.txt
│
├── classification/
│   │
│   ├── cifar100/                    ← CIFAR-100 main pipeline
│   │   ├── pipeline.py              ← Iterative Critic-Editor refinement loop
│   │   ├── llm_handler.py           ← LLM API wrapper (Gemini)
│   │   ├── train_hierarchical_classifier.py  ← Swin + Hierarchical Soft Labels
│   │   ├── evaluate_hierarchical.py ← Full hierarchy-aware metric suite
│   │   ├── spearman.py              ← Tree-Visual Alignment (GloVe)
│   │   ├── eval_flat_model.py       ← Flat baseline evaluation
│   │   ├── train_swin_backbone.py   ← Backbone pre-training
│   │   ├── extract_features.py      ← Feature extraction
│   │   └── experiments/             ← Individual baseline training scripts
│   │       ├── flat_swin.py
│   │       ├── soft_labels_swin.py
│   │       ├── mbm_swin.py
│   │       └── evaluate.py
│   │
│   ├── fer2013/                     ← FER-2013 ab initio experiment
│   │   ├── pipeline.py
│   │   ├── llm_handler.py
│   │   ├── train_hierarchical_classifier.py
│   │   └── evaluate_hierarchical.py
│   │
│   └── tiered_imagenet/             ← Tiered-ImageNet pipeline
│       ├── pipeline.py
│       ├── llm_handler.py
│       ├── train_hierarchical_classifier.py  ← HAF node-level classifiers
│       ├── evaluate_hierarchical.py
│       ├── spearman.py
│       ├── eval_flat_model.py
│       ├── train_swin_backbone.py
│       ├── train_resnet_backbone.py
│       ├── extract_features.py
│       ├── utils.py                 ← WNID→human name (NLTK WordNet)
│       ├── wnid_search.py
│       ├── restructure_imagenet.py
│       └── experiments/             ← Individual baseline training scripts
│           ├── train_hxe_swin.py
│           ├── train_hie_swin.py
│           └── soft_labels_swin.py
│
├── ablations/                       ← Ablation & extension experiments
│   ├── baseline_benchmark_cifar100.py       ← Unified sweep: all methods × trees (CIFAR-100)
│   ├── baseline_benchmark_tiered_imagenet.py ← Unified sweep: all methods × trees (Tiered-ImageNet)
│   ├── pipeline_no_critic.py        ← Critic-removed ablation
│   ├── llm_handler_editor_only.py   ← LLM handler for editor-only ablation
│   │
│   ├── pipeline.py                  ← Dense prediction (segmentation) pipeline
│   ├── llm_handler.py               ← LLM handler for segmentation
│   ├── critic_utils.py              ← Hierarchy metrics for segmentation
│   ├── tree_utils.py                ← Tree I/O for segmentation
│   ├── smith_loader.py              ← HELEN dataset loader
│   ├── vistas_loader.py             ← Mapillary Vistas dataset loader
│   ├── validate_seg.py              ← Segmentation evaluation
│   ├── config_smith_faces.yml
│   ├── config_mapillary_vistas.yml
│   └── assets/                      ← Segmentation class lists & hierarchies
│       ├── smith_faces_classes.txt
│       ├── smith_faces_hierarchy.txt
│       ├── mapillary_vistas_classes.txt
│       ├── mapillary_vistas_hierarchy.txt
│       ├── prompt_critic.txt
│       └── prompt_main_edit.txt
│
└── analysis/                        ← Post-hoc analysis tools
    ├── spearman_cifar100.py         ← Tree-Visual Alignment (GloVe, CIFAR-100)
    ├── spearman_tiered_imagenet.py  ← Tree-Visual Alignment (GloVe, Tiered-ImageNet)
    ├── embed_tree_builder.py        ← Bengio et al. (2010) spectral baseline tree
    ├── render_graph.py              ← Hierarchy visualisation
    └── tree_changes.py              ← Tree evolution analysis across iterations
```

---

## Installation

```bash
conda create -n fluid-hierarchies python=3.10 -y
conda activate fluid-hierarchies

# PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip install -r requirements.txt

# NLTK WordNet data (needed for Tiered-ImageNet WNID mapping)
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# Set Gemini API key (required for Critic-Editor loop only)
export GEMINI_API_KEY="your_key_here"
```

---

## Datasets

| Dataset | Used in | Download |
|---|---|---|
| CIFAR-100 | Main pipeline | Auto-downloaded by torchvision |
| Tiered-ImageNet | Main pipeline | [few-shot-ssl-public](https://github.com/renmengye/few-shot-ssl-public) |
| FER-2013 | Ab initio experiment | [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) |
| HELEN (Smith Faces) | Segmentation ablation | [HELEN dataset](http://www.ifp.illinois.edu/~vuongle2/helen/) |
| Mapillary Vistas | Segmentation ablation | [Mapillary](https://www.mapillary.com/dataset/vistas) |

**Tiered-ImageNet expected layout:**
```
data/tiered_imagenet_standard/
    train/<wnid>/*.JPEG
    test/<wnid>/*.JPEG
```

---

## Usage

### CIFAR-100 Pipeline

```bash
cd classification/cifar100

# Full iterative refinement pipeline
python pipeline.py \
    --hierarchy-path ../../assets/cifar100/wordnet_tree.json \
    --classes-file   ../../assets/cifar100/cifar100_classes.txt

# Evaluate best model
python evaluate_hierarchical.py \
    --hierarchy-path ../../assets/cifar100/cifar100_best_tree.json \
    --model-path     weights/best_model.pth \
    --split          test
```

### Tiered-ImageNet Pipeline

```bash
cd classification/tiered_imagenet

# Step 1: Extract features with Swin backbone
python extract_features.py --data-path /path/to/tiered_imagenet_standard

# Step 2: Run the full refinement pipeline
python pipeline.py \
    --hierarchy-path ../../assets/tiered_imagenet/wordnet_tree.json \
    --data-path      /path/to/tiered_imagenet_standard
```

### FER-2013 (Ab Initio — no pre-existing hierarchy)

```bash
cd classification/fer2013
python pipeline.py  # generates initial tree from label names alone
```

### Baseline Benchmark

```bash
cd ablations

# CIFAR-100: all methods × WordNet + learned tree in one sweep
python baseline_benchmark_cifar100.py \
    --wordnet-path ../assets/cifar100/wordnet_tree.json \
    --mytree-path  ../assets/cifar100/cifar100_best_tree.json \
    --output-csv   ../logs/benchmark_cifar100.csv \
    --epochs 20

# Tiered-ImageNet: same sweep
python baseline_benchmark_tiered_imagenet.py \
    --wordnet-path ../assets/tiered_imagenet/wordnet_tree.json \
    --mytree-path  ../assets/tiered_imagenet/wordnet_tree.json \
    --data-path    /path/to/tiered_imagenet_standard \
    --output-csv   ../logs/benchmark_tiered.csv \
    --epochs 20
```

Both scripts are **checkpoint-resumable** — interrupted runs restart from the CSV.  
Methods covered: `Standard | Soft-1 | Soft-5 | Soft-10 | HXE | HIE | BiLT | BiLT+AIGDL`

### Dense Prediction (Segmentation)

```bash
cd ablations

# HELEN faces
python pipeline.py --config config_smith_faces.yml

# Mapillary Vistas
python pipeline.py --config config_mapillary_vistas.yml
```

### Critic-Ablation (Editor-only, no Critic)

```bash
cd ablations
python pipeline_no_critic.py --max-iter 20 --patience 5
```

### Analysis

```bash
# Tree-Visual Alignment (Spearman ρ vs GloVe)
python analysis/spearman_cifar100.py \
    --tree     assets/cifar100/cifar100_best_tree.json \
    --baseline assets/cifar100/wordnet_tree.json \
    --classes  assets/cifar100/cifar100_classes.txt \
    --glove    assets/glove.6B.100d.txt   # download from nlp.stanford.edu/projects/glove

# Bengio et al. (2010) confusion-matrix baseline tree
python analysis/embed_tree_builder.py \
    --classes          assets/cifar100/cifar100_classes.txt \
    --confusion_matrix logs/confusion_matrix.npy \
    --output           assets/cifar100/bengio_tree.json
```

---

## Metrics Reference

| Metric | Description |
|---|---|
| Top-1 Accuracy | Standard flat accuracy |
| Mistake-Only Rel. LCA Depth (MRD) | Mean `depth(LCA) / avg_leaf_depth` over incorrect predictions |
| **Master Metric (Acc × MRD)** | **Primary pipeline optimisation objective** |
| Avg LCA Depth | Mean LCA depth for mistake pairs |
| Hierarchical Dist @ K | Mean LCA subtree height across top-K predictions |
| Tree-Visual Alignment (TVA) | Spearman ρ between tree path-distances and visual feature distances |

---

## License

Released for academic, non-commercial use.
