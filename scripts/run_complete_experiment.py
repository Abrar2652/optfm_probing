#!/usr/bin/env python3
"""
[DEPRECATED] Early 1-WL probe that runs against data/milp_pairs.py.

This script predates the sanity-check in scripts/sanity_check_pairs.py
that revealed the "canonical 1-WL pair" in data/milp_pairs.py is
graph-isomorphic (not 1-WL equivalent non-isomorphic). The cos_sim = 1.0
values this script reports are therefore uninformative about 1-WL
expressiveness: any permutation-equivariant model maps isomorphic
inputs to identical embeddings by construction.

Use scripts/run_main_experiment.py instead. It runs the same ablation
over 6 architectures (including the full hierarchical OPTFM
cross-attention variant) on the corrected pair family in
data/milp_pairs_v2.py (C_{4k}-bipartite vs k*C_4-bipartite, verified
1-WL equivalent AND non-isomorphic for every k).

This file is retained only as a historical reference for the
experiment_report.txt that earlier pipelines produced. Running it will
print a prominent deprecation warning.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.sgformer_mip import (
    SGFormer_MIP, TransConvOnly, GNNOnly, SimpleGCN,
    create_model, load_pretrained_weights
)
from data.milp_pairs import (
    MILPPair, MILPInstance, milp_to_tensors,
    construct_canonical_pair, construct_scaled_pair,
    construct_control_pair, construct_rhs_variant_pair,
    construct_objective_variant_pair, verify_1wl_equivalence,
    get_all_test_pairs
)


# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================

def extract_embedding(model: nn.Module, milp: MILPInstance) -> torch.Tensor:
    """Extract graph embedding from MILP instance."""
    cons_x, edge_index, edge_attr, var_x = milp_to_tensors(milp)
    
    with torch.no_grad():
        embedding = model.get_graph_embedding(cons_x, edge_index, edge_attr, var_x)
    
    return embedding


def compute_similarity(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
    """Compute cosine similarity."""
    return float(F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item())


def check_exact_equality(emb_a: torch.Tensor, emb_b: torch.Tensor, tol: float = 1e-6) -> bool:
    """Check if embeddings are exactly equal."""
    return bool(torch.allclose(emb_a, emb_b, atol=tol))


# ============================================================================
# FEASIBILITY CLASSIFICATION PROBE
# ============================================================================

class LinearProbe(nn.Module):
    """Linear probe for feasibility classification."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 2)
    
    def forward(self, x):
        return self.fc(x)


class MLPProbe(nn.Module):
    """MLP probe for feasibility classification."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def generate_feasibility_dataset(model: nn.Module, n_samples: int = 200, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate dataset for feasibility classification.
    
    Uses the canonical pair structure with random RHS variations.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    embeddings = []
    labels = []
    
    base = construct_canonical_pair()
    A = base.milp_a.A
    
    for _ in range(n_samples):
        # Randomly generate RHS
        b = np.random.randint(1, 4, size=5).astype(np.float32)
        
        # Check feasibility (simplified: sum constraints)
        # For x1+x2=1, x3+x4=1, x5+x6=1: feasible if b[3]+b[4] = 3 (since each pair sums to 1)
        is_feasible = (b[3] + b[4] == 3) and all(b[:3] <= 2)
        
        # Create MILP instance
        from data.milp_pairs import create_ecole_features
        var_feat, cons_feat, edge_idx, edge_attr = create_ecole_features(
            base.milp_a.c, A, b, base.milp_a.sense,
            base.milp_a.lb, base.milp_a.ub, base.milp_a.vtype
        )
        
        milp = MILPInstance(
            name=f"sample_{_}", var_features=var_feat, cons_features=cons_feat,
            edge_index=edge_idx, edge_attr=edge_attr,
            c=base.milp_a.c, A=A, b=b, sense=base.milp_a.sense,
            lb=base.milp_a.lb, ub=base.milp_a.ub, vtype=base.milp_a.vtype,
            is_feasible=is_feasible
        )
        
        emb = extract_embedding(model, milp)
        embeddings.append(emb)
        labels.append(1 if is_feasible else 0)
    
    return torch.stack(embeddings), torch.tensor(labels)


def train_probe(probe: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
                X_val: torch.Tensor, y_val: torch.Tensor,
                epochs: int = 100, lr: float = 0.01) -> Tuple[float, float]:
    """Train probe and return train/val accuracy."""
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        probe.train()
        optimizer.zero_grad()
        logits = probe(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        
        probe.eval()
        with torch.no_grad():
            train_pred = probe(X_train).argmax(dim=1)
            train_acc = (train_pred == y_train).float().mean().item()
            
            val_pred = probe(X_val).argmax(dim=1)
            val_acc = (val_pred == y_val).float().mean().item()
            
            best_val_acc = max(best_val_acc, val_acc)
    
    return train_acc, best_val_acc


def run_feasibility_probe_experiment(model: nn.Module, n_trials: int = 5) -> Dict:
    """Run feasibility classification experiment with probes."""
    results = {'linear': [], 'mlp': []}
    
    for trial in range(n_trials):
        # Generate data
        X, y = generate_feasibility_dataset(model, n_samples=300, seed=42 + trial)
        
        # Split
        n_train = int(0.7 * len(X))
        n_val = int(0.15 * len(X))
        
        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
        X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
        
        # Train probes
        input_dim = X.shape[1]
        
        linear_probe = LinearProbe(input_dim)
        _, linear_acc = train_probe(linear_probe, X_train, y_train, X_test, y_test)
        results['linear'].append(linear_acc)
        
        mlp_probe = MLPProbe(input_dim)
        _, mlp_acc = train_probe(mlp_probe, X_train, y_train, X_test, y_test)
        results['mlp'].append(mlp_acc)
    
    return {
        'linear_mean': np.mean(results['linear']),
        'linear_std': np.std(results['linear']),
        'mlp_mean': np.mean(results['mlp']),
        'mlp_std': np.std(results['mlp']),
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_1wl_test(model: nn.Module, pairs: List[MILPPair], 
                 model_name: str, n_seeds: int = 5) -> List[Dict]:
    """Run 1-WL distinguishability test."""
    results = []
    
    for pair in pairs:
        similarities = []
        exact_matches = []
        
        for seed in range(n_seeds):
            torch.manual_seed(42 + seed)
            
            emb_a = extract_embedding(model, pair.milp_a)
            emb_b = extract_embedding(model, pair.milp_b)
            
            sim = compute_similarity(emb_a, emb_b)
            exact = check_exact_equality(emb_a, emb_b)
            
            similarities.append(sim)
            exact_matches.append(exact)
        
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        all_exact = all(exact_matches)
        
        # Can distinguish if NOT all exact matches
        can_distinguish = not all_exact
        matches_expectation = can_distinguish == pair.expected_distinguishable
        
        results.append({
            'pair': pair.description,
            'model': model_name,
            'expected_distinguishable': pair.expected_distinguishable,
            'is_1wl_equivalent': pair.is_1wl_equivalent,
            'mean_similarity': float(mean_sim),
            'std_similarity': float(std_sim),
            'embeddings_exact': all_exact,
            'can_distinguish': can_distinguish,
            'matches_expectation': matches_expectation,
        })
    
    return results


def run_ablation_study(checkpoint_path: Optional[str], pairs: List[MILPPair]) -> Dict[str, List[Dict]]:
    """Run ablation study comparing different model components."""
    ablation_results = {}
    
    model_configs = [
        ('OPTFM (pre-trained)', 'optfm', checkpoint_path),
        ('OPTFM (random)', 'random', None),
        ('TransConv only', 'transconv_only', None),
        ('GNN only', 'gnn_only', None),
        ('Simple GCN', 'gcn', None),
    ]
    
    for name, model_type, ckpt in model_configs:
        print(f"  Testing {name}...")
        
        if model_type == 'optfm' and ckpt:
            model = create_model('optfm', pretrained_path=ckpt)
        else:
            model = create_model(model_type)
        
        model.eval()
        results = run_1wl_test(model, pairs, name)
        ablation_results[name] = results
    
    return ablation_results


def compute_statistics(results: List[Dict]) -> Dict:
    """Compute summary statistics."""
    n_correct = sum(1 for r in results if r['matches_expectation'])
    n_total = len(results)
    
    # Separate by expected outcome
    should_distinguish = [r for r in results if r['expected_distinguishable']]
    should_not_distinguish = [r for r in results if not r['expected_distinguishable']]
    
    stats = {
        'total_tests': n_total,
        'correct': n_correct,
        'accuracy': n_correct / n_total if n_total > 0 else 0,
        
        'true_positives': sum(1 for r in should_distinguish if r['can_distinguish']),
        'false_negatives': sum(1 for r in should_distinguish if not r['can_distinguish']),
        'true_negatives': sum(1 for r in should_not_distinguish if not r['can_distinguish']),
        'false_positives': sum(1 for r in should_not_distinguish if r['can_distinguish']),
    }
    
    return stats


# ============================================================================
# VISUALIZATION
# ============================================================================

def generate_figures(all_results: Dict, output_dir: Path):
    """Generate publication-quality figures."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'figure.figsize': (10, 6),
            'figure.dpi': 150,
        })
        
        # Figure 1: Similarity comparison across models
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(all_results.keys())
        pair_names = [r['pair'][:30] + '...' if len(r['pair']) > 30 else r['pair'] 
                      for r in all_results[models[0]]]
        
        x = np.arange(len(pair_names))
        width = 0.15
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
        
        for i, model in enumerate(models):
            sims = [r['mean_similarity'] for r in all_results[model]]
            stds = [r['std_similarity'] for r in all_results[model]]
            ax.bar(x + i * width, sims, width, label=model, color=colors[i % len(colors)],
                   yerr=stds, capsize=2)
        
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlabel('MILP Pair')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Embedding Similarity Across Models and MILP Pairs')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(pair_names, rotation=45, ha='right')
        ax.legend(loc='lower left')
        ax.set_ylim(0.7, 1.05)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'fig1_similarity_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Accuracy comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        
        model_names = []
        accuracies = []
        
        for model, results in all_results.items():
            stats = compute_statistics(results)
            model_names.append(model)
            accuracies.append(stats['accuracy'] * 100)
        
        bars = ax.barh(model_names, accuracies, color=colors[:len(model_names)])
        ax.set_xlabel('Accuracy (%)')
        ax.set_title('1-WL Distinguishability Test Accuracy')
        ax.set_xlim(0, 105)
        
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f'{acc:.1f}%', va='center')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'fig2_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Figures saved to {output_dir}")
        
    except ImportError:
        print("matplotlib not available, skipping figures")


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(all_results: Dict, probe_results: Dict, output_dir: Path) -> str:
    """Generate comprehensive experiment report."""
    
    lines = [
        "=" * 80,
        "1-WL EXPRESSIVENESS EXPERIMENT - COMPLETE REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "=" * 80,
        "EXECUTIVE SUMMARY",
        "=" * 80,
        "",
    ]
    
    # Check canonical pair result for OPTFM pre-trained
    optfm_pretrained = all_results.get('OPTFM (pre-trained)', [])
    canonical = next((r for r in optfm_pretrained if 'Canonical' in r['pair']), None)
    
    if canonical:
        if canonical['can_distinguish']:
            lines.append("FINDING: OPTFM CAN distinguish 1-WL equivalent MILP pairs")
            lines.append("INTERPRETATION: OPTFM overcomes 1-WL expressiveness limitations!")
        else:
            lines.append("FINDING: OPTFM CANNOT distinguish 1-WL equivalent MILP pairs")
            lines.append("INTERPRETATION: OPTFM inherits 1-WL expressiveness limitations.")
        lines.append(f"Evidence: Canonical pair cosine similarity = {canonical['mean_similarity']:.6f}")
        lines.append(f"         Embeddings exactly equal: {canonical['embeddings_exact']}")
    
    lines.extend(["", "=" * 80, "DETAILED RESULTS BY MODEL", "=" * 80, ""])
    
    for model_name, results in all_results.items():
        stats = compute_statistics(results)
        lines.append(f"\n### {model_name}")
        lines.append(f"Accuracy: {stats['correct']}/{stats['total_tests']} ({stats['accuracy']*100:.1f}%)")
        lines.append(f"True Positives: {stats['true_positives']}, False Negatives: {stats['false_negatives']}")
        lines.append(f"True Negatives: {stats['true_negatives']}, False Positives: {stats['false_positives']}")
        lines.append("")
        
        for r in results:
            status = '✓' if r['matches_expectation'] else '✗'
            lines.append(f"  {status} {r['pair'][:50]}")
            lines.append(f"      Similarity: {r['mean_similarity']:.6f} ± {r['std_similarity']:.6f}")
            lines.append(f"      Exact match: {r['embeddings_exact']}, Can distinguish: {r['can_distinguish']}")
    
    lines.extend(["", "=" * 80, "FEASIBILITY CLASSIFICATION RESULTS", "=" * 80, ""])
    
    if probe_results:
        for model_name, pr in probe_results.items():
            lines.append(f"\n### {model_name}")
            lines.append(f"Linear probe accuracy: {pr['linear_mean']*100:.1f}% ± {pr['linear_std']*100:.1f}%")
            lines.append(f"MLP probe accuracy: {pr['mlp_mean']*100:.1f}% ± {pr['mlp_std']*100:.1f}%")
    
    lines.extend(["", "=" * 80, "IMPLICATIONS", "=" * 80, ""])
    
    if canonical and not canonical['can_distinguish']:
        lines.extend([
            "",
            "1. OPTFM's global transformer attention does NOT overcome 1-WL limitations",
            "2. The combination of TransConv + GNN still cannot distinguish 1-WL equivalent graphs",
            "3. This confirms the theoretical prediction from Chen et al. (2023)",
            "4. Feasibility classification from embeddings is limited by this expressiveness gap",
            "",
            "Potential improvements:",
            "- Add random node features (breaks 1-WL)",
            "- Use higher-order WL tests (k-WL)",
            "- Incorporate solver-derived features",
        ])
    
    report = "\n".join(lines)
    
    report_path = output_dir / 'report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    return report


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "!" * 78)
    print("! DEPRECATION WARNING")
    print("!" * 78)
    print("! This script probes an isomorphic pair and is UNINFORMATIVE about 1-WL")
    print("! expressiveness. Run scripts/run_main_experiment.py instead, which uses")
    print("! the corrected C_{4k} vs k*C_4 pair family from data/milp_pairs_v2.py.")
    print("! See scripts/sanity_check_pairs.py for the diagnosis.")
    print("!" * 78 + "\n")

    parser = argparse.ArgumentParser(description="[DEPRECATED] Early 1-WL Expressiveness Experiment")
    parser.add_argument('--checkpoint', type=str, 
                       default='../OPTFM/node_pretrain/Models_SCIP/model_params_epoch_31.pth',
                       help='Path to OPTFM checkpoint')
    parser.add_argument('--output_dir', type=str, default='results/complete',
                       help='Output directory')
    parser.add_argument('--n_seeds', type=int, default=5,
                       help='Number of random seeds for statistics')
    parser.add_argument('--skip_probes', action='store_true',
                       help='Skip feasibility probe experiments')
    parser.add_argument('--skip_figures', action='store_true',
                       help='Skip figure generation')
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("1-WL EXPRESSIVENESS EXPERIMENT FOR OPTFM")
    print("=" * 80)
    print(f"Output: {output_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Get test pairs
    print("\n[1/5] Constructing test MILP pairs...")
    pairs = get_all_test_pairs()
    print(f"Created {len(pairs)} test pairs")
    
    # Verify 1-WL equivalence
    print("\n[2/5] Verifying 1-WL equivalence...")
    for pair in pairs:
        is_equiv = verify_1wl_equivalence(pair.milp_a, pair.milp_b)
        status = '✓' if is_equiv == pair.is_1wl_equivalent else '✗'
        print(f"  {status} {pair.description[:50]}: 1-WL equiv = {is_equiv}")
    
    # Run ablation study
    print("\n[3/5] Running ablation study...")
    checkpoint_path = args.checkpoint if Path(args.checkpoint).exists() else None
    if checkpoint_path:
        print(f"  Using checkpoint: {checkpoint_path}")
    else:
        print(f"  Checkpoint not found, using random initialization")
    
    all_results = run_ablation_study(checkpoint_path, pairs)
    
    # Run feasibility probes
    probe_results = {}
    if not args.skip_probes:
        print("\n[4/5] Running feasibility classification probes...")
        
        for model_name in ['OPTFM (pre-trained)', 'OPTFM (random)']:
            if model_name in all_results:
                print(f"  Training probes for {model_name}...")
                
                if 'pre-trained' in model_name and checkpoint_path:
                    model = create_model('optfm', pretrained_path=checkpoint_path)
                else:
                    model = create_model('random')
                
                model.eval()
                probe_results[model_name] = run_feasibility_probe_experiment(model)
    else:
        print("\n[4/5] Skipping feasibility probes...")
    
    # Generate figures
    if not args.skip_figures:
        print("\n[5/5] Generating figures...")
        generate_figures(all_results, output_dir)
    else:
        print("\n[5/5] Skipping figures...")
    
    # Save results
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'ablation': {k: v for k, v in all_results.items()},
            'probes': probe_results,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Generate report
    report = generate_report(all_results, probe_results, output_dir)
    print("\n" + report)
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE - Results saved to {output_dir}")
    print(f"{'='*80}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
