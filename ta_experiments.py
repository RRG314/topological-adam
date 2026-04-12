"""
Topological Adam MHD Analogy Experiment
======================================
Testing whether coupling current J_t → 0 as training converges.
"""

import sys
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from topological_adam import TopologicalAdam, TopologicalAdamV2

# ============================================================
# Step 1: Monkey-patch get_field_stats method
# ============================================================
def get_field_stats_v1(self):
    """
    Compute field statistics for TopologicalAdam (V1)
    Returns: (E_t, J_t_magnitude, alpha_beta_correlation)
    """
    total_E = 0.0
    total_J = 0.0
    total_corr = 0.0
    count = 0
    
    for group in self.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            state = self.state[p]
            if 'alpha' not in state:
                continue
            
            alpha = state['alpha']
            beta = state['beta']
            g = p.grad
            
            # Energy: E = 0.5 * (||alpha||^2 + ||beta||^2)
            E = 0.5 * (alpha.pow(2).mean() + beta.pow(2).mean())
            
            # Coupling current magnitude: J = mean(|alpha - beta| * |g|)
            J = ((alpha - beta) * g).abs().mean()
            
            # Correlation between alpha and beta
            a_flat = alpha.flatten()
            b_flat = beta.flatten()
            if a_flat.std() > 1e-8 and b_flat.std() > 1e-8:
                try:
                    corr = torch.corrcoef(torch.stack([a_flat, b_flat]))[0, 1].item()
                except:
                    corr = 0.0
            else:
                corr = 0.0
            
            total_E += E.item()
            total_J += J.item()
            total_corr += corr
            count += 1
    
    if count == 0:
        return 0.0, 0.0, 0.0
    
    return total_E/count, total_J/count, total_corr/count

def get_field_stats_v2(self):
    """
    Compute field statistics for TopologicalAdamV2
    Returns: (E_t, J_t_magnitude, alpha_beta_correlation)
    """
    total_E = 0.0
    total_J = 0.0
    total_corr = 0.0
    count = 0
    
    for group in self.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            state = self.state[p]
            if 'alpha' not in state:
                continue
            
            alpha = state['alpha']
            beta = state['beta']
            g = p.grad
            
            # Energy: E = 0.5 * (||alpha||^2 + ||beta||^2)
            E = 0.5 * (alpha.pow(2).mean() + beta.pow(2).mean())
            
            # Coupling current magnitude: J = mean(|alpha - beta| * |g|)
            J = ((alpha - beta) * g).abs().mean()
            
            # Correlation between alpha and beta
            a_flat = alpha.flatten()
            b_flat = beta.flatten()
            if a_flat.std() > 1e-8 and b_flat.std() > 1e-8:
                try:
                    corr = torch.corrcoef(torch.stack([a_flat, b_flat]))[0, 1].item()
                except:
                    corr = 0.0
            else:
                corr = 0.0
            
            total_E += E.item()
            total_J += J.item()
            total_corr += corr
            count += 1
    
    if count == 0:
        return 0.0, 0.0, 0.0
    
    return total_E/count, total_J/count, total_corr/count

# Monkey-patch both versions
TopologicalAdam.get_field_stats = get_field_stats_v1
TopologicalAdamV2.get_field_stats = get_field_stats_v2

# ============================================================
# Step 2: Create synthetic MNIST-like dataset (fast)
# ============================================================
def create_simple_mnist_dataset(num_samples=1000, input_dim=28*28, num_classes=10):
    """Create synthetic MNIST-like dataset for quick training"""
    X = torch.randn(num_samples, input_dim) * 0.5
    y = torch.randint(0, num_classes, (num_samples,))
    
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)

# ============================================================
# Step 3: Define simple MLP
# ============================================================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# ============================================================
# Step 4: Training function with tracking
# ============================================================
def train_epoch(model, loader, optimizer, criterion, device, epoch, condition_name,
                track_metrics=None):
    """Train for one epoch and track metrics"""
    if track_metrics is None:
        track_metrics = {}
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    step_loss = []
    step_E = []
    step_J = []
    step_corr = []
    step_grad_norm = []
    
    for batch_idx, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        
        # Compute gradient norm
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm().item() ** 2
        grad_norm = math.sqrt(grad_norm)
        
        optimizer.step()
        
        # Get field statistics
        E_t, J_t, corr = optimizer.get_field_stats()
        
        total_loss += loss.item()
        num_batches += 1
        
        step_loss.append(loss.item())
        step_E.append(E_t)
        step_J.append(J_t)
        step_corr.append(corr)
        step_grad_norm.append(grad_norm)
    
    avg_loss = total_loss / num_batches
    avg_E = np.mean(step_E) if step_E else 0.0
    avg_J = np.mean(step_J) if step_J else 0.0
    avg_corr = np.mean(step_corr) if step_corr else 0.0
    avg_grad = np.mean(step_grad_norm) if step_grad_norm else 0.0
    
    return {
        'loss': avg_loss,
        'E_t': avg_E,
        'J_t': avg_J,
        'alpha_beta_corr': avg_corr,
        'grad_norm': avg_grad,
        'step_loss': step_loss,
        'step_E': step_E,
        'step_J': step_J,
    }

# ============================================================
# Step 5: EXPERIMENT A - Standard Adam vs Topological Adam
# ============================================================
print("=" * 80)
print("EXPERIMENT A: Standard Adam vs Topological Adam")
print("=" * 80)

device = torch.device('cpu')
criterion = nn.CrossEntropyLoss()
train_loader = create_simple_mnist_dataset(num_samples=1000, input_dim=28*28, num_classes=10)

# Condition A: Standard Adam (w_topo = 0)
print("\n[CONDITION A] Training with Standard Adam (w_topo=0)")
print("-" * 80)
model_adam = SimpleMLP(input_dim=28*28, hidden_dim=256, num_classes=10).to(device)
optimizer_adam = TopologicalAdamV2(
    model_adam.parameters(),
    lr=1e-3,
    eta=0.0,  # Disable topological coupling
    w_topo=0.0,  # Disable topological correction
    mu0=1.0,
    target_energy=1e-3
)

results_adam = []
print("Epoch | Loss      | E_t       | J_t_mag   | alpha_beta_corr | grad_norm")
print("-" * 80)
for epoch in range(10):
    metrics = train_epoch(model_adam, train_loader, optimizer_adam, criterion, device,
                         epoch, "Standard Adam", track_metrics={})
    results_adam.append(metrics)
    print(f"{epoch:5d} | {metrics['loss']:9.6f} | {metrics['E_t']:9.6f} | {metrics['J_t']:9.6f} | "
          f"{metrics['alpha_beta_corr']:15.6f} | {metrics['grad_norm']:9.6f}")

# Condition B: Topological Adam (w_topo = 0.01)
print("\n[CONDITION B] Training with Topological Adam (w_topo=0.01, eta=0.01, mu0=1.0)")
print("-" * 80)
model_topo = SimpleMLP(input_dim=28*28, hidden_dim=256, num_classes=10).to(device)
optimizer_topo = TopologicalAdamV2(
    model_topo.parameters(),
    lr=1e-3,
    eta=0.01,
    w_topo=0.01,
    mu0=1.0,
    target_energy=1.0
)

results_topo = []
print("Epoch | Loss      | E_t       | J_t_mag   | alpha_beta_corr | grad_norm")
print("-" * 80)
for epoch in range(10):
    metrics = train_epoch(model_topo, train_loader, optimizer_topo, criterion, device,
                         epoch, "Topological Adam", track_metrics={})
    results_topo.append(metrics)
    print(f"{epoch:5d} | {metrics['loss']:9.6f} | {metrics['E_t']:9.6f} | {metrics['J_t']:9.6f} | "
          f"{metrics['alpha_beta_corr']:15.6f} | {metrics['grad_norm']:9.6f}")

# ============================================================
# Step 6: Analysis - Correlation between J_t and Loss
# ============================================================
print("\n" + "=" * 80)
print("ANALYSIS A: Does J_t predict loss? (Pearson correlation)")
print("=" * 80)

# Flatten all step metrics
all_loss_adam = []
all_J_adam = []
for r in results_adam:
    all_loss_adam.extend(r['step_loss'])
    all_J_adam.extend(r['step_J'])

all_loss_topo = []
all_J_topo = []
for r in results_topo:
    all_loss_topo.extend(r['step_loss'])
    all_J_topo.extend(r['step_J'])

if len(all_loss_adam) > 1 and len(all_J_adam) > 1:
    corr_adam, pval_adam = pearsonr(all_J_adam, all_loss_adam)
    print(f"\nStandard Adam:")
    print(f"  Pearson r(J_t, loss) = {corr_adam:.6f} (p={pval_adam:.6e})")
else:
    print("\nStandard Adam: Insufficient data for correlation")

if len(all_loss_topo) > 1 and len(all_J_topo) > 1:
    corr_topo, pval_topo = pearsonr(all_J_topo, all_loss_topo)
    print(f"\nTopological Adam:")
    print(f"  Pearson r(J_t, loss) = {corr_topo:.6f} (p={pval_topo:.6e})")
else:
    print("\nTopological Adam: Insufficient data for correlation")

# ============================================================
# Step 7: Check J_t monotonicity
# ============================================================
print("\n" + "=" * 80)
print("ANALYSIS B: Does J_t decrease monotonically with loss?")
print("=" * 80)

def check_monotonicity(losses, currents):
    """Check if current decreases as loss decreases"""
    if len(losses) < 2 or len(currents) < 2:
        return None
    
    # Create epochs by taking means
    loss_by_epoch = losses
    current_by_epoch = currents
    
    # Check if trend is downward
    loss_trend = np.diff(loss_by_epoch)
    current_trend = np.diff(current_by_epoch)
    
    decreases = np.sum(current_trend < 0)
    total = len(current_trend)
    
    return decreases / total if total > 0 else 0.0

mono_adam = check_monotonicity([r['loss'] for r in results_adam],
                                [r['J_t'] for r in results_adam])
mono_topo = check_monotonicity([r['loss'] for r in results_topo],
                                [r['J_t'] for r in results_topo])

print(f"\nStandard Adam:   J_t decreases {mono_adam*100:.1f}% of the time")
print(f"Topological Adam: J_t decreases {mono_topo*100:.1f}% of the time")

# ============================================================
# Step 8: J_t at start vs end
# ============================================================
print("\n" + "=" * 80)
print("ANALYSIS C: J_t START vs END")
print("=" * 80)

print(f"\nStandard Adam:")
print(f"  J_t at epoch 0: {results_adam[0]['J_t']:.8f}")
print(f"  J_t at epoch 9: {results_adam[9]['J_t']:.8f}")
print(f"  Ratio (end/start): {results_adam[9]['J_t'] / (results_adam[0]['J_t'] + 1e-12):.4f}")

print(f"\nTopological Adam:")
print(f"  J_t at epoch 0: {results_topo[0]['J_t']:.8f}")
print(f"  J_t at epoch 9: {results_topo[9]['J_t']:.8f}")
print(f"  Ratio (end/start): {results_topo[9]['J_t'] / (results_topo[0]['J_t'] + 1e-12):.4f}")

# ============================================================
# Step 9: EXPERIMENT B - Hyperparameter Landscape (E_target)
# ============================================================
print("\n" + "=" * 80)
print("EXPERIMENT B: Hyperparameter Landscape - E_target Sensitivity")
print("=" * 80)

E_target_values = [0.1, 1.0, 10.0, 100.0]
results_landscape = {}

print("\nE_target | Final Loss | Final E_t | E_t/E_target ratio")
print("-" * 70)

for E_target in E_target_values:
    model = SimpleMLP(input_dim=28*28, hidden_dim=256, num_classes=10).to(device)
    optimizer = TopologicalAdamV2(
        model.parameters(),
        lr=1e-3,
        eta=0.01,
        w_topo=0.01,
        mu0=1.0,
        target_energy=E_target
    )
    
    metrics_list = []
    for epoch in range(3):  # 3 epochs as specified
        metrics = train_epoch(model, train_loader, optimizer, criterion, device,
                             epoch, f"E_target={E_target}", track_metrics={})
        metrics_list.append(metrics)
    
    final_loss = metrics_list[-1]['loss']
    final_E = metrics_list[-1]['E_t']
    ratio = final_E / E_target if E_target > 0 else 0.0
    
    results_landscape[E_target] = {
        'final_loss': final_loss,
        'final_E': final_E,
        'ratio': ratio
    }
    
    print(f"{E_target:8.1f} | {final_loss:10.6f} | {final_E:9.6f} | {ratio:13.4f}")

print("\nInterpretation:")
print("- If ratio ≈ 1.0, the optimizer converges to E_t ≈ E_target (MHD equilibrium analogy)")
print("- If ratio << 1.0, optimizer is conservative (undershoots energy target)")
print("- If ratio >> 1.0, optimizer is aggressive (overshoots energy target)")

# ============================================================
# Step 10: EXPERIMENT C - J_t as convergence signal
# ============================================================
print("\n" + "=" * 80)
print("EXPERIMENT C: Using J_t as Early Stopping Signal")
print("=" * 80)

# Re-collect all J_t values from Topological Adam training
print("\nJ_t trajectory over epochs:")
print("Epoch | J_t       | Loss      | J_t < 0.01? | J_t < 0.001?")
print("-" * 70)

j_threshold_low = 0.01
j_threshold_high = 0.001
early_stop_low = None
early_stop_high = None

for epoch, r in enumerate(results_topo):
    j_val = r['J_t']
    loss_val = r['loss']
    
    if j_val < j_threshold_low and early_stop_low is None:
        early_stop_low = epoch
    if j_val < j_threshold_high and early_stop_high is None:
        early_stop_high = epoch
    
    check_low = "Yes" if j_val < j_threshold_low else "No"
    check_high = "Yes" if j_val < j_threshold_high else "No"
    
    print(f"{epoch:5d} | {j_val:9.6f} | {loss_val:9.6f} | {check_low:11s} | {check_high:12s}")

print("\nEarly Stopping Analysis:")
if early_stop_low is not None:
    print(f"  Would stop at epoch {early_stop_low} if J_t < {j_threshold_low}")
else:
    print(f"  J_t never fell below {j_threshold_low}")

if early_stop_high is not None:
    print(f"  Would stop at epoch {early_stop_high} if J_t < {j_threshold_high}")
else:
    print(f"  J_t never fell below {j_threshold_high}")

# ============================================================
# Final Summary
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY OF FINDINGS")
print("=" * 80)

print("""
1. MHD RECONNECTION TEST (J_t → 0):
   - Topological Adam shows J_t behavior (coupling current magnitude)
   - If J_t decreases as loss decreases, this validates the MHD analogy
   - High correlation between J_t and loss suggests J_t predicts convergence

2. HYPERPARAMETER LANDSCAPE:
   - E_target controls the "magnetic energy" equilibrium level
   - Ratio E_t/E_target ≈ 1.0 indicates the optimizer reaches energy equilibrium
   - This mirrors magnetic reconnection equilibrium in plasma physics

3. CONVERGENCE SIGNAL:
   - J_t can be used as an alternative convergence criterion
   - When J_t → 0, the optimizer has reached "magnetic reconnection" (equilibrium)
   - This provides a physics-inspired stopping criterion
""")

print("\nKey Numerical Results:")
print(f"  Standard Adam final J_t: {results_adam[-1]['J_t']:.8f}")
print(f"  Topological Adam final J_t: {results_topo[-1]['J_t']:.8f}")
print(f"  Standard Adam final loss: {results_adam[-1]['loss']:.8f}")
print(f"  Topological Adam final loss: {results_topo[-1]['loss']:.8f}")

if 'corr_topo' in locals():
    print(f"  Topological Adam: r(J_t, loss) = {corr_topo:.6f}")
    if abs(corr_topo) > 0.5:
        print(f"    -> STRONG correlation: J_t is a good convergence signal")
    elif abs(corr_topo) > 0.3:
        print(f"    -> MODERATE correlation: J_t has some predictive value")
    else:
        print(f"    -> WEAK correlation: J_t and loss are loosely related")

print("\n" + "=" * 80)
print("Experiment complete!")
print("=" * 80)
