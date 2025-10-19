# =============================================================
# Energy-Stabilized Topological Adam (α–β Coupling)
# =============================================================
import math, time
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ---------- TPU / GPU / CPU Auto-Detection ----------
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    XLA_AVAILABLE = True
    device = xm.xla_device()
    print(f"✓ Using device: {device} (TPU/XLA)")
except ImportError:
    XLA_AVAILABLE = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")

# ----------------------- Models -----------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,10)
    def forward(self,x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class CNNsmall(nn.Module):
    def __init__(self, in_ch=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,32,3,1,1)
        self.conv2 = nn.Conv2d(32,64,3,1,1)
        self.fc1 = nn.Linear(64*16*16,128)
        self.fc2 = nn.Linear(128,num_classes)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ----------------- Topological Adam (Stable α–β) -----------------
class TopologicalAdam(torch.optim.Optimizer):
    """Energy-Stabilized α–β Coupling Optimizer"""
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8,
                 eta=0.02, mu0=0.5, w_topo=0.15, field_init_scale=0.01,
                 target_energy=1e-3):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        eta=eta, mu0=mu0, w_topo=w_topo,
                        field_init_scale=field_init_scale,
                        target_energy=target_energy)
        super().__init__(params, defaults)
        self._energy = 0.0; self._J_accum = 0.0; self._J_count = 0
        self._alpha_norm = 0.0; self._beta_norm = 0.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        self._energy = 0.0; self._J_accum = 0.0; self._J_count = 0
        self._alpha_norm = 0.0; self._beta_norm = 0.0
        for group in self.param_groups:
            lr,(b1,b2),eps = group['lr'],group['betas'],group['eps']
            eta,mu0,w_topo,field_init_scale,target_energy = \
                group['eta'],group['mu0'],group['w_topo'],\
                group['field_init_scale'],group['target_energy']

            for p in group['params']:
                if p.grad is None: continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step']=0
                    state['m']=torch.zeros_like(p,device=p.device)
                    state['v']=torch.zeros_like(p,device=p.device)
                    std = field_init_scale*(2.0/p.numel())**0.5
                    state['alpha']=torch.randn_like(p,device=p.device)*std*3.0
                    state['beta']=torch.randn_like(p,device=p.device)*std*1.0
                state['step']+=1
                m,v,a,b = state['m'],state['v'],state['alpha'],state['beta']

                # Adam base update
                m.mul_(b1).add_(g,alpha=1-b1)
                v.mul_(b2).addcmul_(g,g,value=1-b2)
                m_hat = m/(1-b1**state['step'])
                v_hat = v/(1-b2**state['step'])
                adam_dir = m_hat/(v_hat.sqrt()+eps)

                g_norm = g.norm()
                if torch.isfinite(g_norm) and g_norm>1e-12:
                    g_dir = g/(g_norm+1e-12)
                    j_alpha = (a*g_dir).sum(); j_beta = (b*g_dir).sum()
                    J = j_alpha - j_beta
                    a_prev = a.clone()

                    a.mul_(1-eta).add_(b,alpha=(eta/mu0)*J)
                    b.mul_(1-eta).add_(a_prev,alpha=-(eta/mu0)*J)

                    # Energy feedback normalization
                    energy_local = 0.5*((a**2+b**2).mean()).item()
                    if energy_local < target_energy:
                        scale = math.sqrt(target_energy/(energy_local+1e-12))
                        a.mul_(scale); b.mul_(scale)
                    elif energy_local > target_energy*10:
                        a.mul_(0.9); b.mul_(0.9)

                    topo_corr = torch.tanh(a-b)
                    self._energy += energy_local
                    self._J_accum += float(abs(J)); self._J_count += 1
                    self._alpha_norm += a.norm().item(); self._beta_norm += b.norm().item()
                else:
                    topo_corr = torch.zeros_like(p)

                p.add_(adam_dir + w_topo*topo_corr, alpha=-lr)
        return loss

    def energy(self): return self._energy
    def J_mean_abs(self): return self._J_accum/max(1,self._J_count)

# ---------------- Utilities ----------------
def get_loaders(name,batch=128):
    if name=="mnist":
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1307,),(0.3081,))])
        tr = datasets.MNIST("./data",train=True,download=True,transform=tf)
        te = datasets.MNIST("./data",train=False,download=True,transform=tf)
    elif name=="fashion":
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5,),(0.5,))])
        tr = datasets.FashionMNIST("./data",train=True,download=True,transform=tf)
        te = datasets.FashionMNIST("./data",train=False,download=True,transform=tf)
    elif name=="cifar":
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        tr = datasets.CIFAR10("./data",train=True,download=True,transform=tf)
        te = datasets.CIFAR10("./data",train=False,download=True,transform=tf)
    return DataLoader(tr,batch,shuffle=True,num_workers=2,pin_memory=True), \
           DataLoader(te,512,shuffle=False,num_workers=2,pin_memory=True)

@torch.no_grad()
def evaluate(model,loader):
    model.eval(); correct=0; total=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        pred=model(x).argmax(1)
        correct+=(pred==y).sum().item(); total+=y.size(0)
    if XLA_AVAILABLE: xm.mark_step()
    return correct/total

def train_epoch(model,opt,loader):
    model.train(); total_loss=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        opt.zero_grad(); out=model(x)
        loss=F.cross_entropy(out,y)
        loss.backward()
        if XLA_AVAILABLE:
            xm.optimizer_step(opt, barrier=True)
            xm.mark_step()
        else:
            opt.step()
        total_loss+=loss.item()
    return total_loss/len(loader)

# ---------------- Benchmark Runner ----------------
def run_dataset(name,model_cls,epochs=10):
    tr,te=get_loaders(name)
    modelA,modelT=model_cls().to(device),model_cls().to(device)
    optA=torch.optim.Adam(modelA.parameters(),lr=1e-3)
    optT=TopologicalAdam(modelT.parameters(),lr=1e-3)
    log={'Adam_acc':[],'Topo_acc':[],'Topo_energy':[]}
    print(f"\n=== {name.upper()} ===")
    for e in range(epochs):
        train_epoch(modelA,optA,tr)
        train_epoch(modelT,optT,tr)
        a=evaluate(modelA,te); t=evaluate(modelT,te)
        log['Adam_acc'].append(a)
        log['Topo_acc'].append(t)
        log['Topo_energy'].append(optT.energy())
        print(f"Epoch {e:02d} | Adam={a*100:.2f}% | Topo={t*100:.2f}% | "
              f"Energy={optT.energy():.3e} | |J|={optT.J_mean_abs():.3e}")
    return log

# ---------------- Run all + Plot ----------------
start=time.time()
results={}
for ds,model in [("mnist",MLP),("fashion",MLP),("cifar",CNNsmall)]:
    results[ds]=run_dataset(ds,model,epochs=10)

print(f"\n✓ All benchmarks complete in {time.time()-start:.1f}s")

# -------- Visualization --------
for ds in results:
    r=results[ds]
    plt.figure(figsize=(8,4))
    plt.plot(r['Adam_acc'],label='Adam Acc')
    plt.plot(r['Topo_acc'],label='TopoAdam Acc')
    plt.title(f'{ds.upper()} Accuracy Comparison')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    plt.show()

    plt.figure(figsize=(6,3))
    plt.plot(r['Topo_energy'],label='Topo Energy',color='orange')
    plt.title(f'{ds.upper()} Topological Energy Stability')
    plt.xlabel('Epoch'); plt.ylabel('Energy'); plt.legend(); plt.grid(True)
    plt.show()
