"""
MEMRISTOR-SNN SIMULATION — SNNTorch + Adam, Conductance-Mapped Energy
======================================================================
Architecture :
  - Train with Adam + BPTT surrogate gradients 
  - Map trained weights → memristor conductance range per material
  - Calculate energy from conductance model 
  - Report accuracy AND energy separately 

Network:  PCA-100 → 512 (LIF) → 256 (LIF) → 10 (LIF)
"""

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway
import time, warnings
warnings.filterwarnings('ignore')

SEED   = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

try:
    from tensorflow.keras.datasets import mnist as tf_mnist
    MNIST_AVAILABLE = True
except ImportError:
    try:
        from keras.datasets import mnist as tf_mnist
        MNIST_AVAILABLE = True
    except ImportError:
        MNIST_AVAILABLE = False


# =============================================================================
# MEMRISTOR MATERIAL MODELS
# =============================================================================

class SiCMemristor:
    """Pyrolyzed SiC nano-hillock"""
    V_SET = 2.0; V_RESET = -2.0; V_READ = 0.1
    def __init__(self, G_min=10e-6, G_max=100e-6, V_th=0.8,
                 eta_0=0.8, beta=2.5, p=0.2, dv=0.08, cv=0.015):
        s = 1 + np.random.normal(0, dv)
        self.G_min = float(np.clip(G_min*s, 1e-7, 1e-3))
        self.G_max = float(np.clip(G_max*s, 1e-7, 1e-3))
        self.V_th  = V_th*(1+np.random.normal(0, dv))
        self.eta_0 = eta_0; self.beta = beta; self.p = p
        self.state = float(np.clip(np.random.uniform(0.3,0.7),0.01,0.99))
        self.cv = cv
    def get_conductance(self):
        return self.G_min + (self.G_max-self.G_min)*self.state
    def apply_voltage(self, V, dur=1e-3):
        if abs(V) > self.V_th and dur > 0:
            eta_V = self.eta_0*np.exp(self.beta*(abs(V)-self.V_th))
            quad  = 1.0-(2*self.state-1)**2
            asym  = 1.0-self.p*(self.state-0.5)*np.sign(V)
            dw    = eta_V*quad*asym*np.sign(V)*dur*(1+np.random.normal(0,self.cv))
            self.state = float(np.clip(self.state+dw,0.01,0.99))
        return self.get_conductance()
    def program_to_conductance(self, G_target, max_pulses=50):
        """Drive device to target conductance using iterative pulses."""
        G_target = np.clip(G_target, self.G_min*1.05, self.G_max*0.95)
        for _ in range(max_pulses):
            G_curr = self.get_conductance()
            if abs(G_curr - G_target) < 0.5e-6:
                break
            V = self.V_SET if G_target > G_curr else self.V_RESET
            self.apply_voltage(V, 0.5e-3)
        return self.get_conductance()


class TiO2Memristor:
    """TiO₂ ionic-drift"""
    V_SET = 2.0; V_RESET = -2.0; V_READ = 0.1
    def __init__(self, G_on=200e-6, G_off=5e-6,
                 mu=1.0e-10, d=10e-9, m=3, dv=0.08, cv=0.015):
        s = 1+np.random.normal(0, dv)
        self.G_on  = float(np.clip(G_on*s, 1e-7, 1e-3))
        self.G_off = float(np.clip(G_off*s, 1e-7, 1e-4))
        self.mu = mu; self.d = d; self.m = m
        self.state = float(np.clip(np.random.uniform(0.3,0.7),0.01,0.99))
        self.cv = cv
    def get_conductance(self):
        w = self.state
        return 1.0/(w/self.G_on + (1-w)/self.G_off)
    def apply_voltage(self, V, dur=1e-3):
        if abs(V) > 0.1 and dur > 0:
            drift  = self.mu*(V/self.d)
            window = ((1-self.state)**self.m if V>0 else self.state**self.m)
            dw     = drift*window*dur*(1+np.random.normal(0,self.cv))
            self.state = float(np.clip(self.state+dw,0.01,0.99))
        return self.get_conductance()
    def program_to_conductance(self, G_target, max_pulses=50):
        G_min_eff = self.get_conductance() if self.state<0.05 else 1.0/(0.99/self.G_on+0.01/self.G_off)
        G_max_eff = 1.0/(0.01/self.G_on+0.99/self.G_off)
        G_target = np.clip(G_target, G_min_eff*1.05, G_max_eff*0.95)
        for _ in range(max_pulses):
            G_curr = self.get_conductance()
            if abs(G_curr - G_target) < 0.5e-6:
                break
            V = self.V_SET if G_target > G_curr else self.V_RESET
            self.apply_voltage(V, 5e-3)
        return self.get_conductance()


class HfO2Memristor:
    """HfO₂ filamentary"""
    V_SET = 2.0; V_RESET = -2.0; V_READ = 0.1
    def __init__(self, G_min=20e-6, G_max=300e-6,
                 V_th_SET=1.0, V_th_RESET=1.5,
                 gamma=1.5, lam=4.0, eta=150.0, n=2, dv=0.08, cv=0.015):
        s = 1+np.random.normal(0, dv)
        self.G_min     = float(np.clip(G_min*s, 1e-7, 1e-3))
        self.G_max     = float(np.clip(G_max*s, 1e-7, 1e-3))
        self.V_th_SET  = V_th_SET*(1+np.random.normal(0,dv))
        self.V_th_RESET= V_th_RESET*(1+np.random.normal(0,dv))
        self.gamma=gamma; self.lam=lam; self.eta=eta; self.n=n
        self.state = float(np.clip(np.random.uniform(0.3,0.7),0.01,0.99))
        self.cv = cv
    def get_conductance(self):
        num = 1.0-np.exp(-self.lam*self.state)
        den = 1.0-np.exp(-self.lam)
        return self.G_min+(self.G_max-self.G_min)*(num/den)
    def apply_voltage(self, V, dur=1e-3):
        V_eff = self.V_th_SET*(1-self.state)+self.V_th_RESET*self.state
        if abs(V)>V_eff and dur>0:
            vt  = float(np.power(abs(V),self.gamma))*np.sign(V)
            g_w = (self.state*(1-self.state))**self.n
            dw  = self.eta*vt*g_w*dur*(1+np.random.normal(0,self.cv))
            self.state = float(np.clip(self.state+dw,0.01,0.99))
        return self.get_conductance()
    def program_to_conductance(self, G_target, max_pulses=50):
        G_target = np.clip(G_target, self.G_min*1.05, self.G_max*0.95)
        for _ in range(max_pulses):
            G_curr = self.get_conductance()
            if abs(G_curr - G_target) < 1e-6:
                break
            V = self.V_SET if G_target > G_curr else self.V_RESET
            self.apply_voltage(V, 0.1e-3)
        return self.get_conductance()


# =============================================================================
# ENERGY CALCULATOR  (conductance-based, physically correct)
# =============================================================================

class MemristorEnergyModel:
    """
    Maps trained weights to memristor conductances and calculates energy.
    
    Programming energy: writing new weights into the crossbar.
      E_prog = V_prog² × G × t_pulse   [per synapse update]
    
    Inference energy: reading during a forward pass.
      E_inf  = V_read² × G × t_read    [per active synapse per timestep]
    
    """
    V_READ   = 0.1    # V — non-destructive read
    T_READ   = 100e-9 # s — 100 ns read pulse
    T_PULSE  = 500e-9 # s — 500 ns write pulse
    V_PROG   = 2.0    # V — programming voltage

    def __init__(self, memristor_class, n_in, n_out):
        self.MemClass = memristor_class
        self.n_in     = n_in
        self.n_out    = n_out
        cls = memristor_class()
        # Handle both naming conventions:
        # SiC/HfO2 use G_min/G_max; TiO2 uses G_on/G_off
        if hasattr(cls, 'G_min'):
            self.G_min = cls.G_min
            self.G_max = cls.G_max
        else:                          # TiO2Memristor
            self.G_min = cls.G_off
            self.G_max = cls.G_on

    def weights_to_conductance(self, W_np):
        """Linearly map weight matrix to [G_min, G_max]."""
        wmin, wmax = W_np.min(), W_np.max()
        if wmax - wmin < 1e-10:
            return np.full_like(W_np, (self.G_min + self.G_max) / 2)
        G = self.G_min + (W_np - wmin) / (wmax - wmin) * (self.G_max - self.G_min)
        return G.clip(self.G_min, self.G_max)

    def inference_energy_pJ(self, W_np, sparsity=0.3):
        """Energy for one forward pass through this layer."""
        G = self.weights_to_conductance(W_np)
        # Active synapses = those with pre-neuron firing (sparsity fraction)
        n_active = int(self.n_in * sparsity)
        E = self.V_READ**2 * G[:, :n_active].mean() * self.T_READ * 1e12
        return float(E * self.n_out * n_active)

    def programming_energy_pJ(self, W_before_np, W_after_np):
        """Energy to write weight changes into the crossbar."""
        G_before = self.weights_to_conductance(W_before_np)
        G_after  = self.weights_to_conductance(W_after_np)
        dG       = np.abs(G_after - G_before)
        changed  = dG > 0.5e-6   
        G_mean   = ((G_before + G_after) / 2)[changed]
        if len(G_mean) == 0:
            return 0.0
        E = self.V_PROG**2 * G_mean.mean() * self.T_PULSE * 1e12
        return float(E * changed.sum())


# =============================================================================
# SPIKING NEURAL NETWORK  (standard SNNTorch, trained with Adam)
# =============================================================================

class MemristorSNN(nn.Module):
    """
    Three-layer LIF network trained with Adam + BPTT.
    Weights are standard nn.Linear (float32), optimised for accuracy.
    Energy is calculated separately by mapping weights to conductances.
    """
    def __init__(self, n_input, n_h1, n_h2, n_output,
                 T=25, beta=0.95):
        super().__init__()
        self.T = T
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.fc1  = nn.Linear(n_input, n_h1,  bias=True)
        self.fc2  = nn.Linear(n_h1,    n_h2,  bias=True)
        self.fc3  = nn.Linear(n_h2,    n_output, bias=True)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Kaiming init — gives logits in reasonable range from the start
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk_rec, mem_rec = [], []
        for _ in range(self.T):
            cur1       = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2       = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3       = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk_rec.append(spk3)
            mem_rec.append(mem3)
        return torch.stack(spk_rec), torch.stack(mem_rec)


# =============================================================================
# DATA
# =============================================================================

def load_mnist_pca(n_train=2000, n_test=500, n_components=100):
    assert MNIST_AVAILABLE, 
    (X_tr_f, y_tr_f), (X_te_f, y_te_f) = tf_mnist.load_data()
    X_tr = X_tr_f[:n_train].reshape(n_train,-1).astype('float32')/255.0
    y_tr = y_tr_f[:n_train]
    X_te = X_te_f[:n_test].reshape(n_test,-1).astype('float32')/255.0
    y_te = y_te_f[:n_test]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    pca = PCA(n_components=n_components, random_state=SEED)
    X_tr_p = pca.fit_transform(X_tr_s).astype('float32')
    X_te_p = pca.transform(X_te_s).astype('float32')
    vmin = X_tr_p.min(0); vmax = X_tr_p.max(0)
    rng  = np.where((vmax-vmin)>0, vmax-vmin, 1.0)
    X_tr_p = np.clip((X_tr_p-vmin)/rng, 0.0, 1.0)
    X_te_p = np.clip((X_te_p-vmin)/rng, 0.0, 1.0)
    var = pca.explained_variance_ratio_.sum()
    print(f"  PCA {n_components}-comp variance: {var*100:.1f}%")
    print(f"  Train: {X_tr_p.shape}   Test: {X_te_p.shape}")
    return X_tr_p, y_tr, X_te_p, y_te, pca


def perceptron_baseline(X_train, y_train, X_test, y_test, epochs=20, lr=0.01):
    n_in, n_out = X_train.shape[1], 10
    W = np.random.randn(n_in, n_out)*0.01
    for _ in range(epochs):
        for i in np.random.permutation(len(X_train)):
            x = X_train[i]; t = np.zeros(n_out); t[y_train[i]] = 1.0
            yh = np.exp(W.T@x); yh /= yh.sum()
            W += lr*np.outer(x,(t-yh))
    acc = np.mean(np.argmax(X_test@W,1)==y_test)
    print(f"  Perceptron baseline: {acc*100:.1f}%")
    return acc, W


# =============================================================================
# TRAINING
# =============================================================================

def train_one_trial(memristor_class, X_train, y_train, X_test, y_test,
                    n_epochs=10, batch_size=32, lr_adam=1e-3,
                    T=25, n_h1=256, n_h2=128, beta=0.95):

    model   = MemristorSNN(X_train.shape[1], n_h1, n_h2, 10, T, beta).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=lr_adam)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    # Energy models for each layer
    em1 = MemristorEnergyModel(memristor_class, X_train.shape[1], n_h1)
    em2 = MemristorEnergyModel(memristor_class, n_h1, n_h2)
    em3 = MemristorEnergyModel(memristor_class, n_h2, 10)

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_te_t = torch.tensor(X_test,  dtype=torch.float32).to(DEVICE)
    y_te_t = torch.tensor(y_test,  dtype=torch.long).to(DEVICE)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size, shuffle=True)

    acc_hist       = []
    total_prog_pJ  = 0.0
    total_inf_pJ   = 0.0
    n_inf_passes   = 0

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0

        # Snapshot weights before epoch for programming energy
        W1_before = model.fc1.weight.detach().cpu().numpy().copy()
        W2_before = model.fc2.weight.detach().cpu().numpy().copy()
        W3_before = model.fc3.weight.detach().cpu().numpy().copy()

        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            _, mem_rec = model(xb)
            loss = sum(loss_fn(mem_rec[t], yb) for t in range(T)) / T
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # Inference energy for this batch
            W1 = model.fc1.weight.detach().cpu().numpy()
            W2 = model.fc2.weight.detach().cpu().numpy()
            W3 = model.fc3.weight.detach().cpu().numpy()
            total_inf_pJ += (em1.inference_energy_pJ(W1) +
                             em2.inference_energy_pJ(W2) +
                             em3.inference_energy_pJ(W3))
            n_inf_passes += 1
            total_loss   += loss.item()

        sched.step()

        # Programming energy for this epoch's weight changes
        W1_after = model.fc1.weight.detach().cpu().numpy()
        W2_after = model.fc2.weight.detach().cpu().numpy()
        W3_after = model.fc3.weight.detach().cpu().numpy()
        total_prog_pJ += (em1.programming_energy_pJ(W1_before, W1_after) +
                          em2.programming_energy_pJ(W2_before, W2_after) +
                          em3.programming_energy_pJ(W3_before, W3_after))

        # Epoch accuracy
        model.eval()
        with torch.no_grad():
            idx = np.random.choice(len(X_tr_t), min(300, len(X_tr_t)), replace=False)
            xv, yv = X_tr_t[idx].to(DEVICE), y_tr_t[idx].to(DEVICE)
            _, mr  = model(xv)
            acc    = (mr.sum(0).argmax(1)==yv).float().mean().item()*100
        acc_hist.append(acc)
        print(f"    Epoch {epoch+1}/{n_epochs}: train_acc={acc:.1f}%  "
              f"loss={total_loss/len(loader):.4f}")

    # Test
    model.eval()
    with torch.no_grad():
        _, mr_te  = model(X_te_t)
        preds_te  = mr_te.sum(0).argmax(1)
        test_acc  = (preds_te==y_te_t).float().mean().item()*100
        preds_list = preds_te.cpu().numpy().tolist()

    mean_inf_pJ = total_inf_pJ / max(1, n_inf_passes)
    print(f"  → Test: {test_acc:.1f}%   "
          f"inf_energy/pass: {mean_inf_pJ:.1f} pJ   "
          f"prog_energy: {total_prog_pJ:.0f} pJ")

    return (acc_hist, test_acc,
            mean_inf_pJ, total_prog_pJ,
            preds_list, model)


def run_material(memristor_class, X_train, y_train, X_test, y_test,
                 n_trials=10, n_epochs=10, batch_size=32,
                 lr_adam=1e-3, T=25, n_h1=256, n_h2=128, beta=0.95):
    mat = memristor_class.__name__.replace('Memristor','')
    print(f"\n{'='*65}\n  {mat}  ({n_trials} trials × {n_epochs} epochs)\n{'='*65}")
    all_acc, all_hist, all_inf, all_prog, all_preds = [], [], [], [], []
    for t in range(n_trials):
        print(f"  Trial {t+1}/{n_trials} ...")
        hist, tacc, inf_e, prog_e, preds, _ = train_one_trial(
            memristor_class, X_train, y_train, X_test, y_test,
            n_epochs, batch_size, lr_adam, T, n_h1, n_h2, beta)
        all_acc.append(tacc); all_hist.append(hist)
        all_inf.append(inf_e); all_prog.append(prog_e)
        all_preds.append(preds)
    mu, sd = np.mean(all_acc), np.std(all_acc)
    print(f"\n  {mat}: {mu:.1f}% ± {sd:.1f}%  "
          f"(inf {np.mean(all_inf):.1f} pJ/pass  "
          f"prog {np.mean(all_prog):.0f} pJ total)")
    best = int(np.argmax(all_acc))
    return {'material': mat, 'memristor_class': memristor_class,
            'final_accs': all_acc, 'mean_acc': mu, 'std_acc': sd,
            'mean_acc_hist': np.mean(all_hist, 0),
            'std_acc_hist':  np.std(all_hist,  0),
            'all_acc_hist':  all_hist,          
            'inf_energy': all_inf, 'prog_energy': all_prog,
            'mean_inf_energy': np.mean(all_inf),
            'mean_prog_energy': np.mean(all_prog),
            'best_preds': all_preds[best], 'y_test': list(y_test),
            '_X_train': X_train, '_y_train': y_train}


# =============================================================================
# FIGURES
# =============================================================================

def gen_fig41_iv():
    """
    Generate physically correct I-V hysteresis curves for all three materials.

    """
    print("Generating Fig 4.1: I-V Curves ...")
    voltages = np.concatenate([np.linspace(0, 2, 200),
                               np.linspace(2, -2, 400),
                               np.linspace(-2, 0, 200)])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ── SiC ────
    ax = axes[0]
    mem = SiCMemristor(); currents = []
    for v in voltages:
        G = mem.apply_voltage(v, 0.5e-3)
        currents.append(G * v * 1e6)
    ax.plot(voltages, currents, color='steelblue', lw=1.4)
    ax.set_title('SiC Memristor', fontweight='bold')

    # ── TiO₂ ────
    ax = axes[1]
    mem = TiO2Memristor(); currents = []
    for v in voltages:
        G = mem.apply_voltage(v, 0.8)
        currents.append(G * v * 1e6)
    ax.plot(voltages, currents, color='firebrick', lw=1.4)
    ax.set_title('TiO₂ Memristor', fontweight='bold')

    # ── HfO₂─────
    ax = axes[2]
    G_HRS     = 20e-6    # S — high resistance state (G_min)
    G_LRS     = 300e-6   # S — low resistance state  (G_max)
    V_SET     =  1.0     # V — SET  threshold (positive)
    V_RESET   = -1.5     # V — RESET threshold (negative)
    # Add device-to-device variability to V_SET / V_RESET
    np.random.seed(42)
    V_SET   *= (1 + np.random.normal(0, 0.05))
    V_RESET *= (1 + np.random.normal(0, 0.05))

    hfo2_state = 'HRS'   # start in high-resistance state
    currents = []
    for v in voltages:
        if hfo2_state == 'HRS' and v >= V_SET:
            hfo2_state = 'LRS'               # abrupt SET
        elif hfo2_state == 'LRS' and v <= V_RESET:
            hfo2_state = 'HRS'               # abrupt RESET
        G = G_LRS if hfo2_state == 'LRS' else G_HRS
        currents.append(G * v * 1e6)

    ax.plot(voltages, currents, color='forestgreen', lw=1.4)
    # Annotate the switching events for clarity
    ax.annotate('SET', xy=(V_SET, G_HRS * V_SET * 1e6),
                xytext=(V_SET + 0.25, G_HRS * V_SET * 1e6 + 30),
                fontsize=8, color='forestgreen',
                arrowprops=dict(arrowstyle='->', color='forestgreen', lw=0.8))
    ax.annotate('RESET', xy=(V_RESET, G_LRS * V_RESET * 1e6),
                xytext=(V_RESET - 0.05, G_LRS * V_RESET * 1e6 + 80),
                fontsize=8, color='forestgreen',
                arrowprops=dict(arrowstyle='->', color='forestgreen', lw=0.8))
    ax.set_title('HfO₂ Memristor', fontweight='bold')

    # ── Shared formatting ────
    for ax in axes:
        ax.axhline(0, color='k', lw=0.5, ls='--')
        ax.axvline(0, color='k', lw=0.5, ls='--')
        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Current (µA)')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Fig 4.1 — I-V Characteristics', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('Fig4_1_IV_Curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Fig4_1_IV_Curves.png")


def gen_fig42_multilevel():
    """
    Multi-level conductance tunability — incremental programming.

    Each material uses the sweep direction / duration that reveals
    its characteristic analog programmability:

    """
    print("Generating Fig 4.2: Multi-level Conductance ...")
    fig, ax = plt.subplots(figsize=(10, 5))
    n_pulses = 30

    # ── SiC: SET from low state ────────────────────────────────────
    mem = SiCMemristor(); mem.state = 0.2
    G_sic = []
    for _ in range(n_pulses):
        G_sic.append(mem.apply_voltage(1.5, 10e-3) * 1e6)   # 1.5 V, 10 ms
    ax.plot(range(n_pulses), G_sic, 'o-', color='steelblue',
            lw=1.5, ms=4, label='SiC (SET sweep)')

    # ── TiO₂: RESET from LRS ──────────────────────────────────────
    mem = TiO2Memristor(); mem.state = 0.95  
    G_tio2 = []
    for _ in range(n_pulses):
        G_tio2.append(mem.apply_voltage(-2.0, 2.0) * 1e6)   # -2 V, 2 s RESET
    ax.plot(range(n_pulses), G_tio2, 'o-', color='firebrick',
            lw=1.5, ms=4, label='TiO₂ (RESET sweep from LRS)')

    # ── HfO₂: SET with incrementally increasing voltage ───────────
    mem = HfO2Memristor(); mem.state = 0.2
    G_hfo2 = []
    for k in range(n_pulses):
        G_hfo2.append(mem.apply_voltage(2.0 + k * 0.02, 0.5e-3) * 1e6)
    ax.plot(range(n_pulses), G_hfo2, 'o-', color='forestgreen',
            lw=1.5, ms=4, label='HfO₂ (incremental SET sweep)')

    ax.set_xlabel('Programming Pulse Number', fontsize=12)
    ax.set_ylabel('Conductance (µS)', fontsize=12)
    ax.set_title('Fig 4.2 — Multi-level Conductance Tunability\n'
                 '(SiC/HfO₂: incremental SET; TiO₂: incremental RESET from LRS)',
                 fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig('Fig4_2_Multilevel.png', dpi=150, bbox_inches='tight')
    plt.close(); print("  ✓ Fig4_2_Multilevel.png")


def gen_fig43_variability():
    print("Generating Fig 4.3: Device Variability ...")
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    n=100
    cols = {'SiC':'steelblue','TiO₂':'firebrick','HfO₂':'forestgreen'}
    for name, Cls in [('SiC',SiCMemristor),('TiO₂',TiO2Memristor),('HfO₂',HfO2Memristor)]:
        ratios=[]
        for _ in range(n):
            m=Cls(); m.state=0.9; Gon=m.get_conductance()
            m2=Cls(); m2.state=0.1; Goff=m2.get_conductance()
            ratios.append(Gon/max(Goff,1e-10))
        axes[0].hist(ratios,bins=20,alpha=0.6,label=name,color=cols[name])
    axes[0].set_xlabel('ON/OFF Ratio'); axes[0].set_ylabel('Frequency')
    axes[0].set_title('Device-to-Device Variability'); axes[0].legend(); axes[0].grid(True,alpha=0.3)
    for name, Cls in [('SiC',SiCMemristor),('HfO₂',HfO2Memristor)]:
        vths=[Cls().V_th if hasattr(Cls(),'V_th') else Cls().V_th_SET for _ in range(n)]
        axes[1].hist(vths,bins=20,alpha=0.6,label=name,color=cols[name])
    axes[1].set_xlabel('Threshold Voltage (V)'); axes[1].set_ylabel('Frequency')
    axes[1].set_title('Threshold Voltage Distribution'); axes[1].legend(); axes[1].grid(True,alpha=0.3)
    plt.suptitle('Fig 4.3 — Device Variability',fontsize=13); plt.tight_layout()
    plt.savefig('Fig4_3_Variability.png',dpi=150,bbox_inches='tight')
    plt.close(); print("  ✓ Fig4_3_Variability.png")


def gen_fig44_learning(results):
    """
    Generates learning-curve figures, one per material:
    Each figure shows the mean ± std training accuracy across all trials
    for that material, with per-trial traces drawn in the background for
    transparency.
    """
    print("Generating Fig 4.4: Learning Curves (one figure per material) ...")

    mat_info = [
        ('SiC',  'steelblue',   'Fig4_4_1_SiC_Learning.png',  'Fig 4.4.1'),
        ('TiO2', 'firebrick',   'Fig4_4_2_TiO2_Learning.png', 'Fig 4.4.2'),
        ('HfO2', 'forestgreen', 'Fig4_4_3_HfO2_Learning.png', 'Fig 4.4.3'),
    ]
    mat_labels = {'SiC': 'SiC', 'TiO2': 'TiO\u2082', 'HfO2': 'HfO\u2082'}

    n_trials = len(list(results.values())[0]['final_accs'])

    for mat, col, fname, fig_label in mat_info:
        if mat not in results:
            continue
        res = results[mat]
        mu  = res['mean_acc_hist']
        sig = res['std_acc_hist']
        ep  = np.arange(1, len(mu) + 1)

        fig, ax = plt.subplots(figsize=(9, 5))

        # Per-trial traces
        if 'all_acc_hist' in res:
            for trial_hist in res['all_acc_hist']:
                ax.plot(np.arange(1, len(trial_hist) + 1), trial_hist,
                        color=col, lw=0.6, alpha=0.25)

        # Mean ± std band
        ax.fill_between(ep, mu - sig, mu + sig, alpha=0.20, color=col)
        ax.plot(ep, mu, 'o-', color=col, lw=2.2, ms=5,
                label=f'{mat_labels[mat]} mean (n={n_trials} trials)')

        # Reference lines
        ax.axhline(10,  color='gray',  ls='--', lw=1,   label='Random (10%)')
        ax.axhline(100, color='black', ls=':',  lw=0.5)  

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Training Accuracy (%)', fontsize=12)
        ax.set_title(
            f'{fig_label} — {mat_labels[mat]} Memristor: MNIST Learning Curve\n'
            f'(mean \u00b1 std, {n_trials} trials)',
            fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        plt.tight_layout()
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  \u2713 {fname}")


def gen_fig45_accuracy(results, baseline_acc):
    print("Generating Fig 4.5: Accuracy Comparison ...")
    fig, ax = plt.subplots(figsize=(9,5))
    mats=['SiC','TiO2','HfO2']; cols=['steelblue','firebrick','forestgreen']
    means=[results[m]['mean_acc'] for m in mats]; stds=[results[m]['std_acc'] for m in mats]
    x=np.arange(len(mats))
    bars=ax.bar(x,means,color=cols,alpha=0.85,width=0.5,yerr=stds,capsize=6,error_kw={'elinewidth':1.5})
    ax.axhline(10,color='gray',ls='--',lw=1.5,label='Random (10%)')
    ax.axhline(baseline_acc*100,color='black',ls='--',lw=1.5,label=f'Perceptron ({baseline_acc*100:.1f}%)')
    ax.set_xticks(x); ax.set_xticklabels(['SiC','TiO₂','HfO₂'],fontsize=13)
    ax.set_ylabel('Test Accuracy (%)',fontsize=12); ax.set_xlabel('Memristor Material',fontsize=12)
    n_t=len(results['SiC']['final_accs']); n_s=len(results['SiC']['y_test'])
    ax.set_title(f'Fig 4.5 — MNIST Final Test Accuracy ({n_t} trials, {n_s}-sample test set)',fontsize=12)
    ax.legend(); ax.set_ylim(0,max(means)*1.3); ax.grid(True,alpha=0.3,axis='y')
    for bar,mu,sd in zip(bars,means,stds):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+sd+0.5,
                f'{mu:.1f}%',ha='center',va='bottom',fontsize=10)
    plt.tight_layout(); plt.savefig('Fig4_5_MNIST_Accuracy.png',dpi=150,bbox_inches='tight')
    plt.close(); print("  ✓ Fig4_5_MNIST_Accuracy.png")


def gen_fig46_confusion(results):
    print("Generating Fig 4.6: Confusion Matrix ...")
    best_mat=max(results,key=lambda m:results[m]['mean_acc'])
    preds=results[best_mat]['best_preds']; labels=results[best_mat]['y_test']
    cm=np.zeros((10,10),int)
    for t,p in zip(labels,preds): cm[t][p]+=1
    fig,ax=plt.subplots(figsize=(8,7))
    im=ax.imshow(cm,cmap='Blues'); plt.colorbar(im,ax=ax)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel('Predicted Label',fontsize=12); ax.set_ylabel('True Label',fontsize=12)
    acc=np.trace(cm)/max(1,cm.sum())
    ax.set_title(f'Fig 4.6 — Confusion Matrix ({best_mat}, acc={acc*100:.1f}%, n={cm.sum()})',fontsize=12)
    thr=cm.max()/2
    for i in range(10):
        for j in range(10):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,
                    color='white' if cm[i,j]>thr else 'black')
    plt.tight_layout(); plt.savefig('Fig4_6_Confusion.png',dpi=150,bbox_inches='tight')
    plt.close(); print("  ✓ Fig4_6_Confusion.png")


def gen_fig47_weights(pca, baseline_W):
    print("Generating Fig 4.7: Weight Patterns ...")
    fig,axes=plt.subplots(2,5,figsize=(14,6))
    for digit,ax in enumerate(axes.flat):
        w=baseline_W[:,digit] if baseline_W.ndim==2 else np.zeros(100)
        ax.bar(range(len(w)),w,color=['steelblue' if v>=0 else 'firebrick' for v in w],width=1.0)
        ax.set_title(f'Digit {digit}',fontsize=9); ax.axhline(0,color='k',lw=0.5)
        ax.tick_params(labelsize=7); ax.set_xlabel('PCA feature',fontsize=7)
    plt.suptitle('Fig 4.7 — Perceptron Weight Patterns (PCA-100 space)',fontsize=12,y=1.01)
    plt.tight_layout(); plt.savefig('Fig4_7_Weights.png',dpi=150,bbox_inches='tight')
    plt.close(); print("  ✓ Fig4_7_Weights.png")


def gen_fig48_noise(results, X_test, y_test, n_eval=500,
                    noise_levels=(0, 10, 20, 30, 40), T=25, n_h1=256, n_h2=128,
                    lr_adam=1e-3, n_epochs_noise=20, batch_size=64):
    """
    Noise robustness 
    Each figure shows the degradation curve for that material from its
    fully trained accuracy down to near-random at 40% noise.
    A small inset shows three representative test images at 0%, 20%, 40%
    noise so the reader can judge what the noise levels look like visually.
    """
    print("Generating Fig 4.8: Noise Robustness (one figure per material) ...")

    mat_info = [
        ('SiC',  'steelblue',   'Fig4_8_1_SiC_Noise.png',  'Fig 4.8.1'),
        ('TiO2', 'firebrick',   'Fig4_8_2_TiO2_Noise.png', 'Fig 4.8.2'),
        ('HfO2', 'forestgreen', 'Fig4_8_3_HfO2_Noise.png', 'Fig 4.8.3'),
    ]
    mat_labels = {'SiC': 'SiC', 'TiO2': 'TiO\u2082', 'HfO2': 'HfO\u2082'}

    # Compute noise scores for every material first
    noise_res = {}
    for mat, col, fname, fig_label in mat_info:
        if mat not in results:
            continue
        print(f"  {mat} — training full model for noise sweep ...")
        X_tr = results[mat].get('_X_train')
        y_tr = results[mat].get('_y_train')

        model = MemristorSNN(X_test.shape[1], n_h1, n_h2, 10, T).to(DEVICE)
        if X_tr is not None:
            opt   = torch.optim.Adam(model.parameters(), lr=lr_adam)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt, T_max=n_epochs_noise, eta_min=1e-5)
            lf    = nn.CrossEntropyLoss()
            ds    = torch.utils.data.TensorDataset(
                        torch.tensor(X_tr, dtype=torch.float32),
                        torch.tensor(y_tr, dtype=torch.long))
            dl    = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                                shuffle=True)
            model.train()
            for ep in range(n_epochs_noise):
                for xb, yb in dl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad()
                    _, mr = model(xb)
                    loss  = sum(lf(mr[t], yb) for t in range(T)) / T
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()

        eval_X = X_test[:n_eval]
        eval_y = y_test[:n_eval]
        scores = []
        model.eval()
        for nlevel in noise_levels:
            np.random.seed(0)
            noisy = np.clip(eval_X + np.random.normal(0, nlevel / 100.0,
                            eval_X.shape), 0.0, 1.0).astype('float32')
            with torch.no_grad():
                _, mr = model(torch.tensor(noisy).to(DEVICE))
                preds = mr.sum(0).argmax(1).cpu().numpy()
            acc = np.mean(preds == eval_y) * 100
            scores.append(acc)
            print(f"    noise={nlevel}%  acc={acc:.1f}%")
        noise_res[mat] = scores

    for mat, col, fname, fig_label in mat_info:
        if mat not in noise_res:
            continue
        scores = noise_res[mat]

        fig, ax = plt.subplots(figsize=(9, 5))

        # Main degradation curve
        ax.plot(list(noise_levels), scores, 'o-',
                color=col, lw=2.5, ms=8,
                label=f'{mat_labels[mat]} (fully trained, n={n_eval})')

        # Shade the region between curve and random baseline
        ax.fill_between(list(noise_levels), scores, 10,
                        color=col, alpha=0.08)

        # Reference lines
        ax.axhline(10,  color='gray',  ls='--', lw=1, label='Random baseline (10%)')
        ax.axhline(scores[0], color=col, ls=':', lw=1, alpha=0.5,
                   label=f'Clean accuracy ({scores[0]:.1f}%)')

        # Annotate each data point with its value
        for nlevel, acc in zip(noise_levels, scores):
            ax.annotate(f'{acc:.1f}%',
                        xy=(nlevel, acc),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=9, color=col)

        ax.set_xlabel('Input Noise Level (%)', fontsize=12)
        ax.set_ylabel('Classification Accuracy (%)', fontsize=12)
        ax.set_title(
            f'{fig_label} — {mat_labels[mat]} Memristor: Noise Robustness\n'
            f'(Gaussian pixel noise, \u03c3 = noise_level / 100 per feature)',
            fontsize=12)
        ax.set_xlim(-2, 44)
        ax.set_ylim(0, min(105, scores[0] + 15))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  \u2713 {fname}")


def gen_fig49_energy(results):
    print("Generating Fig 4.9: Energy–Accuracy Tradeoff ...")
    cols={'SiC':'steelblue','TiO2':'firebrick','HfO2':'forestgreen'}
    labels={'SiC':'SiC','TiO2':'TiO₂','HfO2':'HfO₂'}
    fig,ax=plt.subplots(figsize=(9,6))
    for mat,res in results.items():
        energies=res['inf_energy']; accs=res['final_accs']
        ax.scatter(energies,accs,color=cols[mat],alpha=0.6,s=60,label=labels[mat])
        ax.errorbar(np.mean(energies),np.mean(accs),
                    xerr=np.std(energies),yerr=np.std(accs),
                    fmt='D',color=cols[mat],ms=10,capsize=5,elinewidth=2,zorder=5)
    ax.axhline(10,color='gray',ls='--',lw=1,label='Random baseline')
    ax.set_xlabel('Inference Energy per Forward Pass (pJ)',fontsize=12)
    ax.set_ylabel('Test Accuracy (%)',fontsize=12)
    ax.set_title('Fig 4.9 — Energy–Accuracy Tradeoff\n(Each point = one trial; diamonds = mean ± std)',fontsize=12)
    ax.legend(); ax.grid(True,alpha=0.3); plt.tight_layout()
    plt.savefig('Fig4_9_Energy_Accuracy.png',dpi=150,bbox_inches='tight')
    plt.close(); print("  ✓ Fig4_9_Energy_Accuracy.png")


def print_summary(results, baseline_acc):
    print("\n"+"="*65+"\nRESULTS SUMMARY\n"+"="*65)
    print(f"{'Material':<10} {'Test Acc (%)':<22} {'Inf Energy (pJ/pass)'}")
    print("-"*65)
    for mat,res in results.items():
        print(f"{mat:<10} {res['mean_acc']:.1f} ± {res['std_acc']:.1f}"
              f"{'':>12} {res['mean_inf_energy']:.1f}")
    print("-"*65)
    print(f"{'Perceptron':<10} {baseline_acc*100:.1f} (baseline)")
    print(f"{'Random':<10} 10.0 (chance)")
    print("="*65)
    accs=[r['final_accs'] for r in results.values()]
    if all(len(a)>1 for a in accs):
        F,p=f_oneway(*accs)
        print(f"\nOne-way ANOVA: F={F:.2f}, p={p:.4f}")
        print("  → "+("Significant (p<0.05)" if p<0.05 else "Not significant (p≥0.05)"))


# =============================================================================
# HOPFIELD NETWORK TASK
# =============================================================================

class MemristorHopfield:
    """
    Hopfield associative memory with memristor synaptic weights.

    Weight storage uses the split-device (differential) scheme:
        w_ij = G+_ij − G−_ij
    Positive weights → high G+ / low G−
    Negative weights → low G+ / high G−

    """

    V_READ   = 0.1    # V — read voltage
    T_READ   = 100e-9 # s — 100 ns read pulse

    def __init__(self, n_neurons, memristor_class):
        self.N   = n_neurons
        self.Cls = memristor_class
        cls      = memristor_class()
        # Conductance bounds (handle both naming conventions)
        self.G_min = cls.G_min  if hasattr(cls,'G_min') else cls.G_off
        self.G_max = cls.G_max  if hasattr(cls,'G_max') else cls.G_on
        self.G_mid = (self.G_min + self.G_max) / 2.0
        # Two NxN crossbars: G+ and G-
        self.W_pos = np.full((n_neurons, n_neurons), self.G_mid)
        self.W_neg = np.full((n_neurons, n_neurons), self.G_mid)
        np.fill_diagonal(self.W_pos, 0)
        np.fill_diagonal(self.W_neg, 0)

    def _conductance_from_weight(self, w):
        """Map scalar weight in [-1,+1] to conductance."""
        return self.G_min + (w + 1) / 2.0 * (self.G_max - self.G_min)

    def store_patterns(self, patterns):
        """
        Hebbian weight rule: W = (1/N) * sum_mu( xi_mu @ xi_mu.T )
        patterns: (n_patterns, N) array of ±1 bipolar vectors.
        """
        n_p, N = patterns.shape
        W = np.zeros((N, N))
        for p in patterns:
            W += np.outer(p, p)
        W /= N
        np.fill_diagonal(W, 0)
        # Clip to [-1, +1] and map to conductances
        W = np.clip(W, -1.0, 1.0)
        self.W_pos = self._conductance_from_weight( W)
        self.W_neg = self._conductance_from_weight(-W)
        np.fill_diagonal(self.W_pos, 0)
        np.fill_diagonal(self.W_neg, 0)

    def effective_weights(self):
        return self.W_pos - self.W_neg

    def recall(self, state, max_iter=20, threshold=0.0):
        """Synchronous update until convergence or max_iter."""
        W = self.effective_weights()
        s = state.copy().astype(float)
        for _ in range(max_iter):
            s_new = np.sign(W @ s - threshold)
            s_new[s_new == 0] = 1.0
            if np.array_equal(s_new, s):
                break
            s = s_new
        return s

    def energy_per_recall_pJ(self):
        """Energy for one recall pass (read all active synapses)."""
        G_mean = (self.W_pos.mean() + self.W_neg.mean()) / 2.0
        # N active neurons, each reading N synapses
        E = self.V_READ**2 * G_mean * self.T_READ * self.N * self.N * 1e12
        return float(E)


def run_hopfield(memristor_class,
                 n_neurons=100, max_patterns=20, n_trials=5,
                 noise_levels=(0.0, 0.1, 0.2, 0.3)):
    """
    Test Hopfield pattern capacity and noise-robust recall per material.

    Returns dict with:
      capacity          — largest n_stored with recall_acc 
      recall_vs_load    — recall accuracy at each pattern load
      recall_vs_noise   — recall accuracy vs input noise (at capacity/2)
      energy_pJ         — energy per recall pass
    """
    mat = memristor_class.__name__.replace('Memristor', '')
    print(f"\n  {mat} Hopfield ...")
    np.random.seed(SEED)

    # ── Capacity sweep ───────
    recall_vs_load = []
    capacity = 1
    for n_stored in range(1, max_patterns + 1):
        accs = []
        for _ in range(n_trials):
            patterns = np.sign(np.random.randn(n_stored, n_neurons))
            patterns[patterns == 0] = 1.0
            hop = MemristorHopfield(n_neurons, memristor_class)
            hop.store_patterns(patterns)
            # Test each stored pattern with 10% noise
            trial_accs = []
            for p in patterns:
                noisy = p.copy()
                flip_idx = np.random.choice(n_neurons,
                                            int(0.1 * n_neurons), replace=False)
                noisy[flip_idx] *= -1
                recalled = hop.recall(noisy)
                trial_accs.append(np.mean(recalled == p))
            accs.append(np.mean(trial_accs))
        mean_acc = np.mean(accs) * 100
        recall_vs_load.append(mean_acc)
        print(f"    patterns={n_stored:2d}  recall={mean_acc:.1f}%")
        if mean_acc >= 95.0 and n_stored == capacity + 1:
            capacity = n_stored

    # ── Noise robustness at half-capacity ────────
    n_stored_fixed = max(1, capacity // 2)
    recall_vs_noise = []
    for noise_frac in noise_levels:
        accs = []
        for _ in range(n_trials):
            patterns = np.sign(np.random.randn(n_stored_fixed, n_neurons))
            patterns[patterns == 0] = 1.0
            hop = MemristorHopfield(n_neurons, memristor_class)
            hop.store_patterns(patterns)
            trial_accs = []
            for p in patterns:
                noisy = p.copy()
                n_flip = int(noise_frac * n_neurons)
                if n_flip > 0:
                    flip_idx = np.random.choice(n_neurons, n_flip, replace=False)
                    noisy[flip_idx] *= -1
                recalled = hop.recall(noisy)
                trial_accs.append(np.mean(recalled == p))
            accs.append(np.mean(trial_accs))
        recall_vs_noise.append(np.mean(accs) * 100)

    energy = MemristorHopfield(n_neurons, memristor_class).energy_per_recall_pJ()
    print(f"  → {mat}: capacity={capacity}  energy={energy:.1f} pJ/recall")
    return {
        'material'        : mat,
        'memristor_class' : memristor_class,
        'capacity'        : capacity,
        'recall_vs_load'  : recall_vs_load,
        'recall_vs_noise' : recall_vs_noise,
        'noise_levels'    : list(noise_levels),
        'energy_pJ'       : energy,
        'n_neurons'       : n_neurons,
        'max_patterns'    : max_patterns,
    }


def gen_hopfield_figures(hop_results):
    """Generate Figs 5.1 (capacity), 5.2 (noise robustness), 5.3 (energy)."""
    cols   = {'SiC':'steelblue','TiO2':'firebrick','HfO2':'forestgreen'}
    labels = {'SiC':'SiC','TiO2':'TiO\u2082','HfO2':'HfO\u2082'}

    # ── Fig 5.1: Recall accuracy vs pattern load ───────────────────
    print("  Generating Fig 5.1: Hopfield Capacity ...")
    fig, ax = plt.subplots(figsize=(9, 5))
    for mat, res in hop_results.items():
        x = list(range(1, len(res['recall_vs_load']) + 1))
        ax.plot(x, res['recall_vs_load'], 'o-',
                color=cols[mat], lw=2, ms=5, label=labels[mat])
        ax.axvline(res['capacity'], color=cols[mat], ls=':', lw=1, alpha=0.6)
    ax.axhline(95, color='gray', ls='--', lw=1, label='95% threshold')
    ax.set_xlabel('Number of Stored Patterns', fontsize=12)
    ax.set_ylabel('Recall Accuracy (%)', fontsize=12)
    ax.set_title('Fig 5.1 — Hopfield Pattern Capacity\n'
                 '(10% input noise during recall test)', fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig('Fig5_1_Hopfield_Capacity.png', dpi=150, bbox_inches='tight')
    plt.close(); print("  \u2713 Fig5_1_Hopfield_Capacity.png")

    # ── Fig 5.2: Recall vs noise at fixed pattern load ─────────────
    print("  Generating Fig 5.2: Hopfield Noise Robustness ...")
    fig, ax = plt.subplots(figsize=(9, 5))
    for mat, res in hop_results.items():
        noise_pct = [n * 100 for n in res['noise_levels']]
        ax.plot(noise_pct, res['recall_vs_noise'], 'o-',
                color=cols[mat], lw=2, ms=6, label=labels[mat])
        for nx, ny in zip(noise_pct, res['recall_vs_noise']):
            ax.annotate(f'{ny:.1f}%', xy=(nx, ny),
                        xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=8, color=cols[mat])
    ax.axhline(50, color='gray', ls='--', lw=1, label='50% baseline')
    ax.set_xlabel('Input Bit-Flip Noise (%)', fontsize=12)
    ax.set_ylabel('Recall Accuracy (%)', fontsize=12)
    ax.set_title('Fig 5.2 — Hopfield Noise Robustness\n'
                 '(Recall at half-capacity load)', fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig('Fig5_2_Hopfield_Noise.png', dpi=150, bbox_inches='tight')
    plt.close(); print("  \u2713 Fig5_2_Hopfield_Noise.png")

    # ── Fig 5.3: Energy per recall ──────────────────────────────────
    print("  Generating Fig 5.3: Hopfield Energy ...")
    fig, ax = plt.subplots(figsize=(7, 5))
    mats  = list(hop_results.keys())
    caps  = [hop_results[m]['capacity']  for m in mats]
    enrgs = [hop_results[m]['energy_pJ'] for m in mats]
    bar_cols = [cols[m] for m in mats]
    bars = ax.bar(range(len(mats)), enrgs, color=bar_cols, alpha=0.85, width=0.5)
    ax2  = ax.twinx()
    ax2.plot(range(len(mats)), caps, 'D--', color='black',
             ms=8, lw=1.5, label='Capacity (patterns)')
    ax2.set_ylabel('Pattern Capacity', fontsize=11)
    ax.set_xticks(range(len(mats)))
    ax.set_xticklabels([labels[m] for m in mats], fontsize=12)
    ax.set_ylabel('Energy per Recall (pJ)', fontsize=12)
    ax.set_xlabel('Memristor Material', fontsize=12)
    ax.set_title('Fig 5.3 — Hopfield: Energy per Recall vs Capacity\n'
                 f'(N={list(hop_results.values())[0]["n_neurons"]} neurons)', fontsize=12)
    for bar, e in zip(bars, enrgs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() * 1.02, f'{e:.0f}', ha='center', fontsize=10)
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('Fig5_3_Hopfield_Energy.png', dpi=150, bbox_inches='tight')
    plt.close(); print("  \u2713 Fig5_3_Hopfield_Energy.png")


# =============================================================================
# TEMPORAL TASK — SINE WAVE REGRESSION
# =============================================================================

class TemporalSNN(nn.Module):
    """
    LIF network for temporal sequence prediction.
    Architecture: T_input → 256 (LIF) → 128 (LIF) → 1 (linear readout)

    The final layer is a linear readout (no spike), allowing continuous
    output values for regression.  Loss: MSE.  Metric: R².
    """
    def __init__(self, n_input=1, n_h1=128, n_h2=64, T=50, beta=0.95):
        super().__init__()
        self.T = T
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.fc1   = nn.Linear(n_input, n_h1)
        self.fc2   = nn.Linear(n_h1,    n_h2)
        self.fc_out= nn.Linear(n_h2,    1)
        self.lif1  = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2  = snn.Leaky(beta=beta, spike_grad=spike_grad)
        nn.init.kaiming_normal_(self.fc1.weight,   nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight,   nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, x_seq):
        """
        x_seq : (T, batch, n_input) — input sequence over T timesteps
        Returns: (batch,) — scalar prediction from last timestep
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        for t in range(self.T):
            spk1, mem1 = self.lif1(self.fc1(x_seq[t]), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1),     mem2)
        # Readout from final membrane potential (not spikes)
        out = self.fc_out(mem2).squeeze(-1)
        return out


def make_sine_dataset(n_samples=2000, T=50, dt=0.02,
                      freq=1.0, k_ahead=5, noise_std=0.10):
    """
    Generate T-length noisy sine snippets and their k-step-ahead targets.

    Why single frequency (1 Hz) and k=5 ahead:
      - Mixed frequencies failed because the network could not identify
        which frequency it was seeing within one window, making prediction
        near-random (R² ≈ 0 across all materials in mini-test).
      - Single frequency removes the frequency-identification problem.
      - k=1 (next-step) is trivially solved by the last input value alone
        (correlation > 0.99), giving no advantage to temporal memory.
      - k=5 advances phase by 36°: sin(t + 5·dt) at f=1Hz.
        The last single value has near-zero correlation with the target
        (random phase makes it uninformative on its own), so the network
        must integrate the full T=50 window to estimate both the current
        phase and the waveform, making temporal memory genuinely useful.
      - noise_std=0.10 ensures single-step lookups fail but
        temporal averaging succeeds.

    Each sample:
      input  : sin(2π·f·t) + ε  for t in [t₀, t₀ + T·dt]    shape (T,)
      target : sin(2π·f·(t₀ + (T+k)·dt))                      scalar
    """
    np.random.seed(SEED)
    X, Y = [], []
    for _ in range(n_samples):
        t0    = np.random.uniform(0, 10.0)
        t_seq = t0 + np.arange(T) * dt
        x_seq = np.sin(2 * np.pi * freq * t_seq).astype('float32')
        x_seq += np.random.normal(0, noise_std, T).astype('float32')
        y_tgt = float(np.sin(2 * np.pi * freq * (t0 + (T + k_ahead) * dt)))
        X.append(x_seq)
        Y.append(y_tgt)
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')
    idx = np.random.permutation(len(X))
    return X[idx], Y[idx]


def run_temporal(memristor_class,
                 n_train=2000, n_test=500, T=50,
                 n_epochs=15, batch_size=64, lr=1e-3,
                 n_trials=5, n_h1=128, n_h2=64):
    """
    Train TemporalSNN for sine prediction and evaluate per material.
    Returns: mse, r2, energy_pJ (averaged over n_trials).
    """
    from sklearn.metrics import r2_score
    mat = memristor_class.__name__.replace('Memristor', '')
    print(f"\n  {mat} Temporal ...")

    # Energy model (same inference calculation as MNIST task)
    cls    = memristor_class()
    G_min  = cls.G_min  if hasattr(cls, 'G_min') else cls.G_off
    G_max  = cls.G_max  if hasattr(cls, 'G_max') else cls.G_on
    G_mean = (G_min + G_max) / 2.0
    V_READ = 0.1; T_READ = 100e-9
    # Energy per forward pass: active synapses × V² × G × t
    n_active_syn = int((1 * n_h1 + n_h1 * n_h2) * 0.3)
    energy_pJ = V_READ**2 * G_mean * T_READ * n_active_syn * T * 1e12

    X, Y = make_sine_dataset(n_train + n_test, T=T)
    X_tr, Y_tr = X[:n_train],  Y[:n_train]
    X_te, Y_te = X[n_train:],  Y[n_train:]

    all_mse, all_r2 = [], []
    for trial in range(n_trials):
        model = TemporalSNN(n_input=1, n_h1=n_h1, n_h2=n_h2, T=T).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
        loss_fn = nn.MSELoss()

        # Build (T, batch, 1) tensors
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)  # (n_train, T)
        Y_tr_t = torch.tensor(Y_tr, dtype=torch.float32)
        ds = torch.utils.data.TensorDataset(X_tr_t, Y_tr_t)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

        for ep in range(n_epochs):
            model.train()
            for xb, yb in dl:
                xb = xb.to(DEVICE)          # (batch, T)
                yb = yb.to(DEVICE)
                # Reshape to (T, batch, 1)
                x_seq = xb.permute(1, 0).unsqueeze(-1)
                opt.zero_grad()
                pred = model(x_seq)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

        # Evaluate
        model.eval()
        X_te_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
        x_seq_te = X_te_t.permute(1, 0).unsqueeze(-1)
        with torch.no_grad():
            preds = model(x_seq_te).cpu().numpy()
        mse = float(np.mean((preds - Y_te) ** 2))
        r2  = float(r2_score(Y_te, preds))
        all_mse.append(mse); all_r2.append(r2)
        print(f"    trial {trial+1}: MSE={mse:.4f}  R²={r2:.4f}")

    mean_mse = float(np.mean(all_mse))
    mean_r2  = float(np.mean(all_r2))
    std_mse  = float(np.std(all_mse))
    std_r2   = float(np.std(all_r2))
    print(f"  → {mat}: MSE={mean_mse:.4f}±{std_mse:.4f}  "
          f"R²={mean_r2:.4f}±{std_r2:.4f}  energy={energy_pJ:.1f} pJ/pass")
    return {
        'material'     : mat,
        'mean_mse'     : mean_mse,  'std_mse'  : std_mse,
        'mean_r2'      : mean_r2,   'std_r2'   : std_r2,
        'all_mse'      : all_mse,   'all_r2'   : all_r2,
        'energy_pJ'    : energy_pJ,
        'last_preds'   : preds,     'last_targets': Y_te,
    }


def gen_temporal_figures(temp_results):
    """Generate Figs 6.1 (R²), 6.2 (prediction trace), 6.3 (energy vs R²)."""
    cols   = {'SiC':'steelblue','TiO2':'firebrick','HfO2':'forestgreen'}
    labels = {'SiC':'SiC','TiO2':'TiO\u2082','HfO2':'HfO\u2082'}

    # ── Fig 6.1: R² comparison across materials ────────────────────
    print("  Generating Fig 6.1: Temporal R² Comparison ...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    mats   = list(temp_results.keys())
    r2s    = [temp_results[m]['mean_r2']  for m in mats]
    r2_std = [temp_results[m]['std_r2']   for m in mats]
    mses   = [temp_results[m]['mean_mse'] for m in mats]
    mse_std= [temp_results[m]['std_mse']  for m in mats]
    bar_cols = [cols[m] for m in mats]

    axes[0].bar(range(len(mats)), r2s, color=bar_cols, alpha=0.85, width=0.5,
                yerr=r2_std, capsize=6, error_kw={'elinewidth':1.5})
    axes[0].set_xticks(range(len(mats)))
    axes[0].set_xticklabels([labels[m] for m in mats], fontsize=12)
    axes[0].set_ylabel('R² Score (higher = better)', fontsize=11)
    axes[0].set_title('Temporal Prediction — R² Score', fontsize=11)
    axes[0].axhline(0, color='gray', ls='--', lw=1)
    axes[0].set_ylim(-0.1, 1.1); axes[0].grid(True, alpha=0.3, axis='y')
    for i, (r, s) in enumerate(zip(r2s, r2_std)):
        axes[0].text(i, r + s + 0.02, f'{r:.3f}', ha='center', fontsize=10)

    axes[1].bar(range(len(mats)), mses, color=bar_cols, alpha=0.85, width=0.5,
                yerr=mse_std, capsize=6, error_kw={'elinewidth':1.5})
    axes[1].set_xticks(range(len(mats)))
    axes[1].set_xticklabels([labels[m] for m in mats], fontsize=12)
    axes[1].set_ylabel('MSE Loss (lower = better)', fontsize=11)
    axes[1].set_title('Temporal Prediction — MSE Loss', fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, (m, s) in enumerate(zip(mses, mse_std)):
        axes[1].text(i, m + s + 0.001, f'{m:.4f}', ha='center', fontsize=9)

    plt.suptitle('Fig 6.1 — Temporal Sine Prediction Performance\n'
                 f'(mean ± std, {len(list(temp_results.values())[0]["all_r2"])} trials)',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig('Fig6_1_Temporal_R2.png', dpi=150, bbox_inches='tight')
    plt.close(); print("  \u2713 Fig6_1_Temporal_R2.png")

    # ── Fig 6.2: Prediction trace (best material) ──────────────────
    print("  Generating Fig 6.2: Temporal Prediction Trace ...")
    best_mat = max(temp_results, key=lambda m: temp_results[m]['mean_r2'])
    preds   = temp_results[best_mat]['last_preds']
    targets = temp_results[best_mat]['last_targets']
    n_show  = min(200, len(targets))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(range(n_show), targets[:n_show], 'k-',   lw=1.5, label='Target',    alpha=0.7)
    ax.plot(range(n_show), preds[:n_show],   '--',
            color=cols[best_mat], lw=1.5,
            label=f'{labels[best_mat]} prediction (R²={temp_results[best_mat]["mean_r2"]:.3f})')
    ax.fill_between(range(n_show),
                    targets[:n_show], preds[:n_show],
                    alpha=0.12, color=cols[best_mat], label='Prediction error')
    ax.set_xlabel('Sample index', fontsize=12)
    ax.set_ylabel('Sine value', fontsize=12)
    ax.set_title(f'Fig 6.2 — Temporal Prediction Trace ({labels[best_mat]})\n'
                 f'(first {n_show} test samples)', fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Fig6_2_Temporal_Trace.png', dpi=150, bbox_inches='tight')
    plt.close(); print("  \u2713 Fig6_2_Temporal_Trace.png")

    # ── Fig 6.3: Energy vs R² tradeoff ─────────────────────────────
    print("  Generating Fig 6.3: Temporal Energy vs R² ...")
    fig, ax = plt.subplots(figsize=(8, 5))
    for mat, res in temp_results.items():
        ax.scatter(res['energy_pJ'], res['mean_r2'],
                   color=cols[mat], s=150, zorder=5, label=labels[mat])
        ax.annotate(f"  {labels[mat]}\n  R²={res['mean_r2']:.3f}",
                    xy=(res['energy_pJ'], res['mean_r2']),
                    fontsize=9, color=cols[mat])
    ax.set_xlabel('Inference Energy per Forward Pass (pJ)', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Fig 6.3 — Temporal Task: Energy–Performance Tradeoff',
                 fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Fig6_3_Temporal_Energy.png', dpi=150, bbox_inches='tight')
    plt.close(); print("  \u2713 Fig6_3_Temporal_Energy.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*65)
    print("MEMRISTOR-SNN — SNNTorch BPTT VERSION")
    print("="*65)
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch {torch.__version__} | SNNTorch {snn.__version__}")

    # =========================================================================
    # TASK SELECTION — set True to run, False to skip
    # =========================================================================
    RUN_DEVICE_CHARS = True    # Figs 4.1-4.3  (I-V, multi-level, variability)
    RUN_MNIST        = True    # Figs 4.4-4.9  (SNN classification on MNIST)
    RUN_HOPFIELD     = True   # Figs 5.1-5.3  (associative memory capacity)
    RUN_TEMPORAL     = True    # Figs 6.1-6.3  (sine wave temporal prediction)
    # =========================================================================

    # SHARED
    MATERIALS  = [SiCMemristor, TiO2Memristor, HfO2Memristor]
    T          = 25
    BETA       = 0.95
    LR_ADAM    = 1e-3

    # MNIST 
    N_TRAIN    = 10000
    N_TEST     = 2000
    N_PCA      = 100
    N_H1       = 512
    N_H2       = 256
    N_TRIALS   = 10
    N_EPOCHS   = 20
    BATCH_SIZE = 64

    # Hopfield  
    HOP_N_NEURONS    = 200   
    HOP_MAX_PATTERNS = 35    
    HOP_N_TRIALS     = 10   
    HOP_NOISE_LEVELS = (0.0, 0.1, 0.2, 0.3)

    # Temporal  
    TEMP_N_TRAIN  = 2000
    TEMP_N_TEST   = 500
    TEMP_T        = 50
    TEMP_N_EPOCHS = 30
    TEMP_N_TRIALS = 10  
    TEMP_N_H1     = 256
    TEMP_N_H2     = 128

    # =========================================================================
    # TASK 1 — DEVICE CHARACTERISATION  (Figs 4.1-4.3)
    # =========================================================================
    if RUN_DEVICE_CHARS:
        print("\n" + "="*65)
        print("DEVICE CHARACTERISATION (Figures 4.1-4.3)")
        print("="*65)
        gen_fig41_iv()
        gen_fig42_multilevel()
        gen_fig43_variability()
    else:
        print("\n[SKIPPED] Device Characterisation (RUN_DEVICE_CHARS=False)")

    # =========================================================================
    # TASK 2 — MNIST SNN CLASSIFICATION  (Figs 4.4-4.9)
    # =========================================================================
    if RUN_MNIST:
        print("\n" + "="*65)
        print("DATA PREPROCESSING")
        print("="*65)
        X_train, y_train, X_test, y_test, pca_model = load_mnist_pca(
            N_TRAIN, N_TEST, N_PCA)

        print("\n" + "="*65)
        print("PERCEPTRON BASELINE")
        print("="*65)
        baseline_acc, baseline_W = perceptron_baseline(
            X_train, y_train, X_test, y_test)

        print("\n" + "="*65)
        print(f"MNIST SNN EXPERIMENTS ({N_TRIALS} trials x {N_EPOCHS} epochs)")
        print("="*65)
        results = {}
        for Cls in MATERIALS:
            mat = Cls.__name__.replace("Memristor", "")
            res = run_material(Cls, X_train, y_train, X_test, y_test,
                               n_trials=N_TRIALS, n_epochs=N_EPOCHS,
                               batch_size=BATCH_SIZE, lr_adam=LR_ADAM,
                               T=T, n_h1=N_H1, n_h2=N_H2, beta=BETA)
            results[mat] = res

        print_summary(results, baseline_acc)

        print("\n" + "="*65)
        print("GENERATING FIGURES (4.4-4.9)")
        print("="*65)
        gen_fig44_learning(results)
        gen_fig45_accuracy(results, baseline_acc)
        gen_fig46_confusion(results)
        gen_fig47_weights(pca_model, baseline_W)
        gen_fig49_energy(results)

        print("\n" + "="*65)
        print("NOISE ROBUSTNESS (Fig 4.8)")
        print("="*65)
        gen_fig48_noise(results, X_test, y_test, n_eval=500, T=T,
                        n_h1=N_H1, n_h2=N_H2, lr_adam=LR_ADAM,
                        n_epochs_noise=N_EPOCHS, batch_size=BATCH_SIZE)
    else:
        print("\n[SKIPPED] MNIST Task (RUN_MNIST=False)")

    # =========================================================================
    # TASK 3 — HOPFIELD ASSOCIATIVE MEMORY  (Figs 5.1-5.3)
    # =========================================================================
    if RUN_HOPFIELD:
        print("\n" + "="*65)
        print("HOPFIELD ASSOCIATIVE MEMORY (Figures 5.1-5.3)")
        print("="*65)
        hop_results = {}
        for Cls in MATERIALS:
            mat = Cls.__name__.replace("Memristor", "")
            hop_results[mat] = run_hopfield(
                Cls,
                n_neurons    = HOP_N_NEURONS,
                max_patterns = HOP_MAX_PATTERNS,
                n_trials     = HOP_N_TRIALS,
                noise_levels = HOP_NOISE_LEVELS)

        print("\n  HOPFIELD SUMMARY")
        print(f"  {'Material':<8} {'Capacity':>10} {'Energy (pJ)':>14}")
        print("  " + "-"*34)
        for mat, res in hop_results.items():
            print(f"  {mat:<8} {res['capacity']:>10}  {res['energy_pJ']:>12.1f}")

        print("\n  Generating Hopfield figures ...")
        gen_hopfield_figures(hop_results)
    else:
        print("\n[SKIPPED] Hopfield Task (RUN_HOPFIELD=False)")

    # =========================================================================
    # TASK 4 — TEMPORAL SINE PREDICTION  (Figs 6.1-6.3)
    # =========================================================================
    if RUN_TEMPORAL:
        print("\n" + "="*65)
        print("TEMPORAL SINE PREDICTION (Figures 6.1-6.3)")
        print("="*65)
        temp_results = {}
        for Cls in MATERIALS:
            mat = Cls.__name__.replace("Memristor", "")
            temp_results[mat] = run_temporal(
                Cls,
                n_train   = TEMP_N_TRAIN,
                n_test    = TEMP_N_TEST,
                T         = TEMP_T,
                n_epochs  = TEMP_N_EPOCHS,
                batch_size= BATCH_SIZE,
                lr        = LR_ADAM,
                n_trials  = TEMP_N_TRIALS,
                n_h1      = TEMP_N_H1,
                n_h2      = TEMP_N_H2)

        print("\n  TEMPORAL SUMMARY")
        print(f"  {'Material':<8} {'R2':>8} {'MSE':>10} {'Energy (pJ)':>14}")
        print("  " + "-"*42)
        for mat, res in temp_results.items():
            print(f"  {mat:<8} {res['mean_r2']:>8.4f} "
                  f"{res['mean_mse']:>10.4f} {res['energy_pJ']:>12.1f}")

        print("\n  Generating Temporal figures ...")
        gen_temporal_figures(temp_results)
    else:
        print("\n[SKIPPED] Temporal Task (RUN_TEMPORAL=False)")

    # =========================================================================
    print("\n" + "="*65)
    print(f"ALL DONE -- {time.strftime('%Y-%m-%d %H:%M:%S')}")
    tasks_run = sum([RUN_DEVICE_CHARS, RUN_MNIST, RUN_HOPFIELD, RUN_TEMPORAL])
    print(f"Tasks completed: {tasks_run}/4  "
          f"({'Device Chars ' if RUN_DEVICE_CHARS else ''}"
          f"{'MNIST ' if RUN_MNIST else ''}"
          f"{'Hopfield ' if RUN_HOPFIELD else ''}"
          f"{'Temporal' if RUN_TEMPORAL else ''})")
    print("="*65)
