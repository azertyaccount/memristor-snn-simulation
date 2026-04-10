# Memristor-Based SNN Simulation

This repository contains the implementation and results of a memristor-based spiking neural network (SNN) using SNNTorch.

## 📌 Features
- SNNTorch-based SNN with BPTT training
- Memristor models:
  - SiC (pyrolyzed nano-hillock)
  - TiO2 (ionic drift)
  - HfO2 (filamentary)
- Energy-aware computation:
  - Inference energy
  - Programming energy

## 🧠 Architecture
PCA-100 → 512 → 256 → 10 (LIF neurons)

## 📊 Results
- Accuracy ≈ 89% (SiC)
- Energy metrics included

## 📁 Files
- `code/Simulation_SNNTorch.py` → Main simulation code
- `results/terminal_output.txt` → Full experiment logs

## ▶️ How to run
```bash
python Simulation_SNNTorch.py
