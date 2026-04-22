

---

## 📌 Project Overview

This project applies **deep learning regression** to forecast the **Net Hourly Electrical Energy Output (PE)** of a combined cycle power plant based on ambient environmental conditions. The model is a fully custom **Artificial Neural Network (ANN)** built from scratch using **PyTorch**, trained with the **Adam optimizer** and **MSELoss**.

---

## 📂 Dataset

**File:** `powerplant_data.csv`  
**Source:** Combined Cycle Power Plant Dataset (UCI ML Repository)  
**Samples:** 9,568 rows

| Feature | Description |
|--------|-------------|
| `AT` | Ambient Temperature (°C) |
| `V` | Exhaust Vacuum (cm Hg) |
| `AP` | Ambient Pressure (millibar) |
| `RH` | Relative Humidity (%) |
| `PE` | ⚡ Net Energy Output (MW) — **Target** |

---

## 🧠 Model Architecture

```
Input Layer     →  4 features (AT, V, AP, RH)
Hidden Layer 1  →  6 neurons + ReLU
Hidden Layer 2  →  6 neurons + ReLU
Output Layer    →  1 neuron  (Predicted PE in MW)
```

Built using `torch.nn.Sequential` for clean, modular design.

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| `PyTorch` | ANN model definition & training loop |
| `scikit-learn` | Train/test split, StandardScaler, R² score |
| `pandas` | Data loading & manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Loss curve visualization |

---

## 🚀 Workflow

```
1. Load & Explore Data       →  pandas EDA, null checks
2. Feature / Target Split    →  X = [AT, V, AP, RH], y = PE
3. Train-Test Split          →  80% train / 20% test (random_state=42)
4. Feature Scaling           →  StandardScaler (fit on train, transform both)
5. Tensor Conversion         →  torch.float32 tensors
6. DataLoader Setup          →  batch_size=32, shuffle=True
7. Model Definition          →  ANN class (nn.Module)
8. Training Loop             →  100 epochs, Adam, MSELoss
9. Model Checkpointing       →  best_model.pt saved on lowest val loss
10. Evaluation               →  MSE + R² score on test set
11. Results Comparison       →  Predicted vs Actual DataFrame
```

---

## 📈 Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (default lr) |
| Loss Function | MSELoss |
| Epochs | 100 |
| Batch Size | 32 |
| Activation | ReLU |
| Checkpointing | `best_model.pt` (lowest val loss) |

---

## 📊 Evaluation Metrics

- **Training MSE** — measures fit on training data
- **Testing MSE** — generalization performance  
- **R² Score** — proportion of variance explained (via `sklearn.metrics.r2_score`)

Loss curves are plotted for both training and validation across all 100 epochs to inspect convergence and detect overfitting.

---

## 📁 File Structure

```
📦 powerplant-energy-prediction-ann/
 ┣ 📓 ANN_Regression.ipynb      # Full notebook with all steps
 ┣ 📄 powerplant_data.csv       # Dataset
 ┣ 💾 best_model.pt             # Saved best model weights
 ┗ 📄 README.md
```

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/your-username/powerplant-energy-prediction-ann.git
cd powerplant-energy-prediction-ann

# 2. Install dependencies
pip install torch scikit-learn pandas numpy matplotlib

# 3. Launch the notebook
jupyter notebook ANN_Regression.ipynb
```

---

## 🧩 Key Concepts Demonstrated

- Building a custom `nn.Module` class in PyTorch
- Using `TensorDataset` and `DataLoader` for mini-batch training
- Manual training loop with forward pass, backprop, and optimizer step
- Validation loop using `torch.no_grad()` for efficiency
- Best model checkpointing with `torch.save` / `torch.load`
- Evaluation with both MSE and R² for regression tasks

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

<div align="center">
  <sub>Built with ❤️ using PyTorch · scikit-learn · pandas</sub>
</div>