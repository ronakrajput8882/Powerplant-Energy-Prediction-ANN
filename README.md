<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,12,20&height=200&section=header&text=⚡%20PowerPlant%20Energy%20Predictor&fontSize=44&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Deep%20Learning%20Regression%20with%20ANN%20%7C%20PyTorch%20%2B%20-Sk-Learn%20%2B%20MSE%20R%C2%B2%20Evaluation&descAlignY=60&descAlign=50" width="100%"/>

<div align="center">
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualisation-11557C?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

</div>

---

## 📌 Project Overview

PowerPlant Energy Predictor is a **deep learning regression project** that forecasts the **Net Hourly Electrical Energy Output (PE)** of a Combined Cycle Power Plant from ambient environmental conditions. The model is a fully custom **Artificial Neural Network (ANN)** built from scratch in **PyTorch**, featuring a complete training pipeline with mini-batch loading, model checkpointing, and rigorous evaluation using **MSE** and **R² Score**.

- **Task:** Supervised Regression (Continuous Output Prediction)
- **Dataset:** Combined Cycle Power Plant Dataset (`powerplant_data.csv`)
- **Goal:** Accurately predict power plant energy output (MW) from temperature, vacuum, pressure, and humidity readings

---

## 📂 Dataset

| Property | Details |
|:---|:---|
| Source | `powerplant_data.csv` (UCI ML Repository — Combined Cycle Power Plant) |
| Total Samples | 9,568 rows |
| Features Used | 4 (`AT`, `V`, `AP`, `RH`) |
| Target Variable | `PE` — Net Hourly Electrical Energy Output (MW) |
| Missing Values | None |
| Preprocessing | StandardScaler (fit on train, transform on both) |

### Feature Descriptions

| Column | Full Name | Unit |
|:---|:---|:---|
| `AT` | Ambient Temperature | °C |
| `V` | Exhaust Vacuum | cm Hg |
| `AP` | Ambient Pressure | millibar |
| `RH` | Relative Humidity | % |
| `PE` | ⚡ Net Energy Output | **MW (Target)** |

---

## 🔄 Pipeline Workflow

```
Raw CSV → Null Check → Feature/Target Split → Train-Test Split → Standard Scaling
        → Tensor Conversion → DataLoader → ANN Training → Checkpointing → Evaluation
```

1️⃣ **Data Loading** — Dataset loaded via `pandas`, shape and nulls inspected

2️⃣ **Feature/Target Split** — `X = [AT, V, AP, RH]`, `y = PE`

3️⃣ **Train-Test Split** — 80% train / 20% test via `train_test_split(random_state=42)`

4️⃣ **Feature Scaling** — `StandardScaler` fitted on train set, applied to both splits

5️⃣ **Tensor Conversion** — NumPy arrays converted to `torch.float32` tensors

6️⃣ **DataLoader Setup** — `TensorDataset` wrapped in `DataLoader` (batch_size=32, shuffle=True)

7️⃣ **Model Training** — 100 epochs of forward pass → MSELoss → backprop → Adam update

8️⃣ **Model Checkpointing** — Best weights saved to `best_model.pt` on lowest validation loss

9️⃣ **Evaluation** — MSE on train & test sets + R² Score for regression quality

🔟 **Results Comparison** — Side-by-side DataFrame of predicted vs. actual PE values

---

## 🤖 Model

### Artificial Neural Network (ANN)

```python
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 6),   # Input → Hidden Layer 1
            nn.ReLU(),
            nn.Linear(6, 6),   # Hidden Layer 1 → Hidden Layer 2
            nn.ReLU(),
            nn.Linear(6, 1),   # Hidden Layer 2 → Output
        )

    def forward(self, x):
        return self.model(x)
```

| Component | Choice | Reason |
|:---|:---|:---|
| Architecture | 4 → 6 → 6 → 1 | Compact yet expressive for tabular regression |
| Activation | ReLU | Avoids vanishing gradients; fast convergence |
| Loss Function | `nn.MSELoss` | Standard for continuous regression targets |
| Optimizer | `Adam` | Adaptive learning rate; robust default choice |
| Epochs | 100 | Sufficient convergence with early stopping via checkpointing |
| Batch Size | 32 | Balances gradient stability and training speed |

---

## 📊 Results

| Metric | Set | Value |
|:---|:---|:---:|
| Loss Function | Both | MSELoss |
| R² Score | Test Set | **~0.96+** |
| Best Model | Saved As | `best_model.pt` |
| PCA-Ready | Input | ✅ Scaled 4D features |
| Algorithms Supported | — | ANN (PyTorch) |

---

## 🔍 Key Insights

- 🌡️ **Ambient Temperature (AT)** is the strongest predictor — higher temperatures reduce turbine efficiency and output
- 💨 **Exhaust Vacuum (V)** shows a clear negative correlation with PE — higher vacuum = lower power output
- 📉 **Loss curves** converge smoothly with no divergence between train and validation, confirming stable learning
- ✅ **Validation checkpointing** prevents saving overfitted weights — only the best generalising model is kept
- 🔁 **Mini-batch training** via `DataLoader` ensures stable gradient updates even on compact tabular data
- 🎯 **R² > 0.96** demonstrates the ANN successfully captures non-linear relationships in ambient power generation data

---

## 🗂️ Repository Structure

```
powerplant-energy-prediction-ann/
│
├── ANN_Regression.ipynb    # Full notebook: EDA → Training → Evaluation
├── powerplant_data.csv     # Raw dataset (9,568 samples)
├── best_model.pt           # Saved best model weights (auto-generated on run)
└── README.md               # Project documentation
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/ronakrajput8882/powerplant-energy-prediction-ann.git
cd powerplant-energy-prediction-ann

# Install dependencies
pip install torch scikit-learn pandas numpy matplotlib

# Launch the notebook
jupyter notebook ANN_Regression.ipynb
```

> **Note:** `best_model.pt` is auto-generated when the training cell executes for the first time.

---

## 🧠 Key Learnings

- Building a full PyTorch training loop from scratch reinforces the mechanics hidden inside high-level APIs
- `StandardScaler` must be **fit only on training data** — fitting on the full set causes data leakage
- `torch.no_grad()` during validation significantly reduces memory usage and speeds up evaluation
- Mini-batch training via `DataLoader` is critical for stable gradient updates even on smaller tabular datasets
- Saving the **best checkpoint** rather than the last epoch's weights is essential for real-world deployment readiness
- R² Score alongside MSE gives a more interpretable picture of model quality for non-technical stakeholders

---

## 🛠️ Tech Stack

| Tool | Use |
|:---|:---|
| Python 3.10+ | Core language |
| PyTorch | ANN definition, training loop, checkpointing |
| scikit-learn | Train/test split, StandardScaler, R² score |
| Pandas | Data loading, manipulation, results DataFrame |
| NumPy | Array operations and tensor preparation |
| Matplotlib | Training & validation loss curve visualisation |

---

<div align="center">

### Connect with me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ronakrajput8882)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/techwithronak)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ronakrajput8882)

*If you found this useful, please ⭐ the repo!*

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,12,20&height=100&section=footer" width="100%"/>

</div>