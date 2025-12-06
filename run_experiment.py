import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ---------- 任务生成 ----------
def generate_task():
    a = random.choice([0.5, -1.0, 2.0])
    b = random.choice([-2.0, 0.0, 1.0])
    return a, b

def sample_data(a, b, n_samples=20):
    x = torch.rand(n_samples, 1) * 2 - 1  # [-1,1]
    y = a * x + b + 0.05 * torch.randn_like(x)
    return x, y

# ---------- 模型定义 ----------
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class SparseMLP(MLP):
    def __init__(self, trainable_ratio=0.5):
        super().__init__()
        total = self.fc1.weight.numel()
        trainable = int(total * trainable_ratio)
        mask = torch.zeros_like(self.fc1.weight).flatten()
        mask[:trainable] = 1.0
        self.register_buffer("mask", mask.view_as(self.fc1.weight))
        with torch.no_grad():
            self.fc1.weight *= self.mask

    def forward(self, x):
        with torch.no_grad():
            self.fc1.weight *= self.mask
        return super().forward(x)

# ---------- 训练与评估 ----------
def train_model(model, train_x, train_y, steps=100, lr=1e-2, l2_lambda=0.0):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        model.train()
        pred = model(train_x)
        loss = F.mse_loss(pred, train_y)
        if l2_lambda > 0:
            l2_penalty = sum(p.pow(2).sum() for p in model.parameters())
            loss += l2_lambda * l2_penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if isinstance(model, SparseMLP):
            with torch.no_grad():
                model.fc1.weight *= model.mask
        losses.append(loss.item())
    return losses

def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        pred = model(x)
        loss = F.mse_loss(pred, y)
    return loss.item(), pred

# ---------- 多任务实验主函数 ----------
def run_experiment(num_tasks=8, steps=100, lr=1e-2, l2_lambda_vals=None, sparse_ratios=None, save_dir="results"):
    if l2_lambda_vals is None:
        l2_lambda_vals = [0.0, 1e-4, 1e-3]
    if sparse_ratios is None:
        sparse_ratios = [0.3, 0.5, 1.0]

    os.makedirs(save_dir, exist_ok=True)

    models_config = {
        "MLP": {"factory": MLP, "l2": 0.0, "sparse_ratio": None},
    }
    # Add L2 models
    for l2_val in l2_lambda_vals:
        if l2_val > 0:
            key = f"L2_{l2_val:.0e}"
            models_config[key] = {"factory": MLP, "l2": l2_val, "sparse_ratio": None}
    # Add Sparse models
    for ratio in sparse_ratios:
        key = f"Sparse_{int(ratio*100)}%"
        models_config[key] = {"factory": lambda: SparseMLP(trainable_ratio=ratio), "l2": 0.0, "sparse_ratio": ratio}

    results = {k: [] for k in models_config.keys()}

    for task_id in range(num_tasks):
        a, b = generate_task()
        train_x, train_y = sample_data(a, b, n_samples=20)
        test_x, test_y = sample_data(a, b, n_samples=20)
        for name, cfg in models_config.items():
            model = cfg["factory"]()
            losses = train_model(model, train_x, train_y, steps=steps, lr=lr, l2_lambda=cfg["l2"])
            test_loss, _ = evaluate(model, test_x, test_y)
            results[name].append(test_loss)
            print(f"Task {task_id+1}/{num_tasks}, Model {name}, Test MSE: {test_loss:.5f}")

    # 统计平均和标准差
    summary = {}
    for name, vals in results.items():
        vals_np = np.array(vals)
        summary[name] = {
            "mean_test_mse": vals_np.mean(),
            "std_test_mse": vals_np.std(),
            "min_test_mse": vals_np.min(),
            "max_test_mse": vals_np.max(),
        }

    # 保存统计结果到文本文件
    with open(f"{save_dir}/summary.txt", "w") as f:
        f.write("Model	Mean Test MSE	Std Test MSE	Min Test MSE	Max Test MSE\n")
        for name, stats in summary.items():
            f.write(f"{name}\t{stats['mean_test_mse']:.6f}\t{stats['std_test_mse']:.6f}\t{stats['min_test_mse']:.6f}\t{stats['max_test_mse']:.6f}\n")

    print("\nSummary of results:")
    for name, stats in summary.items():
        print(f"{name}: Mean={stats['mean_test_mse']:.6f}, Std={stats['std_test_mse']:.6f}, Min={stats['min_test_mse']:.6f}, Max={stats['max_test_mse']:.6f}")

    return results, summary

if __name__ == "__main__":
    run_experiment()
