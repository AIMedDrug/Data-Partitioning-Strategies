import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 路径设置
csv_file = '/home/data1/BGM/fenziduiqi/check_embed_list.csv'
output_dir = '/home/data1/BGM/fenziduiqi/All_X-_UNIPORT_4179_DNN_result'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 加载 CSV 文件
df = pd.read_csv(csv_file)

# 获取 uniprot_ids
uniprot_ids = df['uniprot_ids'].values

# 解析嵌入向量
def parse_emb(emb_str):
    return np.array([float(x) for x in emb_str.split(',')])

# 处理 X1, X2, ECFP 列
X1 = np.array(df['X1'].apply(parse_emb).tolist())
X2 = np.array(df['X2'].apply(parse_emb).tolist())
df_ecfp = np.array(df['ECFP'].apply(parse_emb).tolist())
y = np.array(df['ddG'])

# 定义 DNN 模型
class DNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, num_layers=1, dropout=0.2):
        super(DNNRegressor, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).view(-1)

# DNN 超参数
hidden_dim_list = [16, 128, 256, 1024]
dropout_list = [0.1, 0.2, 0.3, 0.4, 0.5]
num_layers_list = [1, 2, 3]

pearson_results = []

for hidden_dim in hidden_dim_list:
    for dropout in dropout_list:
        for num_layers in num_layers_list:
            # ====================== 基于 uniprot_ids 划分数据集 ======================
            unique_ids = np.unique(uniprot_ids)

            # 将 unique_ids 按照 80% (train + val) 和 20% test 划分
            train_ids, temp_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)

            # 将 temp_ids 再按照一半一半划分：10% val 和 10% test
            val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

            # 根据掩码获取对应行
            train_mask = np.isin(uniprot_ids, train_ids)
            val_mask = np.isin(uniprot_ids, val_ids)
            test_mask = np.isin(uniprot_ids, test_ids)

            # 分割数据
            X1_train = X1[train_mask]
            X2_train = X2[train_mask]
            ecfp_train = df_ecfp[train_mask]
            y_train = y[train_mask]

            X1_val = X1[val_mask]
            X2_val = X2[val_mask]
            ecfp_val = df_ecfp[val_mask]
            y_val = y[val_mask]

            X1_test = X1[test_mask]
            X2_test = X2[test_mask]
            ecfp_test = df_ecfp[test_mask]
            y_test = y[test_mask]

            # 合并特征：改为 (X1-X2, df_ecfp)
            X_train_diff = X1_train - X2_train  # 计算 X1 和 X2 的差值
            X_train = np.concatenate((X_train_diff, ecfp_train), axis=1)

            X_val_diff = X1_val - X2_val  # 计算 X1 和 X2 的差值
            X_val = np.concatenate((X_val_diff, ecfp_val), axis=1)

            X_test_diff = X1_test - X2_test  # 计算 X1 和 X2 的差值
            X_test = np.concatenate((X_test_diff, ecfp_test), axis=1)

            # 验证划分比例
            total_size = len(X1)
            print(f"总样本数: {total_size}")
            print(f"训练集大小: {len(X_train)} ({len(X_train)/total_size:.2%})")
            print(f"验证集大小: {len(X_val)} ({len(X_val)/total_size:.2%})")
            print(f"测试集大小: {len(X_test)} ({len(X_test)/total_size:.2%})")
            print(f"Shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

            # 转换为张量（DNN 不需要额外的序列维度）
            X_train_tensor = torch.from_numpy(X_train).float()
            y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)
            X_val_tensor = torch.from_numpy(X_val).float()
            y_val_tensor = torch.from_numpy(y_val).float().view(-1, 1)
            X_test_tensor = torch.from_numpy(X_test).float()
            y_test_tensor = torch.from_numpy(y_test).float().view(-1, 1)

            # 创建 DataLoader
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # 初始化模型
            model = DNNRegressor(
                input_dim=X_train.shape[1],
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            )
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
            
            # 使用 GPU
            device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            X_train_tensor = X_train_tensor.to(device)
            y_train_tensor = y_train_tensor.to(device)
            X_val_tensor = X_val_tensor.to(device)
            y_val_tensor = y_val_tensor.to(device)
            X_test_tensor = X_test_tensor.to(device)
            y_test_tensor = y_test_tensor.to(device)
            
            # 训练模型
            num_epochs = 100
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y.view(-1))
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                
                avg_loss = running_loss / len(train_loader)
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y.view(-1))
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 测试集评估
            model.eval()
            y_pred_all = []
            y_test_all = []
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    y_pred = model(batch_X).cpu().numpy()
                    y_pred_all.extend(y_pred)
                    y_test_all.extend(batch_y.cpu().numpy().flatten())
            
            y_pred_all = np.array(y_pred_all)
            y_test_all = np.array(y_test_all)
            
            # 计算指标
            mse = mean_squared_error(y_test_all, y_pred_all)
            pearson_corr, _ = pearsonr(y_test_all, y_pred_all)
            spearman_corr, _ = spearmanr(y_test_all, y_pred_all)
            
            print(f"Parameters: hidden_dim={hidden_dim}, dropout={dropout}, num_layers={num_layers}")
            print(f"MSE: {mse:.4f}, Pearson: {pearson_corr:.4f}, Spearman: {spearman_corr:.4f}")
            
            # 生成预测图
            plt.figure(figsize=(5, 4))
            plt.scatter(y_test_all, y_pred_all, color='blue', label='Data points', s=30)
            plt.plot(
                [min(y_test_all), max(y_test_all)],
                [min(y_test_all), max(y_test_all)],
                color='red', linewidth=2, label='Ideal'
            )
            plt.xlabel('True DDG', fontsize=8)
            plt.ylabel('Predicted DDG', fontsize=8)
            plt.title('True vs Predicted DDG - DNN', fontsize=8)
            plt.legend(loc='upper left', fontsize=8)
            plt.text(0.05, 0.83, f'Pearson Corr: {pearson_corr:.2f}', transform=plt.gca().transAxes, fontsize=8)
            plt.text(0.05, 0.78, f'Spearman Corr: {spearman_corr:.2f}', transform=plt.gca().transAxes, fontsize=8)
            
            # 保存预测图
            figname = f'DNN_prediction_hidden{hidden_dim}_drop{dropout}_layers{num_layers}.png'
            plt.savefig(os.path.join(output_dir, figname), dpi=300)
            plt.close()
            
            # 保存预测结果
            prediction_file = f'DNN_prediction_hidden{hidden_dim}_drop{dropout}_layers{num_layers}.csv'
            results_df = pd.DataFrame({'True DDG': y_test_all, 'Predicted DDG': y_pred_all})
            results_df.to_csv(os.path.join(output_dir, prediction_file), index=False)
            
            pearson_results.append((hidden_dim, dropout, num_layers, pearson_corr))
            
            del model

# 找到最佳参数
max_corr = -np.inf
best_hidden_dim = None
best_dropout = None
best_num_layers = None
for hidden_dim, dropout, num_layers, corr in pearson_results:
    if corr > max_corr:
        max_corr = corr
        best_hidden_dim = hidden_dim
        best_dropout = dropout
        best_num_layers = num_layers

# 打印最佳结果
print(f"\nBest Pearson Correlation: {max_corr:.4f}")
print(f"Best Parameters: hidden_dim={best_hidden_dim}, dropout={best_dropout}, num_layers={best_num_layers}")

# 保存皮尔逊结果
results_df = pd.DataFrame(
    pearson_results,
    columns=['Hidden Dim', 'Dropout', 'Num Layers', 'Pearson Corr']
)
results_df.to_csv(os.path.join(output_dir, 'DNN_UNIPORT_results.csv'), index=False)