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
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 路径设置
csv_file = '/home/data1/BGM/fenziduiqi/check_embed_list.csv'
output_dir = '/home/data1/BGM/fenziduiqi/All_X-_BiLSTM_4179_result'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 加载 CSV 文件
df = pd.read_csv(csv_file)

# 解析嵌入向量
def parse_emb(emb_str):
    return np.array([float(x) for x in emb_str.split(',')])

# 处理 X1, X2, ECFP 列
X1 = np.array(df['X1'].apply(parse_emb).tolist())
X2 = np.array(df['X2'].apply(parse_emb).tolist())
df_ecfp = np.array(df['ECFP'].apply(parse_emb).tolist())
y = np.array(df['ddG'])

# 合并特征：改为 (X1-X2, df_ecfp)
X_diff = X1 - X2  # 计算 X1 和 X2 的差值
X = np.concatenate((X_diff, df_ecfp), axis=1)

# 定义 BiLSTM 模型
class BiLSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, num_layers=1, dropout=0.2):
        super(BiLSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # 设置为双向 LSTM
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc_final = nn.Linear(hidden_dim*2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # 提取最后一个时间步的隐藏状态
        last_hidden = self.dropout(last_hidden)
        x = self.fc_final(last_hidden)  # 保持输出为 [batch_size, 1]
        return x

# BiLSTM 超参数
hidden_dim_list = [16, 128, 256, 1024]
dropout_list = [0.1, 0.2, 0.3, 0.4, 0.5]
num_layers_list = [1, 2, 3]

pearson_results = []

# 10 次随机划分
for idx in range(10):
    random_seed = random.randint(1, 9999999)
    print(f"Random seed: {random_seed}")
    
    # 数据划分：80% 训练，10% 验证，10% 测试
    train_data, temp_data = train_test_split(df, test_size=0.2, random_state=random_seed)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # 训练集
    X1_train = np.array(train_data['X1'].apply(parse_emb).tolist())
    X2_train = np.array(train_data['X2'].apply(parse_emb).tolist())
    ecfp_train = np.array(train_data['ECFP'].apply(parse_emb).tolist())
    X_train_diff = X1_train - X2_train  # 计算差值
    X_train = np.concatenate((X_train_diff, ecfp_train), axis=1)
    y_train = np.array(train_data['ddG'])
    
    # 验证集
    X1_val = np.array(val_data['X1'].apply(parse_emb).tolist())
    X2_val = np.array(val_data['X2'].apply(parse_emb).tolist())
    ecfp_val = np.array(val_data['ECFP'].apply(parse_emb).tolist())
    X_val_diff = X1_val - X2_val  # 计算差值
    X_val = np.concatenate((X_val_diff, ecfp_val), axis=1)
    y_val = np.array(val_data['ddG'])
    
    # 测试集
    X1_test = np.array(test_data['X1'].apply(parse_emb).tolist())
    X2_test = np.array(test_data['X2'].apply(parse_emb).tolist())
    ecfp_test = np.array(test_data['ECFP'].apply(parse_emb).tolist())
    X_test_diff = X1_test - X2_test  # 计算差值
    X_test = np.concatenate((X_test_diff, ecfp_test), axis=1)
    y_test = np.array(test_data['ddG'])
    
    print(f"Train set size: {len(y_train)}")
    print(f"Test set size: {len(y_test)}")
    print(f"Shapes: {X_train.shape}, {X_val.shape}, {X_test.shape}")
    
    # 调整数据形状以适配 BiLSTM
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val_lstm = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    # 转换为张量，保持 y 为 [batch_size, 1]
    X_train_tensor = torch.from_numpy(X_train_lstm).float()
    y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)
    X_val_tensor = torch.from_numpy(X_val_lstm).float()
    y_val_tensor = torch.from_numpy(y_val).float().view(-1, 1)
    X_test_tensor = torch.from_numpy(X_test_lstm).float()
    y_test_tensor = torch.from_numpy(y_test).float().view(-1, 1)
    
    # 创建 DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    for hidden_dim in hidden_dim_list:
        for dropout in dropout_list:
            for num_layers in num_layers_list:
                # 初始化模型
                model = BiLSTMRegressor(
                    input_dim=X_train_lstm.shape[2],
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout
                )
                criterion = nn.MSELoss()
                optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
                
                # 使用 GPU
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                        outputs = model(batch_X)  # 输出为 [batch_size, 1]
                        loss = criterion(outputs, batch_y)  # 保持 batch_y 为 [batch_size, 1]
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
                            loss = criterion(outputs, batch_y)
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
                        y_pred = model(batch_X).cpu().numpy()  # [batch_size, 1]
                        y_pred_all.extend(y_pred.flatten())  # 展平为一维以用于评估
                        y_test_all.extend(batch_y.cpu().numpy().flatten())  # 展平为一维以用于评估
                
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
                plt.title('True vs Predicted DDG - BiLSTM', fontsize=8)
                plt.legend(loc='upper left', fontsize=8)
                plt.text(0.05, 0.83, f'Pearson Corr: {pearson_corr:.2f}', transform=plt.gca().transAxes, fontsize=8)
                plt.text(0.05, 0.78, f'Spearman Corr: {spearman_corr:.2f}', transform=plt.gca().transAxes, fontsize=8)
                
                # 保存预测图
                figname = f'BiLSTM_prediction_random{idx}_hidden{hidden_dim}_drop{dropout}_layers{num_layers}.png'
                plt.savefig(os.path.join(output_dir, figname), dpi=300)
                plt.close()
                
                # 保存预测结果
                prediction_file = f'BiLSTM_prediction_random{idx}_hidden{hidden_dim}_drop{dropout}_layers{num_layers}.csv'
                results_df = pd.DataFrame({'True DDG': y_test_all, 'Predicted DDG': y_pred_all})
                results_df.to_csv(os.path.join(output_dir, prediction_file), index=False)
                
                pearson_results.append((random_seed, hidden_dim, dropout, num_layers, pearson_corr))
                
                del model

# 找到最佳参数
max_corr = -np.inf
best_seed = None
best_hidden_dim = None
best_dropout = None
best_num_layers = None
for seed, hidden_dim, dropout, num_layers, corr in pearson_results:
    if corr > max_corr:
        max_corr = corr
        best_seed = seed
        best_hidden_dim = hidden_dim
        best_dropout = dropout
        best_num_layers = num_layers

# 打印最佳结果
print(f"\nBest Pearson Correlation: {max_corr:.4f}")
print(f"Best Parameters: random_seed={best_seed}, hidden_dim={best_hidden_dim}, dropout={best_dropout}, num_layers={best_num_layers}")

# 保存皮尔逊结果
results_df = pd.DataFrame(
    pearson_results,
    columns=['Random Seed', 'Hidden Dim', 'Dropout', 'Num Layers', 'Pearson Corr']
)
results_df.to_csv(os.path.join(output_dir, 'BiLSTM_results.csv'), index=False)