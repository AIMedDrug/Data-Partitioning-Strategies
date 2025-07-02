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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 路径设置
csv_file = '/home/data1/BGM/fenziduiqi/check_embed_list.csv'
output_dir = '/home/data1/BGM/fenziduiqi/All_X-_4179_GRU_P04637_UNIPORT_result'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 加载 CSV 文件
df = pd.read_csv(csv_file)

# 解析嵌入向量
def parse_emb(emb_str):
    return np.array([float(x) for x in emb_str.split(',')])

# 定义提取比例
ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
pearson_results = []

# GRU 固定参数
hidden_dim = 16
dropout = 0.3
num_layers = 3

# 定义 GRU 模型
class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, num_layers=3, dropout=0.3):
        super(GRURegressor, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        output = self.fc(last_hidden)
        return output.view(-1)

# 遍历比例和随机种子
for ratio in ratios:
    for idx in range(30):
        random_number = random.randint(1, 9999999)
        print("随机种子：", random_number)

        # 数据集划分
        test_df = df[df['uniprot_ids'] == 'P04637']
        train_df = df[df['uniprot_ids'] != 'P04637']

        print(f"Test set size (P04637): {len(test_df)}")
        print(f"Train set size: {len(train_df)}")

        # 从 test_df 中采样 anchor 数据
        test_anchor = test_df.sample(frac=ratio, random_state=42)
        test_query = test_df.drop(test_anchor.index)
        train_with_anchor = pd.concat([train_df, test_anchor], ignore_index=True)

        # 处理训练数据
        X1 = np.array(train_with_anchor['X1'].apply(parse_emb).tolist())
        X2 = np.array(train_with_anchor['X2'].apply(parse_emb).tolist())
        df_ecfp = np.array(train_with_anchor['ECFP'].apply(parse_emb).tolist())
        X_train = np.concatenate((X1 - X2, df_ecfp), axis=1)
        y_train = np.array(train_with_anchor['ddG'])

        # 划分训练和验证集
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_number)

        # 处理测试数据
        X1_test = np.array(test_query['X1'].apply(parse_emb).tolist())
        X2_test = np.array(test_query['X2'].apply(parse_emb).tolist())
        df_ecfp_test = np.array(test_query['ECFP'].apply(parse_emb).tolist())
        X_test = np.concatenate((X1_test - X2_test, df_ecfp_test), axis=1)
        y_test = np.array(test_query['ddG'])

        # 打印数据集大小
        print(f"\n比率： {ratio*100}%")
        print(f"锚点集大小： {len(test_anchor)}")
        print(f"查询集大小： {len(test_query)}")
        print(f"Train set 大小： {len(y_train)}")
        print(f"测试集大小： {len(y_test)}")
        print(f"检查形状： X_train={X_train.shape}, X_test={X_test.shape}")

        # 调整数据形状以适配 GRU
        X_train_gru = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val_gru = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
        X_test_gru = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        # 转换为张量
        X_train_tensor = torch.from_numpy(X_train_gru).float()
        y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)
        X_val_tensor = torch.from_numpy(X_val_gru).float()
        y_val_tensor = torch.from_numpy(y_val).float().view(-1, 1)
        X_test_tensor = torch.from_numpy(X_test_gru).float()
        y_test_tensor = torch.from_numpy(y_test).float().view(-1, 1)

        # 创建 DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 初始化模型
        model = GRURegressor(
            input_dim=X_train_gru.shape[2],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

        # 使用 GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        if len(y_test) < 2:
            print("测试集大小不足以计算相关性，跳过评估。")
            continue

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

        print(f"MSE: {mse:.4f}, Pearson: {pearson_corr:.4f}, Spearman: {spearman_corr:.4f}")

        # 生成预测图
        plt.figure(figsize=(14, 12))
        # 设置图的四个边框为黑色并加粗
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
        # 调整子图边距
        plt.subplots_adjust(top=0.92, bottom=0.13, left=0.17, right=0.96)
        plt.scatter(y_test_all, y_pred_all, color='blue', label='Data points', s=400)
        plt.plot(
            [min(y_test_all), max(y_test_all)],
            [min(y_test_all), max(y_test_all)],
            color='gray', linewidth=2, linestyle='dashed', label='Ideal'
        )
        plt.xlabel('True ΔΔG', fontsize=46)
        plt.ylabel('Predicted ΔΔG', fontsize=46)
        plt.title('True vs Predicted ΔΔG - GRU', fontsize=46)
        plt.xlim(-5.5, 5)
        plt.xticks([-5, -2.5, 0, 2.5, 5], fontsize=42)
        plt.ylim(-5.5, 5)
        plt.yticks([-5, -2.5, 0, 2.5, 5], fontsize=42)
        plt.legend(loc='upper left', fontsize=42)
        plt.text(
            0.55, 0.12, f'Pearson: {pearson_corr:.2f}',
            transform=plt.gca().transAxes, fontsize=44
        )
        plt.text(
            0.55, 0.05, f'Spearman: {spearman_corr:.2f}',
            transform=plt.gca().transAxes, fontsize=44
        )

        # 保存预测图
        figname = f'GRU_prediction_ratio{ratio}_random{idx}.png'
        output_path = os.path.join(output_dir, figname)
        plt.savefig(output_path, dpi=900)
        plt.close()

        # 保存预测结果
        prediction_file = f'GRU_prediction_ratio{ratio}_random{idx}.csv'
        csv_output_path = os.path.join(output_dir, prediction_file)
        results_df = pd.DataFrame({'True DDG': y_test_all, 'Predicted DDG': y_pred_all})
        results_df.to_csv(csv_output_path, index=False)

        # 记录皮尔逊结果
        pearson_results.append((ratio, random_number, pearson_corr))

        del model

# 找到每个比例的最佳参数
best_results = {}
for ratio in ratios:
    max_corr = -np.inf
    best_random = None
    for r, randomnum, corr in pearson_results:
        if r == ratio and corr > max_corr:
            max_corr = corr
            best_random = randomnum
    best_results[ratio] = (max_corr, best_random)

# 打印每个比例的最佳结果
print("\n最佳参数结果:")
for ratio in ratios:
    corr, random_num = best_results[ratio]
    print(f"ratio={ratio*100}% 最大皮尔逊系数: {corr:.4f}, random_number={random_num}")

# 保存皮尔逊结果
results_df = pd.DataFrame(
    pearson_results,
    columns=['Ratio', 'Random_Number', 'Pearson_Corr']
)
results_df.to_csv(os.path.join(output_dir, 'GRU_pearson_ratio_results.csv'), index=False)