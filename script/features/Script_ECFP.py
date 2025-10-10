import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import AllChem

# 设置路径
csv_file_path = '/home/data1/BGM/fenziduiqi/All_data_output_modified.csv'
output_csv_path = '/home/data1/BGM/fenziduiqi/ECFP_values.csv'

# 读取CSV文件
data = pd.read_csv(csv_file_path)

# 提取SMILES列
smiles_column = data.iloc[:, -1]

# 提取蛋白质标识符（假设使用SMILES列的第一部分作为标识符）
# 你可以根据实际需求调整如何提取标识符
protein_names = data.iloc[:, 0]  # 假设蛋白质名称在第一列

# 将SMILES转换为ECFP
def smiles_to_ecfp(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # 生成ECFP（半径为2，位数为1024）
            ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            return list(ecfp)
        else:
            return None  # 返回None以处理无效的SMILES
    except Exception as e:
        print(f"Error processing SMILES: {smiles} - {str(e)}")
        return None

# 应用转换函数并过滤掉无效的ECFP
ecfp_list = smiles_column.apply(smiles_to_ecfp).dropna()

# 创建一个新的DataFrame，将蛋白质标识符放在第一列，后面是ECFP值
ecfp_df = pd.DataFrame(ecfp_list.tolist())  # 转换为DataFrame
ecfp_df.insert(0, 'Protein_ID', protein_names[ecfp_list.index])  # 将蛋白质标识符插入到第一列

# 保存ECFP到CSV
ecfp_df.to_csv(output_csv_path, index=False, header=False)

print(f"ECFP values have been saved to {output_csv_path}")
