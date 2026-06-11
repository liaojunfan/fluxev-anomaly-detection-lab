import pandas as pd
import os

# 改用相对路径，避免转义报错
dataset_root = "../OmniAnomaly-master/OmniAnomaly-master/ServerMachineDataset"
train_path = os.path.join(dataset_root, "train")

# 读取第一条训练文件
sample_file = os.listdir(train_path)[0]
df = pd.read_csv(os.path.join(train_path, sample_file), header=None)
print("数据集读取成功，数据维度：", df.shape)
print("前5行数据：")
print(df.head())