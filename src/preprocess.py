import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# 1. 路径配置（相对路径，无转义报错）
dataset_root = "../OmniAnomaly-master/OmniAnomaly-master/ServerMachineDataset"
train_dir = os.path.join(dataset_root, "train")
test_dir = os.path.join(dataset_root, "test")
label_dir = os.path.join(dataset_root, "test_label")

# 2. 读取单文件、提取第0列CPU单变量
def load_single_ts(file_path):
    df = pd.read_csv(file_path, header=None)
    # 取第0列作为单变量时序
    ts = df.iloc[:, 0].values
    return ts

# 3. 论文规定：分段缺失值填充
def fill_missing(series, period=1440):
    data = pd.Series(series)
    idx_nan = data[data.isna()].index
    for pos in idx_nan:
        start = pos
        while start > 0 and pd.isna(data[start-1]):
            start -= 1
        end = pos
        while end < len(data) and pd.isna(data[end]):
            end += 1
        miss_len = end - start
        if miss_len < 5:
            # 短缺失：线性插值
            fill_vals = np.linspace(data[start-1], data[end], miss_len)
            data.iloc[start:end] = fill_vals
        else:
            # 长缺失：取前周期同位置填充
            offset = pos % period
            data.iloc[start:end] = data.iloc[offset : offset+miss_len].values
    return data.values

# 4. 构造滑动窗口
def create_window(series, win_size=10):
    windows = []
    for i in range(len(series) - win_size):
        windows.append(series[i:i+win_size])
    return np.array(windows)

# ---------------------- 主执行流程 ----------------------
if __name__ == "__main__":
    # 读取训练集（纯正常数据，用于FluxEV SPOT初始化）
    train_file = os.listdir(train_dir)[0]
    train_ts = load_single_ts(os.path.join(train_dir, train_file))
    train_ts = fill_missing(train_ts)
    
    # 读取测试集+真值标签
    test_file = os.listdir(test_dir)[0]
    test_ts = load_single_ts(os.path.join(test_dir, test_file))
    test_ts = fill_missing(test_ts)
    test_label = pd.read_csv(os.path.join(label_dir, test_file), header=None).values.flatten()

    # 归一化（统一缩放0~1）
    scaler = MinMaxScaler()
    train_ts = scaler.fit_transform(train_ts.reshape(-1,1)).flatten()
    test_ts = scaler.transform(test_ts.reshape(-1,1)).flatten()

    # 生成窗口样本
    train_win = create_window(train_ts, win_size=10)
    test_win = create_window(test_ts, win_size=10)
    # 标签同步裁剪（窗口末尾对应标签）
    test_label_cut = test_label[10:]

    # 划分训练/验证 49% / 21%
    split_1 = int(len(train_win) * 0.49)
    split_2 = int(len(train_win) * 0.7)
    train_data = train_win[:split_1]
    val_data = train_win[split_1:split_2]

    # 保存处理后数据集
    np.save("../exp_result/train.npy", train_data)
    np.save("../exp_result/val.npy", val_data)
    np.save("../exp_result/test.npy", test_win)
    np.save("../exp_result/test_label.npy", test_label_cut)

    print("预处理完成！")
    print(f"训练集窗口数量：{len(train_data)}")
    print(f"验证集窗口数量：{len(val_data)}")
    print(f"测试集窗口数量：{len(test_win)}")
    print(f"测试异常标签长度：{len(test_label_cut)}")