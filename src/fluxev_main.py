import numpy as np
import matplotlib.pyplot as plt
# 解决matplotlib中文方框乱码
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

from sklearn.metrics import precision_score, recall_score, f1_score

# ---------------------- 1. 加载预处理好的数据 ----------------------
def load_data():
    base = "../exp_result/"
    train = np.load(base + "train.npy")    # (样本数, 窗口长度10)
    val = np.load(base + "val.npy")
    test = np.load(base + "test.npy")
    test_label = np.load(base + "test_label.npy") # 和test一一对应标签
    return train, val, test, test_label

# ---------------------- 2. EWMA 波动特征提取（每个窗口输出1个分数） ----------------------
def ewma_score_per_window(win_data, alpha=0.3):
    score_list = []
    for win in win_data:
        ewma_val = np.zeros_like(win)
        ewma_val[0] = win[0]
        for t in range(1, len(win)):
            ewma_val[t] = alpha * win[t] + (1 - alpha) * ewma_val[t-1]
        # 取窗口最后一个值作为该样本波动分数
        score_list.append(ewma_val[-1])
    return np.array(score_list)

# ---------------------- 3. POT极值理论动态阈值类 ----------------------
class POT:
    def __init__(self, q=0.92):  # 下调分位数，降低阈值，能检出异常
        self.q = q
        self.threshold = None

    def fit(self, normal_scores):
        # 使用正常数据计算分位数作为异常阈值
        u = np.quantile(normal_scores, self.q)
        self.threshold = u
        print(f"POT自适应异常阈值：{self.threshold:.4f}")

    def predict(self, score):
        # 超过阈值=异常(1)，否则正常(0)
        return 1 if score > self.threshold else 0

# ---------------------- 4. FluxEV 完整检测主逻辑 ----------------------
def fluxev_detect(train_win, test_win):
    # 计算训练集、测试集每个窗口的EWMA波动分数
    train_scores = ewma_score_per_window(train_win)
    test_scores = ewma_score_per_window(test_win)

    # 用纯正常训练数据拟合阈值
    pot = POT(q=0.98)
    pot.fit(train_scores)

    # 逐窗口生成异常预测标签，长度与test_label完全匹配
    y_pred = np.array([pot.predict(s) for s in test_scores])
    return y_pred, test_scores

# ---------------------- 5. 指标评估 + 绘图可视化 ----------------------
def evaluate(y_true, y_pred, score):
    # 计算异常检测标准指标
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("===== FluxEV 异常检测评估指标 =====")
    print(f"精确率 Precision: {prec:.4f}")
    print(f"召回率 Recall:    {rec:.4f}")
    print(f"F1综合分数 F1-Score:   {f1:.4f}")

    # 绘制前2000个样本时序图
    plt.figure(figsize=(16,5), dpi=120)
    plt.plot(score[:2000], label="EWMA波动分数", c="#1f77b4", linewidth=1.2)
    # 筛选出预测异常的坐标
    anomaly_idx = np.where(y_pred[:2000]==1)[0]
    plt.scatter(anomaly_idx, score[anomaly_idx], c="red", s=10, label="预测异常样本")
    plt.xlabel("测试窗口序号")
    plt.ylabel("EWMA波动分数")
    plt.title("SMD服务器CPU时序 EWMA波动与异常检测结果")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    # 保存图片到exp_result文件夹
    plt.savefig("../exp_result/result_plot.png", dpi=150, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    train_win, val_win, test_win, test_label = load_data()
    # 输出长度校验，确保标签与预测对齐
    print(f"真实标签总长度：{len(test_label)}")
    print(f"测试窗口总数量：{len(test_win)}")

    y_pred, test_score = fluxev_detect(train_win, test_win)
    evaluate(test_label, y_pred, test_score)