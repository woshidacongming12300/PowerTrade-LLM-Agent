import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


# --- 1. 定义 LSTM 神经网络模型结构 ---
class ElectricPriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(ElectricPriceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 全连接输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # 前向传播
        out, _ = self.lstm(x, (h0, c0))
        # 只取序列最后一个时间步的输出作为最终预测
        out = self.fc(out[:, -1, :])
        return out


# --- 2. 训练脚本 (只用跑一次) ---
def train_model(csv_path="../datasets/clean_electric_data.csv", seq_length=7):
    """
    读取数据并训练 LSTM，保存模型和归一化器参数
    """
    print("🚀 开始加载电力数据进行训练...")
    # 读取数据
    df = pd.read_csv(csv_path)

    # 清洗数据：按交割日排序，并提取加权平均价
    df['Delivery start date'] = pd.to_datetime(df['Delivery start date'], format='mixed')
    df = df.sort_values(by='Delivery start date')
    prices = df['Wtd avg price $/MWh'].values.reshape(-1, 1)

    # 归一化 (深度学习对数值范围很敏感，必须缩放到 0-1 之间)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)

    # 构造滑动窗口特征 (用过去 seq_length 天，预测第 seq_length+1 天)
    X, y = [], []
    for i in range(len(scaled_prices) - seq_length):
        X.append(scaled_prices[i:i + seq_length])
        y.append(scaled_prices[i + seq_length])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)

    # 实例化模型、损失函数和优化器
    model = ElectricPriceLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 开始训练循环 (轻量级，只跑 50 轮)
    print("🧠 开始训练 LSTM 模型...")
    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}')

    # 保存模型权重和归一化的最大最小值 (推理时需要用来反算真实电价)
    model_save_path = "lstm_epf.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_min': scaler.data_min_[0],
        'scaler_max': scaler.data_max_[0]
    }, model_save_path)
    print(f"✅ 模型训练完毕，已保存至 {model_save_path}")


# --- 3. 供 Agent 调用的推理接口 ---
def predict_future_price(target_date: str) -> float:
    """
    这正是我们要提供给 Agent 工具箱的黑盒函数！
    """
    # 1. 查找模型文件
    model_path = os.path.join(os.path.dirname(__file__), "lstm_epf.pth")
    if not os.path.exists(model_path):
        return 0.0  # 容错机制

    # 2. 加载模型和归一化参数
    checkpoint = torch.load(model_path, weights_only=False)
    scaler_min = checkpoint['scaler_min']
    scaler_max = checkpoint['scaler_max']

    model = ElectricPriceLSTM()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. 模拟获取过去 7 天的历史数据作为输入
    # (在真实生产环境中，这里会去数据库 select 过去7天的数据)
    # 这里为了跑通闭环，我们先用你在 CSV 里的一段真实序列片段的最后 7 天
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../datasets/clean_electric_data.csv"))
    last_7_prices = df['Wtd avg price $/MWh'].tail(7).values.reshape(-1, 1)

    # 预处理：按同样的比例归一化
    scaled_input = (last_7_prices - scaler_min) / (scaler_max - scaler_min)
    tensor_input = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0)  # [1, 7, 1]

    # 4. 执行模型推理
    with torch.no_grad():
        pred_scaled = model(tensor_input).item()

    # 5. 逆归一化，还原成真实的 $/MWh 价格
    real_price = pred_scaled * (scaler_max - scaler_min) + scaler_min

    return round(real_price, 4)


# --- 仅在直接运行此脚本时触发训练 ---
if __name__ == "__main__":
    train_model(csv_path="../datasets/clean_electric_data.csv")

    print("\n🔮 测试推理接口:")
    test_price = predict_future_price("明天")
    print(f"预测出的明日加权平均电价为: {test_price} $/MWh")