import torch
import torch.nn as nn

# 根據錯誤訊息修正後的維度結構
class ReconstructedDQN(nn.Module):
    def __init__(self):
        super(ReconstructedDQN, self).__init__()
        # 第一層：輸入 17 -> 輸出 128
        self.fc1 = nn.Linear(17, 128) 
        # LSTM 層：輸入 128 -> 隱藏層 128 (4*128=512)
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        # 輸出層：輸入 128 (來自 LSTM)
        self.fc_action = nn.Linear(128, 11) 
        self.fc_param = nn.Linear(128, 64)  

    def forward(self, x):
        # 確保輸入是 (batch, seq, feature)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = torch.relu(self.fc1(x))
        x, _ = self.lstm(x)
        x = x[:, -1, :] # 取最後一個時間步
        
        action = self.fc_action(x)
        param = self.fc_param(x)
        return action, param

# 初始化模型
model = ReconstructedDQN()

# 載入權重
try:
    state_dict = torch.load('dqn_model.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    print("權重載入成功！")
except Exception as e:
    print(f"載入失敗: {e}")

model.eval()

# 建立虛擬輸入 (Batch Size=1, Input Dim=17)
dummy_input = torch.randn(1, 17)

# 導出為 ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    "dqn_model_flow.onnx", 
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input'], 
    output_names=['action', 'param']
)

print("轉換完成！請將 dqn_model_flow.onnx 拖入 Netron。")