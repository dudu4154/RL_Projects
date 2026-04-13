import torch
import torch.nn as nn
from torchview import draw_graph

# 定義你的 DQN 模型結構
class ReconstructedDQN(nn.Module):
    def __init__(self):
        super(ReconstructedDQN, self).__init__()
        self.fc1 = nn.Linear(17, 128) 
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.fc_action = nn.Linear(128, 11) 
        self.fc_param = nn.Linear(128, 64)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = torch.relu(self.fc1(x))
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        action = self.fc_action(x)
        param = self.fc_param(x)
        return action, param

# 實例化模型
model = ReconstructedDQN()

# 繪製圖表
model_graph = draw_graph(
    model, 
    input_size=(1, 1, 17), # 對應星海爭霸 AI 的輸入維度
    graph_name="SC2_AI_Model",
    roll=True,             # 將 LSTM 摺疊起來，讓圖表更簡潔漂亮
    expand_nested=True,    # 展開巢狀結構
    device='cpu'
)

# 儲存並產出圖檔
# 注意：你的電腦需要安裝 Graphviz 軟體才能產出 PNG
model_graph.visual_effect.render(format='png')
print("✅ 圖表已生成！請查看資料夾中的 SC2_AI_Model.png")