import atexit
import csv
import os
import random
import datetime
import subprocess
import time
import ctypes
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import sc2_env
from absl import app

# ─────────────────────────────────────────────
#  儲存路徑設定
# ─────────────────────────────────────────────
SAVE_DIR        = Path(r"C:\RL_Projects\CYJ_and_don't_touch\my_agent\models\Simple96")
MODEL_PATH      = SAVE_DIR / "ppo_marauder.pth"
CSV_PATH        = SAVE_DIR / "training_log.csv"
SAVE_EVERY_EP   = 10   # 每幾個 episode 自動存一次模型
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  程式結束時自動關閉 SC2 視窗
# ─────────────────────────────────────────────
def _kill_sc2():
    """強制結束所有 SC2_x64 process，確保視窗不殘留。"""
    try:
        subprocess.call(
            ["taskkill", "/F", "/IM", "SC2_x64.exe"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("[Cleanup] SC2 視窗已關閉")
    except Exception as e:
        print(f"[Cleanup] 關閉 SC2 失敗：{e}")

# 無論正常結束、Ctrl+C、或例外，都會觸發
atexit.register(_kill_sc2)

# ─────────────────────────────────────────────
#  強制 SC2 視窗到前景並觸發重繪（修正黑屏）
# ─────────────────────────────────────────────
def _focus_sc2_window():
    """
    找到 SC2 視窗並強制拉到前景、觸發重繪。
    解決 NVIDIA + VS Code 環境下 SC2 視窗全黑問題。
    """
    try:
        user32 = ctypes.windll.user32

        # EnumWindows 找到標題含 StarCraft II 的視窗
        hwnd_found = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
        def enum_callback(hwnd, lparam):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                if "StarCraft" in buf.value or "星海爭霸" in buf.value:
                    hwnd_found.append(hwnd)
            return True

        user32.EnumWindows(enum_callback, 0)

        if hwnd_found:
            hwnd = hwnd_found[0]
            SW_RESTORE = 9
            user32.ShowWindow(hwnd, SW_RESTORE)          # 還原視窗（若最小化）
            user32.SetForegroundWindow(hwnd)              # 拉到前景
            user32.RedrawWindow(hwnd, None, None,         # 強制重繪
                ctypes.c_uint(0x0001 | 0x0100 | 0x0200)) # RDW_INVALIDATE|ERASE|ALLCHILDREN
            print("[Display] SC2 視窗已拉到前景並觸發重繪")
        else:
            print("[Display] 找不到 SC2 視窗，跳過重繪")
    except Exception as e:
        print(f"[Display] 視窗操作失敗（非 Windows 環境？）：{e}")

# ─────────────────────────────────────────────
#  CSV 工具：初始化標頭 & 寫入一行
# ─────────────────────────────────────────────
def csv_init():
    """若 CSV 不存在則建立並寫入標頭列。"""
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Episode", "Timestamp",
                "Reward",
                "PolicyLoss", "ValueLoss", "Loss", "Entropy",
            ])
        print(f"[CSV] 已建立訓練紀錄檔：{CSV_PATH}")

def csv_write(episode: int, reward: float,
              policy_loss: float, value_loss: float,
              total_loss: float, entropy: float):
    """追加一筆訓練紀錄。"""
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            episode,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"{reward:.4f}",
            f"{policy_loss:.6f}",
            f"{value_loss:.6f}",
            f"{total_loss:.6f}",
            f"{entropy:.6f}",
        ])

# ─────────────────────────────────────────────
#  模型存檔 & 讀取工具
# ─────────────────────────────────────────────
def save_checkpoint(net: "FullyConvNet", optimizer, episode: int):
    """儲存模型權重、optimizer 狀態及目前 episode 編號。"""
    checkpoint = {
        "episode":        episode,
        "model_state":    net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(checkpoint, MODEL_PATH)
    print(f"[Checkpoint] 已儲存 episode={episode} → {MODEL_PATH}")

def load_checkpoint(net: "FullyConvNet", optimizer) -> int:
    """
    若 MODEL_PATH 存在則載入，回傳上次儲存的 episode 編號（用於接續）。
    若不存在則回傳 0（從頭開始）。
    """
    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        net.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        ep = checkpoint.get("episode", 0)
        print(f"[Checkpoint] 載入已有模型，從 episode={ep + 1} 繼續訓練 ← {MODEL_PATH}")
        return ep
    else:
        print(f"[Checkpoint] 未找到既有模型，從頭開始訓練")
        return 0

# ─────────────────────────────────────────────
#  常數定義
# ─────────────────────────────────────────────
COMMAND_CENTER_ID     = 18
SCV_ID                = 45
SUPPLY_DEPOT_ID       = 19
REFINERY_ID           = 20
BARRACKS_ID           = 21
BARRACKS_TECHLAB_ID   = 37
GEYSER_ID             = 342
MINERAL_FIELD_ID      = 341
MARAUDER_ID           = 51

NO_OP                    = actions.FUNCTIONS.no_op()
HARVEST_ACTION           = actions.FUNCTIONS.Harvest_Gather_screen.id
TRAIN_SCV_ACTION         = actions.FUNCTIONS.Train_SCV_quick.id
BUILD_SUPPLYDEPOT_ACTION = actions.FUNCTIONS.Build_SupplyDepot_screen.id
BUILD_REFINERY_ACTION    = actions.FUNCTIONS.Build_Refinery_screen.id
BUILD_BARRACKS_ACTION    = actions.FUNCTIONS.Build_Barracks_screen.id
BUILD_TECHLAB_ACTION     = actions.FUNCTIONS.Build_TechLab_quick.id
TRAIN_MARAUDER_ACTION    = actions.FUNCTIONS.Train_Marauder_quick.id
ATTACK_SCREEN_ACTION     = actions.FUNCTIONS.Attack_screen.id
MOVE_SCREEN_ACTION       = actions.FUNCTIONS.Move_screen.id
SELECT_ARMY_ACTION       = actions.FUNCTIONS.select_army.id
SELECT_POINT_ACTION      = actions.FUNCTIONS.select_point.id

SCREEN_SIZE      = 84
CAMERA_INTERVAL  = 8   # 每 8 步才移動一次視角，不干擾 PPO rollout

# ─────────────────────────────────────────────
#  動作空間定義（6 個離散動作）
# ─────────────────────────────────────────────
ACTION_MOVE_AWAY    = 0   # 遠離敵人
ACTION_MOVE_TOWARD  = 1   # 靠近敵人
ACTION_MOVE_SCREEN  = 2   # A-Move（點擊地面）
ACTION_ATTACK       = 3   # 攻擊（點擊敵人）
ACTION_SELECT_ALL   = 4   # 全選掠奪者
ACTION_SELECT_ONE   = 5   # 單選掠奪者
NUM_ACTIONS = 6

# ─────────────────────────────────────────────
#  PPO 超參數
# ─────────────────────────────────────────────
GAMMA       = 0.99
LAM         = 0.95
CLIP_EPS    = 0.2
LR          = 3e-4
EPOCHS      = 4
MINI_BATCH  = 32
ROLLOUT_LEN = 128
ENT_COEF    = 0.01
VF_COEF     = 0.5
MAX_GRAD    = 0.5

# ─────────────────────────────────────────────
#  FullyConv 神經網路
#  輸入通道 (4 張 84×84 feature map):
#    ch0: 敵方位置（0/1）
#    ch1: 我方位置（0/1）
#    ch2: 敵方血量（僅敵方格子，0–1）
#    ch3: 我方血量（僅我方格子，0–1）
# ─────────────────────────────────────────────
class FullyConvNet(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=1, padding=2)  # 84→84
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # 84→84
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # 84→42

        flat_dim = 64 * 42 * 42  # 113,568

        self.policy_fc  = nn.Linear(flat_dim, 256)
        self.policy_out = nn.Linear(256, num_actions)

        self.value_fc  = nn.Linear(flat_dim, 256)
        self.value_out = nn.Linear(256, 1)

    def forward(self, x):
        """x: (B, 4, 84, 84)  →  (logits, value)"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)

        logits = self.policy_out(F.relu(self.policy_fc(x)))
        value  = self.value_out(F.relu(self.value_fc(x))).squeeze(-1)
        return logits, value


# ─────────────────────────────────────────────
#  PPO 訓練器（GAE + Mini-batch + Action Masking）
# ─────────────────────────────────────────────
class PPOTrainer:
    def __init__(self, net: FullyConvNet):
        self.net = net
        self.optimizer = optim.Adam(net.parameters(), lr=LR)
        self._clear()

    def _clear(self):
        self.obs_buf  = []
        self.act_buf  = []
        self.logp_buf = []
        self.rew_buf  = []
        self.val_buf  = []
        self.done_buf = []
        self.mask_buf = []

    def store(self, obs, action, logp, reward, value, done, mask):
        self.obs_buf.append(obs)
        self.act_buf.append(action)
        self.logp_buf.append(logp)
        self.rew_buf.append(reward)
        self.val_buf.append(value)
        self.done_buf.append(done)
        self.mask_buf.append(mask)

    def ready(self):
        return len(self.rew_buf) >= ROLLOUT_LEN

    def _compute_gae(self, last_value: float):
        """Generalized Advantage Estimation"""
        rewards = np.array(self.rew_buf,  dtype=np.float32)
        values  = np.array(self.val_buf,  dtype=np.float32)
        dones   = np.array(self.done_buf, dtype=np.float32)
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + GAMMA * next_val * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * LAM * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def train(self, last_value: float) -> tuple[float, float, float, float] | None:
        """
        執行 PPO 更新。
        回傳 (policy_loss, value_loss, total_loss, entropy)；
        若 buffer 為空則回傳 None。
        """
        advantages, returns = self._compute_gae(last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t  = torch.stack(self.obs_buf).float()
        act_t  = torch.tensor(self.act_buf,  dtype=torch.long)
        logp_t = torch.tensor(self.logp_buf, dtype=torch.float32)
        adv_t  = torch.tensor(advantages,    dtype=torch.float32)
        ret_t  = torch.tensor(returns,       dtype=torch.float32)
        mask_t = torch.stack(self.mask_buf).float()

        T = obs_t.size(0)
        indices = np.arange(T)
        policy_loss = value_loss = total_loss = entropy = None

        for _ in range(EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, T, MINI_BATCH):
                mb = indices[start: start + MINI_BATCH]
                mb_obs  = obs_t[mb];  mb_act  = act_t[mb]
                mb_logp = logp_t[mb]; mb_adv  = adv_t[mb]
                mb_ret  = ret_t[mb];  mb_mask = mask_t[mb]

                logits, value = self.net(mb_obs)
                logits = logits + (1 - mb_mask) * (-1e9)

                dist     = Categorical(logits=logits)
                new_logp = dist.log_prob(mb_act)
                entropy  = dist.entropy().mean()

                ratio = torch.exp(new_logp - mb_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = F.mse_loss(value, mb_ret)
                total_loss  = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), MAX_GRAD)
                self.optimizer.step()

        self._clear()
        if policy_loss is None:
            return None

        pl = policy_loss.item()
        vl = value_loss.item()
        tl = total_loss.item()
        en = entropy.item()
        print(f"[PPO] policy={pl:.4f}  value={vl:.4f}  "
              f"total={tl:.4f}  entropy={en:.4f}")
        return pl, vl, tl, en


# ─────────────────────────────────────────────
#  觀察值建構：4 通道 84×84 feature map
#
#  修正 #4：血量 channel 僅包含敵方/我方各自的格子，
#           使用 player_relative mask 精確隔離，
#           不混入 SCV、建築等其他單位。
# ─────────────────────────────────────────────
def build_obs_tensor(obs) -> torch.Tensor:
    player_rel = obs.observation.feature_screen[
        features.SCREEN_FEATURES.player_relative.index]
    hp_ratio   = obs.observation.feature_screen[
        features.SCREEN_FEATURES.unit_hit_points_ratio.index].astype(np.float32)

    # 精確 mask：只用 player_relative 區分敵我
    enemy_mask = (player_rel == features.PlayerRelative.ENEMY).astype(np.float32)
    self_mask  = (player_rel == features.PlayerRelative.SELF).astype(np.float32)

    # 血量只取對應陣營格子，再正規化到 0–1
    enemy_hp = enemy_mask * (hp_ratio / 255.0)
    self_hp  = self_mask  * (hp_ratio / 255.0)

    stacked = np.stack([enemy_mask, self_mask, enemy_hp, self_hp], axis=0)
    return torch.from_numpy(stacked)


# ─────────────────────────────────────────────
#  無效動作遮罩
# ─────────────────────────────────────────────
def build_action_mask(obs, enemy_exists: bool, marauder_exists: bool) -> torch.Tensor:
    available = obs.observation.available_actions
    mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32)

    if MOVE_SCREEN_ACTION in available and marauder_exists:
        mask[ACTION_MOVE_AWAY]   = 1
        mask[ACTION_MOVE_TOWARD] = 1
        mask[ACTION_MOVE_SCREEN] = 1

    if ATTACK_SCREEN_ACTION in available and enemy_exists and marauder_exists:
        mask[ACTION_ATTACK] = 1

    if SELECT_ARMY_ACTION in available:
        mask[ACTION_SELECT_ALL] = 1

    if marauder_exists:
        mask[ACTION_SELECT_ONE] = 1

    # 防止全為 0
    if mask.sum() == 0:
        mask[ACTION_SELECT_ALL] = 1

    return mask


# ─────────────────────────────────────────────
#  離散動作 → PySC2 指令
# ─────────────────────────────────────────────
def discrete_to_sc2(action_id: int, obs) -> actions.FunctionCall:
    available  = obs.observation.available_actions
    screen     = obs.observation.feature_screen
    player_rel = screen[features.SCREEN_FEATURES.player_relative.index]
    unit_type  = screen[features.SCREEN_FEATURES.unit_type.index]

    ey, ex = (player_rel == features.PlayerRelative.ENEMY).nonzero()
    my, mx = (unit_type  == MARAUDER_ID).nonzero()

    enemy_center = (int(ex.mean()), int(ey.mean())) if ex.size > 0 else (42, 42)
    self_center  = (int(mx.mean()), int(my.mean())) if mx.size > 0 else (42, 42)

    if action_id == ACTION_MOVE_AWAY:
        if MOVE_SCREEN_ACTION in available:
            dx = self_center[0] - enemy_center[0]
            dy = self_center[1] - enemy_center[1]
            norm = max(float(np.sqrt(dx**2 + dy**2)), 1e-5)
            tx = int(np.clip(self_center[0] + 20 * dx / norm, 0, SCREEN_SIZE - 1))
            ty = int(np.clip(self_center[1] + 20 * dy / norm, 0, SCREEN_SIZE - 1))
            return actions.FUNCTIONS.Move_screen("now", (tx, ty))

    elif action_id == ACTION_MOVE_TOWARD:
        if MOVE_SCREEN_ACTION in available:
            dx = enemy_center[0] - self_center[0]
            dy = enemy_center[1] - self_center[1]
            norm = max(float(np.sqrt(dx**2 + dy**2)), 1e-5)
            tx = int(np.clip(self_center[0] + 15 * dx / norm, 0, SCREEN_SIZE - 1))
            ty = int(np.clip(self_center[1] + 15 * dy / norm, 0, SCREEN_SIZE - 1))
            return actions.FUNCTIONS.Move_screen("now", (tx, ty))

    elif action_id == ACTION_MOVE_SCREEN:
        if MOVE_SCREEN_ACTION in available:
            tx = int((self_center[0] + enemy_center[0]) / 2)
            ty = int((self_center[1] + enemy_center[1]) / 2)
            return actions.FUNCTIONS.Move_screen("now", (tx, ty))

    elif action_id == ACTION_ATTACK:
        if ATTACK_SCREEN_ACTION in available and ex.size > 0:
            hp_ratio = screen[features.SCREEN_FEATURES.unit_hit_points_ratio.index]
            hp_vals  = hp_ratio[ey, ex]
            min_idx  = np.argmin(hp_vals)
            return actions.FUNCTIONS.Attack_screen("now", (int(ex[min_idx]), int(ey[min_idx])))

    elif action_id == ACTION_SELECT_ALL:
        if SELECT_ARMY_ACTION in available:
            return actions.FUNCTIONS.select_army("select")

    elif action_id == ACTION_SELECT_ONE:
        if mx.size > 0:
            idx = random.randint(0, mx.size - 1)
            return actions.FUNCTIONS.select_point("select", (int(mx[idx]), int(my[idx])))

    return NO_OP


# ─────────────────────────────────────────────
#  報酬係數設定（集中在此方便調整）
# ─────────────────────────────────────────────
R_ENEMY_HP_LOSS   =  0.05   # 每步敵方掉血獎勵
R_SELF_HP_LOSS    = -0.05   # 每步我方掉血懲罰
R_KILL            =  1.0    # 擊殺一隻敵方單位
R_ALLY_DEATH      = -0.5    # 我方一隻單位陣亡
R_WIN             =  3.0    # 全殲勝利
R_LOSE            = -1.0    # 失敗（我方全滅或時間到）
R_TIME_PENALTY    = -0.001  # 每步時間懲罰（鼓勵快速結束）
R_GRAZE_PENALTY   = -0.1    # 刮痧懲罰（連續10步有掉血但無擊殺）
GRAZE_WINDOW      = 10      # 刮痧判定窗口（步數）

# ─────────────────────────────────────────────
#  報酬函式
#
#  設計原則：
#    1. 所有單步獎勵絕對值 <= 0.05，終局獎勵最大 3.0
#    2. 不依賴稀少大事件主導梯度
#    3. 刮痧懲罰用窗口判定，不干擾正常攻擊流程
#    4. 時間懲罰鼓勵積極進攻
# ─────────────────────────────────────────────
def compute_reward(obs, prev_obs, graze_counter: list) -> float:
    """
    graze_counter: 長度 1 的 list，用於跨步持久化刮痧計數器
                   傳 list 是為了讓函式內可修改外部變數（Python 可變物件）
    """
    reward = 0.0
    screen      = obs.observation.feature_screen
    prev_screen = prev_obs.observation.feature_screen
    score       = obs.observation.score_cumulative
    prev_score  = prev_obs.observation.score_cumulative
    player_rel      = screen[features.SCREEN_FEATURES.player_relative.index]
    prev_player_rel = prev_screen[features.SCREEN_FEATURES.player_relative.index]
    hp_ratio        = screen[features.SCREEN_FEATURES.unit_hit_points_ratio.index].astype(np.float32)
    prev_hp_ratio   = prev_screen[features.SCREEN_FEATURES.unit_hit_points_ratio.index].astype(np.float32)

    # ── 1. 時間懲罰（每步固定扣，鼓勵快速結束）──
    reward += R_TIME_PENALTY

    # ── 2. 敵方 HP 損失獎勵 ──
    enemy_mask      = (player_rel      == features.PlayerRelative.ENEMY).astype(np.float32)
    prev_enemy_mask = (prev_player_rel == features.PlayerRelative.ENEMY).astype(np.float32)
    enemy_hp_now    = (enemy_mask      * hp_ratio).sum()
    enemy_hp_prev   = (prev_enemy_mask * prev_hp_ratio).sum()
    enemy_hp_delta  = enemy_hp_prev - enemy_hp_now          # 掉血為正
    if enemy_hp_delta > 0:
        reward += R_ENEMY_HP_LOSS

    # ── 3. 我方 HP 損失懲罰 ──
    self_mask      = (player_rel      == features.PlayerRelative.SELF).astype(np.float32)
    prev_self_mask = (prev_player_rel == features.PlayerRelative.SELF).astype(np.float32)
    my_hp_now   = (self_mask      * hp_ratio).sum()
    my_hp_prev  = (prev_self_mask * prev_hp_ratio).sum()
    self_hp_delta = my_hp_now - my_hp_prev                  # 掉血為負
    if self_hp_delta < 0:
        reward += R_SELF_HP_LOSS

    # ── 4. 擊殺獎勵（用 killed_value_units 差值判斷）──
    killed_this_step = max(0, score.killed_value_units - prev_score.killed_value_units)
    had_kill = killed_this_step > 0
    reward += R_KILL if had_kill else 0.0

    # ── 5. 我方陣亡懲罰（用螢幕上掠奪者數量差判斷）──
    unit_type     = screen[features.SCREEN_FEATURES.unit_type.index]
    prev_unit_type = prev_screen[features.SCREEN_FEATURES.unit_type.index]
    marauder_now  = int((unit_type      == MARAUDER_ID).sum())
    marauder_prev = int((prev_unit_type == MARAUDER_ID).sum())
    if marauder_now < marauder_prev:
        reward += (marauder_prev - marauder_now) * R_ALLY_DEATH

    # ── 6. 刮痧懲罰（連續 GRAZE_WINDOW 步有掉血但無擊殺）──
    if enemy_hp_delta > 0 and not had_kill:
        graze_counter[0] += 1
    else:
        graze_counter[0] = 0   # 有擊殺或沒有掉血就重置
    if graze_counter[0] >= GRAZE_WINDOW:
        reward += R_GRAZE_PENALTY
        graze_counter[0] = 0   # 觸發後重置，不連續扣

    # ── 7. 終局獎勵（episode 最後一步才計算）──
    if obs.last():
        ey, ex = (player_rel == features.PlayerRelative.ENEMY).nonzero()
        if ex.size == 0:
            reward += R_WIN    # 全殲敵方
        else:
            reward += R_LOSE   # 未能全殲

    return float(np.clip(reward, -2.0, 2.0))


# ─────────────────────────────────────────────
#  視角跟隨工具函式
#
#  修正 #7：改用 player_relative minimap（SELF 區域重心）
#           取代 unit_type minimap（掠奪者常不顯示）。
# ─────────────────────────────────────────────
def get_camera_follow_action(obs):
    """回傳 move_camera 指令，跟隨我方單位重心；找不到時回傳 None。"""
    player_rel_mini = obs.observation.feature_minimap[
        features.MINIMAP_FEATURES.player_relative.index]
    my, mx = (player_rel_mini == features.PlayerRelative.SELF).nonzero()
    if mx.size > 0:
        return actions.FUNCTIONS.move_camera((int(mx.mean()), int(my.mean())))
    return None


# ─────────────────────────────────────────────
#  PPO 戰鬥 Agent
#
#  修正 #1 #2：視角移動與 PPO rollout 完全分離。
#    - 視角移動在 env.step() 之前以獨立呼叫插入，
#      不進入 PPO 的 store() 流程，
#      確保 (obs, action, logp, val, mask) 永遠配對。
# ─────────────────────────────────────────────
class PPOMarauderAgent(base_agent.BaseAgent):

    def __init__(self, start_episode: int = 0):
        super().__init__()
        self.net     = FullyConvNet(NUM_ACTIONS)
        self.trainer = PPOTrainer(self.net)

        # 接續訓練：載入既有模型（若存在）
        self.episode = load_checkpoint(self.net, self.trainer.optimizer)
        if start_episode > 0:          # 外部指定時優先使用
            self.episode = start_episode

        # 累計指標（供 episode 結束時寫 CSV）
        self._last_losses: tuple | None = None  # (pl, vl, tl, en) from last train()
        self._reset_state()

    def _reset_state(self):
        self.prev_obs        = None
        self.prev_obs_tensor = None
        self.prev_act        = None
        self.prev_logp       = None
        self.prev_val        = None
        self.prev_mask       = None
        self.step_count      = 0
        self.episode_reward  = 0.0
        self.graze_counter   = [0]   # 刮痧計數器，用 list 讓 compute_reward 可修改

    def reset(self):
        super().reset()
        self._reset_state()
        print(f"[PPO Agent] Episode {self.episode + 1} 開始")

    # ──────────────────────────────────────────
    #  step()：純 PPO 決策，不處理視角移動
    #  視角移動由主迴圈的 _maybe_move_camera() 負責
    # ──────────────────────────────────────────
    def step(self, obs) -> actions.FunctionCall:
        super().step(obs)
        self.step_count += 1

        obs_tensor = build_obs_tensor(obs)

        player_rel      = obs.observation.feature_screen[features.SCREEN_FEATURES.player_relative.index]
        unit_type       = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
        enemy_exists    = bool((player_rel == features.PlayerRelative.ENEMY).any())
        marauder_exists = bool((unit_type  == MARAUDER_ID).any())

        # ── 存入上一步的 (obs, act, logp, val, mask, reward) ──
        if self.prev_obs is not None:
            reward = compute_reward(obs, self.prev_obs, self.graze_counter)
            self.episode_reward += reward
            done = float(obs.last())
            self.trainer.store(
                obs    = self.prev_obs_tensor,
                action = self.prev_act,
                logp   = self.prev_logp,
                reward = reward,
                value  = self.prev_val,
                done   = done,
                mask   = self.prev_mask,
            )

            if self.trainer.ready():
                with torch.no_grad():
                    _, last_val = self.net(obs_tensor.unsqueeze(0))
                result = self.trainer.train(last_val.item())
                if result is not None:
                    self._last_losses = result   # 保留最後一次 loss 供 episode 結束時寫 CSV

            if obs.last():
                self.episode += 1
                pl, vl, tl, en = self._last_losses if self._last_losses else (0., 0., 0., 0.)
                print(f"[PPO Agent] Episode {self.episode} 結束 | "
                      f"reward={self.episode_reward:.3f}  "
                      f"policy={pl:.4f}  value={vl:.4f}  entropy={en:.4f}")

                # ── CSV 寫入（每個 episode 結束時記錄一行）──
                csv_write(
                    episode     = self.episode,
                    reward      = self.episode_reward,
                    policy_loss = pl,
                    value_loss  = vl,
                    total_loss  = tl,
                    entropy     = en,
                )

                # ── 定期存檔（每 SAVE_EVERY_EP 個 episode）──
                if self.episode % SAVE_EVERY_EP == 0:
                    save_checkpoint(self.net, self.trainer.optimizer, self.episode)

                self._last_losses = None

        # ── 建立遮罩 & 選動作 ──
        mask = build_action_mask(obs, enemy_exists, marauder_exists)

        with torch.no_grad():
            logits, value = self.net(obs_tensor.unsqueeze(0))
            masked_logits = logits + (1 - mask.unsqueeze(0)) * (-1e9)
            dist    = Categorical(logits=masked_logits)
            action  = dist.sample()
            logp    = dist.log_prob(action)

        action_id  = action.item()
        sc2_action = discrete_to_sc2(action_id, obs)

        # ── 記錄本步（obs/act/logp/val/mask 全部對齊） ──
        self.prev_obs        = obs
        self.prev_obs_tensor = obs_tensor
        self.prev_act        = action_id
        self.prev_logp       = logp.item()
        self.prev_val        = value.item()
        self.prev_mask       = mask

        return sc2_action


# ─────────────────────────────────────────────
#  人族建造腳本
#
#  修正 #6：改用 count_units(MARAUDER_ID) >= 5
#           判斷掠奪者是否實際完成，取代下令次數計數。
# ─────────────────────────────────────────────
class TerranBot(base_agent.BaseAgent):

    def __init__(self):
        super().__init__()
        self.state = -1
        self.supply_depot_target   = None
        self.refinery_target       = None
        self.barracks_target       = None
        self.gas_workers_assigned  = 0
        self.selecting_worker      = True
        self.recent_selected_coords = []
        self.busy_depot_locations  = []
        self.depots_built          = 0
        self.last_depot_pixels     = 0
        self.select_attempts       = 0
        self.initial_mineral_coords = None
        self.first_depot_coords    = None
        self.build_dir             = (1, 1)
        self.base_minimap_coords   = None
        self.cc_x_screen           = 0
        self.cc_y_screen           = 0
        self.camera_centered       = False
        self.is_first_step         = True
        self.handoff               = False

    def step(self, obs):
        super().step(obs)

        unit_type_screen   = obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
        current_minerals   = obs.observation.player.minerals
        current_vespene    = obs.observation.player.vespene
        current_workers    = obs.observation.player.food_workers

        def count_units(uid):
            return int(np.sum(unit_type_screen == uid))

        def get_select_cc_action():
            y_c, x_c = (unit_type_screen == COMMAND_CENTER_ID).nonzero()
            if x_c.size:
                self.cc_x_screen = int(x_c.mean())
                self.cc_y_screen = int(y_c.mean())
                return actions.FUNCTIONS.select_point("select", (self.cc_x_screen, self.cc_y_screen))
            return NO_OP

        def get_select_scv_action(select_type="select"):
            y_c, x_c = (unit_type_screen == SCV_ID).nonzero()
            if x_c.size:
                mask = np.ones(len(x_c), dtype=bool)
                for bx, by in self.busy_depot_locations:
                    mask &= np.sqrt((x_c - bx)**2 + (y_c - by)**2) > 10
                if self.supply_depot_target:
                    mask &= np.sqrt((x_c - self.supply_depot_target[0])**2 +
                                    (y_c - self.supply_depot_target[1])**2) > 10
                if self.refinery_target:
                    mask &= np.sqrt((x_c - self.refinery_target[0])**2 +
                                    (y_c - self.refinery_target[1])**2) > 15
                pool = np.where(mask)[0] if mask.any() else np.arange(len(x_c))
                idx  = random.choice(pool)
                return actions.FUNCTIONS.select_point(select_type, (int(x_c[idx]), int(y_c[idx])))
            return NO_OP

        def get_select_barracks_action():
            y_c, x_c = (unit_type_screen == BARRACKS_ID).nonzero()
            if x_c.size:
                return actions.FUNCTIONS.select_point("select", (int(x_c.mean()), int(y_c.mean())))
            return NO_OP

        # ── 初始化 ──
        if self.is_first_step:
            self.is_first_step = False
            min_c = (unit_type_screen == MINERAL_FIELD_ID).nonzero()
            if min_c[0].size:
                self.initial_mineral_coords = (int(min_c[1][0]), int(min_c[0][0]))
            cc_c = (unit_type_screen == COMMAND_CENTER_ID).nonzero()
            if cc_c[0].size:
                self.cc_x_screen = int(cc_c[1].mean())
                self.cc_y_screen = int(cc_c[0].mean())
            pmini = obs.observation.feature_minimap[features.MINIMAP_FEATURES.player_relative.index]
            ym, xm = (pmini == features.PlayerRelative.SELF).nonzero()
            if xm.size:
                self.base_minimap_coords = (int(xm.mean()), int(ym.mean()))
            self.last_depot_pixels = count_units(SUPPLY_DEPOT_ID)

        # ── 狀態機 ──
        if self.state == -1:
            if HARVEST_ACTION in obs.observation.available_actions and self.initial_mineral_coords:
                self.state = 0
                return actions.FUNCTIONS.Harvest_Gather_screen("now", self.initial_mineral_coords)
            return get_select_scv_action("select_all_type")

        elif self.state == 0:
            if current_workers < 15:
                if TRAIN_SCV_ACTION in obs.observation.available_actions and current_minerals >= 50:
                    return actions.FUNCTIONS.Train_SCV_quick("now")
                elif current_minerals >= 50:
                    return get_select_cc_action()
            elif current_minerals >= 100:
                self.state = 1
                offset_x = 15 if self.initial_mineral_coords[0] < self.cc_x_screen else -15
                offset_y = 15 if self.initial_mineral_coords[1] < self.cc_y_screen else -15
                self.supply_depot_target = (
                    int(np.clip(self.cc_x_screen + offset_x, 0, 83)),
                    int(np.clip(self.cc_y_screen + offset_y, 0, 83)))
                self.first_depot_coords = self.supply_depot_target
                self.build_dir = (1 if offset_x > 0 else -1, 1 if offset_y > 0 else -1)
                print(f"[Terran] 準備建造第 1 個 Supply Depot: {self.supply_depot_target}")
                return get_select_scv_action()
            return NO_OP

        elif self.state == 1:
            if BUILD_SUPPLYDEPOT_ACTION in obs.observation.available_actions:
                if current_minerals >= 100:
                    self.state = 1.1
                    self.select_attempts = 0
                    return actions.FUNCTIONS.Build_SupplyDepot_screen("now", self.supply_depot_target)
            elif current_minerals >= 100:
                return get_select_scv_action()
            return NO_OP

        elif self.state == 1.1:
            current_pixels = count_units(SUPPLY_DEPOT_ID)
            if current_pixels > self.last_depot_pixels + 5:
                self.depots_built += 1
                self.last_depot_pixels = current_pixels
                self.busy_depot_locations.append(self.supply_depot_target)
                print(f"[Terran] 第 {self.depots_built} 個 Supply Depot 開始建造")
                if self.depots_built < 3:
                    dir_x, dir_y = self.build_dir
                    bx, by = self.first_depot_coords
                    if self.depots_built == 1:
                        nx, ny = bx + 12 * dir_x, by
                    else:
                        nx, ny = bx + 6 * dir_x, by + 12 * dir_y
                    self.supply_depot_target = (int(np.clip(nx, 0, 83)), int(np.clip(ny, 0, 83)))
                    self.state = 1
                    return get_select_scv_action()
                else:
                    self.state = 2
            self.select_attempts += 1
            if self.select_attempts > 100:
                ox, oy = self.supply_depot_target
                self.supply_depot_target = (
                    int(np.clip(ox + random.randint(-15, 15), 0, 83)),
                    int(np.clip(oy + random.randint(-15, 15), 0, 83)))
                self.state = 1
                return get_select_scv_action()
            return NO_OP

        elif self.state == 2:
            if BUILD_REFINERY_ACTION in obs.observation.available_actions:
                if current_minerals >= 75:
                    if self.refinery_target is None:
                        yg, xg = (unit_type_screen == GEYSER_ID).nonzero()
                        if xg.size:
                            m = (np.abs(xg - xg[0]) < 10) & (np.abs(yg - yg[0]) < 10)
                            self.refinery_target = (int(xg[m].mean()), int(yg[m].mean()))
                        else:
                            return NO_OP
                    self.state = 2.1
                    return actions.FUNCTIONS.Build_Refinery_screen("now", self.refinery_target)
            else:
                return get_select_scv_action()

        elif self.state == 2.1:
            if count_units(REFINERY_ID) > 0:
                self.state = 3
            return NO_OP

        elif self.state == 3:
            if self.gas_workers_assigned < 3:
                if self.selecting_worker:
                    yc, xc = (unit_type_screen == SCV_ID).nonzero()
                    if xc.size:
                        dist = np.sqrt((xc - self.refinery_target[0])**2 +
                                       (yc - self.refinery_target[1])**2)
                        mask = dist > 15
                        for px, py in self.recent_selected_coords:
                            mask &= np.sqrt((xc - px)**2 + (yc - py)**2) > 5
                        if mask.any():
                            idx = random.choice(np.where(mask)[0])
                            tx, ty = int(xc[idx]), int(yc[idx])
                            self.recent_selected_coords.append((tx, ty))
                            self.selecting_worker = False
                            return actions.FUNCTIONS.select_point("select", (tx, ty))
                    return NO_OP
                else:
                    if HARVEST_ACTION in obs.observation.available_actions:
                        self.gas_workers_assigned += 1
                        self.selecting_worker = True
                        return actions.FUNCTIONS.Harvest_Gather_screen("now", self.refinery_target)
                    else:
                        self.selecting_worker = True
            else:
                self.state = 4
            return NO_OP

        elif self.state == 4:
            if not self.camera_centered and self.base_minimap_coords:
                self.camera_centered = True
                return actions.FUNCTIONS.move_camera(self.base_minimap_coords)
            if BUILD_BARRACKS_ACTION in obs.observation.available_actions:
                if current_minerals >= 150:
                    if self.barracks_target is None:
                        offset_x = 30 if self.base_minimap_coords[0] <= 32 else -30
                        self.barracks_target = (int(np.clip(42 + offset_x, 0, 83)), 42)
                    self.state = 4.1
                    return actions.FUNCTIONS.Build_Barracks_screen("now", self.barracks_target)
            else:
                return get_select_scv_action()

        elif self.state == 4.1:
            if count_units(BARRACKS_ID) > 0:
                self.state = 5
            return NO_OP

        elif self.state == 5:
            if count_units(BARRACKS_TECHLAB_ID) > 0:
                self.state = 7
                return NO_OP
            if BUILD_TECHLAB_ACTION in obs.observation.available_actions:
                if current_minerals >= 50 and current_vespene >= 25:
                    self.state = 6
                    return actions.FUNCTIONS.Build_TechLab_quick("now")
            else:
                return get_select_barracks_action()

        elif self.state == 6:
            if count_units(BARRACKS_TECHLAB_ID) > 0:
                self.state = 7
            return NO_OP

        elif self.state == 7:
            actual_marauders = count_units(MARAUDER_ID)
            if actual_marauders < 5:
                # ── 人口上限檢查：掠奪者佔 2 人口，若剩餘不足則先等待 ──
                food_used = obs.observation.player.food_used
                food_cap  = obs.observation.player.food_cap
                food_left = food_cap - food_used
                if food_left < 2:
                    # 人口不足，等待（補給站建造中或已滿）
                    return NO_OP

                if TRAIN_MARAUDER_ACTION in obs.observation.available_actions:
                    if current_minerals >= 100 and current_vespene >= 25:
                        print(f"[Terran] 訓練掠奪者（目前 {actual_marauders}/5）")
                        return actions.FUNCTIONS.Train_Marauder_quick("now")
                    # 資源不足，等待
                    return NO_OP
                else:
                    # 兵營未被選取，重新點選
                    return get_select_barracks_action()
            else:
                self.state = 8
                self.handoff = True
                print("[Terran] 建造完成！切換至 PPO 接管")
            return NO_OP

        return NO_OP


# ─────────────────────────────────────────────
#  主迴圈
#
#  修正 #1 #2：視角移動改由主迴圈獨立呼叫 env.step()，
#              完全不進入 PPO agent 的 step()，
#              確保 rollout buffer 資料永遠對齊。
# ─────────────────────────────────────────────
def main(unused_argv):
    # SC2 安裝路徑（非預設路徑時必須指定）
    os.environ["SC2PATH"] = r"D:\Game\StarCraft II"

    # CSV 初始化（第一次執行時建立標頭）
    csv_init()

    build_agent = TerranBot()
    ppo_agent   = PPOMarauderAgent()   # __init__ 內自動 load_checkpoint
    active      = build_agent
    ppo_step_counter = 0

    try:
        with sc2_env.SC2Env(
            map_name="Simple96",
            players=[
                sc2_env.Agent(sc2_env.Race.terran),
                sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy),
            ],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=SCREEN_SIZE, minimap=96),
                use_feature_units=True,
            ),
            step_mul=8,
            game_steps_per_episode=50000,  # 約 3 分鐘上限（50000 / 22.4 / 8 ≈ 279 秒）
            visualize=False,
            realtime=False,
            # 黑屏問題請修改 SC2 設定檔：
            # C:\Users\<你的使用者名>\Documents\StarCraft II\Variables.txt
            # 加入：displaymode=0 / windowwidth=1024 / windowheight=768
        ) as env:

            build_agent.setup(env.observation_spec(), env.action_spec())
            ppo_agent.setup(env.observation_spec(), env.action_spec())

            timesteps = env.reset()
            build_agent.reset()

            # SC2 啟動後稍等 2 秒讓渲染初始化，再強制拉到前景觸發重繪
            time.sleep(2)
            _focus_sc2_window()

            while True:
                # ── 視角跟隨（PPO 階段，每 CAMERA_INTERVAL 步執行一次）──
                # 獨立於 PPO step()，用獨立的 env.step() 送出，不汙染 rollout
                if active is ppo_agent:
                    ppo_step_counter += 1
                    if ppo_step_counter % CAMERA_INTERVAL == 0:
                        cam_action = get_camera_follow_action(timesteps[0])
                        if cam_action is not None:
                            timesteps = env.step([cam_action])
                            if timesteps[0].last():
                                break
                            continue  # 本輪只做視角移動，直接進下一輪

                # ── 主動作（建造腳本 or PPO）──
                step_action = [active.step(timesteps[0])]

                # 建造完成時切換
                if active is build_agent and build_agent.handoff:
                    print("[Main] 切換至 PPO Marauder Agent！")
                    active = ppo_agent
                    ppo_agent.reset()
                    ppo_step_counter = 0

                if timesteps[0].last():
                    break

                timesteps = env.step(step_action)

    except KeyboardInterrupt:
        print("[Main] 手動中斷，儲存模型...")
        save_checkpoint(ppo_agent.net, ppo_agent.trainer.optimizer, ppo_agent.episode)
        print(f"[Main] 儲存完成，下次執行將從 episode {ppo_agent.episode + 1} 繼續")


if __name__ == "__main__":
    app.run(main)