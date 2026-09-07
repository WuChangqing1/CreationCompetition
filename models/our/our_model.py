import torch
import os
import json
from collections import OrderedDict
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.lstm import LSTMEncoder
from models.networks.classifier import FcClassifier
from models.utils.config import OptConfig
import torch.nn as nn

class FocalLoss(nn.Module):
    """
    自适应 Focal Loss，聚焦于难分类样本 (Hard Samples)
    """
    def __init__(self, gamma=3.0, alpha=torch.tensor([1,8])):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()

class ModalityAdapter(nn.Module):
    """
    跨模态适配器：将序列压缩为固定长度的摘要向量。
    修复版本：使用 adaptive_avg_pool1d 调整长度，避免维度错误。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(out_channels // 2, 1, kernel_size=1)

    def forward(self, x, target_len=None):
        # x: [batch, seq_len, dim] -> [batch, dim, seq_len]
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)                         # [batch, 1, seq_len]
        if target_len is not None and target_len != x.size(2):
            x = F.adaptive_avg_pool1d(x, target_len)  # [batch, 1, target_len]
        x = x.squeeze(1)                          # [batch, target_len]
        return x


class LightweightVEM(nn.Module):
    """
    轻量级视觉增强模块 (VEM)：用多头自注意力强化视觉特征，
    支持 key_padding_mask 忽略无效帧。
    """
    def __init__(self, d_model, nhead=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, key_padding_mask=None):
        # src: [batch, seq_len, d_model]
        # key_padding_mask: [batch, seq_len] (True 表示 padding 位置)
        attn_mask = None
        if key_padding_mask is not None:
            # 将 mask 转为 float 类型，用于 attention mask（可选）
            attn_mask = key_padding_mask.float().masked_fill(key_padding_mask, float('-inf'))
        src2, _ = self.self_attn(src, src, src, key_padding_mask=key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class BiCFNetCrossFeedbackModule(nn.Module):
    """
    Bi-CFNet 跨模态反馈机制：双向门控增强。
    """
    def __init__(self, d_model):
        super().__init__()
        self.audio_adapter = ModalityAdapter(d_model, d_model)
        self.visual_adapter = ModalityAdapter(d_model, d_model)
        self.gate_a = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.gate_v = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())

    def forward(self, audio_feat, visual_feat):
        batch_size, seq_len_a, dim = audio_feat.size()
        _, seq_len_v, _ = visual_feat.size()

        # 视觉反馈给音频
        v_summarized = self.visual_adapter(visual_feat, target_len=seq_len_a)  # [B, seq_len_a]
        v_feedback = v_summarized.unsqueeze(-1).expand(-1, -1, dim)            # [B, seq_len_a, dim]
        gate_a_val = self.gate_a(torch.cat([audio_feat, v_feedback], dim=-1))
        audio_enhanced = audio_feat + gate_a_val * v_feedback

        # 音频反馈给视觉
        a_summarized = self.audio_adapter(audio_feat, target_len=seq_len_v)    # [B, seq_len_v]
        a_feedback = a_summarized.unsqueeze(-1).expand(-1, -1, dim)            # [B, seq_len_v, dim]
        gate_v_val = self.gate_v(torch.cat([visual_feat, a_feedback], dim=-1))
        visual_enhanced = visual_feat + gate_v_val * a_feedback

        return audio_enhanced, visual_enhanced


class ourModel(BaseModel, nn.Module):
    def __init__(self, opt):
        super().__init__(opt)
        nn.Module.__init__(self)

        self.loss_names = []
        self.model_names = []

        # 声学编码器
        self.netEmoA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
        self.model_names.append('EmoA')

        # 视觉编码器
        self.netEmoV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
        self.model_names.append('EmoV')

        # 投影层
        self.netProjA = nn.Linear(opt.embd_size_a, opt.hidden_size)
        self.netProjV = nn.Linear(opt.embd_size_v, opt.hidden_size)
        self.model_names.append('ProjA')
        self.model_names.append('ProjV')

        # 轻量级视觉增强模块 (VEM)
        self.netVEM = LightweightVEM(
            d_model=opt.hidden_size,
            nhead=getattr(opt, 'vem_nhead', 4),
            dim_feedforward=getattr(opt, 'vem_ffn_dim', 512),
            dropout=getattr(opt, 'dropout_rate', 0.1)
        )
        self.model_names.append('VEM')

        # 跨模态反馈模块 (CFM)
        self.netCFM = BiCFNetCrossFeedbackModule(d_model=opt.hidden_size)
        self.model_names.append('CFM')

        # 个性化投影
        self.netProjP = nn.Linear(1024, opt.hidden_size)
        self.model_names.append('ProjP')

        # 分类器
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        cls_input_size = opt.hidden_size * 3  # audio + visual + personalized

        self.netEmoC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoC')
        self.loss_names.append('emo_CE')

        self.netEmoCF = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoCF')
        self.loss_names.append('EmoF_CE')

        # 损失函数：替换 criterion_focal 为我们定义的 FocalLoss
        self.criterion_ce = torch.nn.CrossEntropyLoss()
        self.criterion_focal = FocalLoss(gamma=3.0,alpha=torch.tensor([1,8]))

        if self.isTrain:
            parameters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            optimizer_name = getattr(opt, 'optimizer', 'adam').lower()
            weight_decay = float(getattr(opt, 'weight_decay', 1e-3))
            if optimizer_name == 'adamw':
                self.optimizer = torch.optim.AdamW(parameters, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=weight_decay)
            else:
                self.optimizer = torch.optim.Adam(parameters, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=weight_decay)
            self.optimizers.append(self.optimizer)

            self.ce_weight = getattr(opt, 'ce_weight', 1.0)
            self.focal_weight = getattr(opt, 'focal_weight', 1.0)

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if self.isTrain:
            os.makedirs(self.save_dir, exist_ok=True)

    def post_process(self):
        """加载预训练权重（如有）"""
        def transform_key_for_parallel(state_dict):
            return OrderedDict([('module.' + key, value) for key, value in state_dict.items()])

        if self.isTrain and hasattr(self, 'pretrained_encoder'):
            print('[ Init ] Load parameters from pretrained encoder network')
            f = lambda x: transform_key_for_parallel(x)
            # 只加载存在的组件，避免缺失 VEM/CFM 时崩溃
            for name in ['EmoA', 'EmoV', 'ProjA', 'ProjV', 'CFM', 'VEM', 'ProjP']:
                if hasattr(self.pretrained_encoder, f'net{name}'):
                    try:
                        getattr(self, f'net{name}').load_state_dict(
                            f(getattr(self.pretrained_encoder, f'net{name}').state_dict()), strict=False
                        )
                    except Exception as e:
                        print(f'Warning: failed to load {name}: {e}')

    def set_input(self, input):
        # 从模型参数动态获取设备，避免 self.device 不同步
        device = next(self.parameters()).device
        self.mask_a = (input['A_feat'].abs().sum(dim=-1) > 0).to(device)
        self.mask_v = (input['V_feat'].abs().sum(dim=-1) > 0).to(device)
        self.acoustic = input['A_feat'].float().to(device)
        self.visual = input['V_feat'].float().to(device)
        self.emo_label = input['emo_label'].to(device)

        self.personalized = (
            input['personalized_feat'].float().to(device)
            if getattr(self.opt, 'use_personality', True) and 'personalized_feat' in input
            else None
        )

    def forward(self, acoustic_feat=None, visual_feat=None):
        # 支持外部传入特征
        if acoustic_feat is not None:
            device = next(self.parameters()).device
            self.acoustic = acoustic_feat.float().to(device)
            self.visual = visual_feat.float().to(device)
            self.mask_a = (acoustic_feat.abs().sum(dim=-1) > 0).to(device)
            self.mask_v = (visual_feat.abs().sum(dim=-1) > 0).to(device)

        # 1. LSTM 编码
        emo_feat_A = self.netEmoA(self.acoustic)       # [B, L_a, embd_a]
        emo_feat_V = self.netEmoV(self.visual)         # [B, L_v, embd_v]

        # 2. 投影到统一维度 d_model
        emo_feat_A = torch.tanh(self.netProjA(emo_feat_A))  # [B, L_a, d_model]
        emo_feat_V = torch.tanh(self.netProjV(emo_feat_V))  # [B, L_v, d_model]

        # 3. 视觉增强（自注意力 + FFN），传入 mask 忽略 padding
        emo_feat_V = self.netVEM(emo_feat_V, key_padding_mask=~self.mask_v)

        # 4. 跨模态反馈 (CFM)：双向交互
        emo_feat_A, emo_feat_V = self.netCFM(emo_feat_A, emo_feat_V)

        # 5. 掩码均值池化 -> 固定维度向量
        pooled_a = self.masked_avg_pool(emo_feat_A, self.mask_a)  # [B, d_model]
        pooled_v = self.masked_avg_pool(emo_feat_V, self.mask_v)  # [B, d_model]
        emo_fusion_feat = torch.cat((pooled_a, pooled_v), dim=-1) # [B, 2*d_model]

        # 6. 拼接个性化特征
        if self.personalized is not None:
            personalized_proj = self.netProjP(self.personalized)   # [B, d_model]
        else:
            personalized_proj = torch.zeros(
                (emo_fusion_feat.size(0), self.netProjP.out_features),
                device=emo_fusion_feat.device, dtype=emo_fusion_feat.dtype
            )
        emo_fusion_feat = torch.cat((emo_fusion_feat, personalized_proj), dim=-1)  # [B, 3*d_model]

        # 7. 双分类器输出
        self.emo_logits_fusion, _ = self.netEmoCF(emo_fusion_feat)  # 辅助分类器
        self.emo_logits, _ = self.netEmoC(emo_fusion_feat)          # 主分类器
        self.emo_pred = F.softmax(self.emo_logits, dim=-1)

    def backward(self):
        # 交叉熵 + 辅助损失（Focal Loss）
        self.loss_emo_CE = self.ce_weight * self.criterion_ce(self.emo_logits, self.emo_label)
        self.loss_EmoF_CE = self.focal_weight * self.criterion_focal(self.emo_logits_fusion, self.emo_label)
        loss = self.loss_emo_CE + self.loss_EmoF_CE
        loss.backward()

        # 梯度裁剪
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)

    def optimize_parameters(self, epoch):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    @staticmethod
    def masked_avg_pool(features, mask):
        """对有效帧做平均池化"""
        mask = mask.unsqueeze(-1).type_as(features)  # [B, L, 1]
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (features * mask).sum(dim=1) / denom
