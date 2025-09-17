
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MemoryImagineDecoder(nn.Module):
    """Memory-Imagine 4D时空解码器"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 特征融合层（参考VLM3R cross-attention）
        self.feature_fusion = self._build_feature_fusion()
        
        # 时序建模
        self.time_modeling = self._build_time_modeling()
        
        # 4D输出头
        self.output_head = self._build_output_head()
    
    def _build_feature_fusion(self):
        """构建特征融合层"""
        return nn.ModuleDict({
            'paligemma_proj': nn.Linear(
                self.config.paligemma_feature_dim, 
                self.config.hidden_dim
            ),
            'state_proj': nn.Linear(
                self.config.robot_state_dim,
                self.config.hidden_dim
            ),
            'cross_attention': nn.MultiheadAttention(
                embed_dim=self.config.hidden_dim,
                num_heads=self.config.num_attention_heads,
                dropout=self.config.dropout,
                batch_first=True
            ),
            'fusion_norm': nn.LayerNorm(self.config.hidden_dim),
            'fusion_mlp': nn.Sequential(
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim)
            )
        })
    
    def _build_time_modeling(self):
        """构建时序建模层"""
        return nn.ModuleDict({
            'time_embed': nn.Embedding(
                self.config.total_time_frames,
                self.config.hidden_dim
            ),
            'pos_embed': nn.Parameter(
                torch.randn(1, self.config.total_time_frames, self.config.hidden_dim) * 0.02
            ),
            'transformer': nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.config.hidden_dim,
                    nhead=self.config.num_attention_heads,
                    dim_feedforward=self.config.hidden_dim * 4,
                    dropout=self.config.dropout,
                    batch_first=True,
                    norm_first=True
                ),
                num_layers=self.config.num_transformer_layers
            )
        })
    
    def _build_output_head(self):
        """构建输出头"""
        return nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim * 2),
            nn.LayerNorm(self.config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(
                self.config.hidden_dim * 2,
                self.config.total_spatial_dim
            )
        )
    
    def forward(self, paligemma_features, robot_state):
        """
        前向传播
        
        Args:
            paligemma_features: (B, L, 2048) PaliGemma特征
            robot_state: (B, 6) 机器人状态
            
        Returns:
            occupancy_4d: (B, T, X, Y, Z, C) 4D语义占用
        """
        B = paligemma_features.shape[0]
        
        # 1. 特征融合
        fused_features = self.fuse_features(paligemma_features, robot_state)
        
        # 2. 时序建模
        temporal_features = self.model_temporal_sequence(fused_features)
        
        # 3. 4D输出生成
        occupancy_4d = self.generate_4d_output(temporal_features)
        
        return occupancy_4d
    
    def fuse_features(self, paligemma_features, robot_state):
        """特征融合（参考VLM3R）"""
        # 投影到统一维度
        pali_proj = self.feature_fusion['paligemma_proj'](
            paligemma_features.mean(dim=1)  # 聚合序列特征
        )  # (B, hidden_dim)
        
        state_proj = self.feature_fusion['state_proj'](robot_state)  # (B, hidden_dim)
        
        # Cross-attention融合
        pali_proj = pali_proj.unsqueeze(1)  # (B, 1, hidden_dim) 
        state_proj = state_proj.unsqueeze(1)  # (B, 1, hidden_dim)
        
        fused, _ = self.feature_fusion['cross_attention'](
            query=pali_proj,
            key=state_proj,
            value=state_proj
        )
        
        # 残差连接和归一化
        fused = fused + pali_proj
        fused = self.feature_fusion['fusion_norm'](fused)
        
        # MLP处理
        fused = fused + self.feature_fusion['fusion_mlp'](fused)
        
        return fused.squeeze(1)  # (B, hidden_dim)
    
    def model_temporal_sequence(self, fused_features):
        """时序建模"""
        B = fused_features.shape[0]
        
        # 时间编码
        time_indices = torch.arange(
            self.config.total_time_frames,
            device=fused_features.device
        )
        time_emb = self.time_modeling['time_embed'](time_indices)  # (T, hidden_dim)
        
        # 构建时序序列
        sequence = fused_features.unsqueeze(1) + time_emb.unsqueeze(0)  # (B, T, hidden_dim)
        sequence = sequence + self.time_modeling['pos_embed']
        
        # Transformer处理
        temporal_features = self.time_modeling['transformer'](sequence)  # (B, T, hidden_dim)
        
        return temporal_features
    
    def generate_4d_output(self, temporal_features):
        """生成4D输出"""
        B, T = temporal_features.shape[:2]
        
        # 对每个时刻生成空间占用
        outputs = []
        for t in range(T):
            time_feature = temporal_features[:, t]  # (B, hidden_dim)
            spatial_output = self.output_head(time_feature)  # (B, spatial_total)
            outputs.append(spatial_output)
        
        # 堆叠并重塑
        output_tensor = torch.stack(outputs, dim=1)  # (B, T, spatial_total)
        
        occupancy_4d = output_tensor.view(
            B, T,
            self.config.spatial_x, self.config.spatial_y, self.config.spatial_z,
            self.config.semantic_classes
        )
        
        return occupancy_4d
