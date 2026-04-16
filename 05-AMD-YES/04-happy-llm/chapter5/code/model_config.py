"""
LLaMA2 模型配置文件

本文件定义了 LLaMA2 模型的超参数和配置类。
用户可以根据硬件和数据集大小调整这些参数。
"""

from dataclasses import dataclass
from transformers import PretrainedConfig


@dataclass
class ModelConfig(PretrainedConfig):
    """
    LLaMA2 模型配置类
    
    继承自 PretrainedConfig，支持与 Hugging Face 生态集成
    """
    
    model_type = "llama"
    
    def __init__(
        self,
        dim: int = 4096,              # 模型隐藏层维度
        n_layers: int = 32,           # Transformer 层数
        n_heads: int = 32,            # 多头注意力头数
        n_kv_heads: int = 8,          # 键值头数量
        vocab_size: int = 32000,      # 词汇表大小
        hidden_dim: int = None,       # 前馈层隐藏维度（默认 dim*4）
        multiple_of: int = 256,       # 维度对齐
        norm_eps: float = 1e-5,       # 归一化 eps
        max_seq_len: int = 2048,      # 最大序列长度
        dropout: float = 0.0,         # Dropout 概率
        flash_attn: bool = True,      # 是否使用 FlashAttention
        use_cache: bool = True,       # 是否使用 KV Cache
        **kwargs,
    ):
        """
        初始化模型配置参数
        
        Args:
            dim: 模型的隐藏层维度，通常是 256 的倍数
            n_layers: Transformer 堆叠的层数
            n_heads: 多头注意力的头数
            n_kv_heads: 键值注意力的头数（支持 GQA）
            vocab_size: 分词器的词汇表大小
            hidden_dim: FFN 层的隐藏维度，默认为 dim * 4 的倍数
            multiple_of: 维度对齐值，确保计算效率
            norm_eps: LayerNorm / RMSNorm 的数值稳定性参数
            max_seq_len: 最大输入序列长度
            dropout: Dropout 概率，用于正则化
            flash_attn: 是否使用 FlashAttention 加速
            use_cache: 推理时是否使用 KV Cache 加速
        """
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim if hidden_dim is not None else int(8 * dim / 3)
        # 向上对齐到 multiple_of 的倍数
        self.hidden_dim = ((self.hidden_dim + multiple_of - 1) // multiple_of) * multiple_of
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        self.use_cache = use_cache
        
        super().__init__(**kwargs)
    
    @property
    def head_dim(self) -> int:
        """单个注意力头的维度"""
        return self.dim // self.n_heads


# 常见模型大小预设
MODEL_CONFIGS = {
    "tiny": ModelConfig(
        dim=256,
        n_layers=8,
        n_heads=8,
        vocab_size=6144,
        max_seq_len=512,
    ),
    "small": ModelConfig(
        dim=512,
        n_layers=12,
        n_heads=8,
        vocab_size=6144,
        max_seq_len=1024,
    ),
    "base": ModelConfig(
        dim=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=6,
        vocab_size=32000,
        max_seq_len=2048,
    ),
    "large": ModelConfig(
        dim=1024,
        n_layers=24,
        n_heads=16,
        n_kv_heads=8,
        vocab_size=32000,
        max_seq_len=2048,
    ),
    "7b": ModelConfig(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        vocab_size=32000,
        max_seq_len=2048,
    ),
    "13b": ModelConfig(
        dim=5120,
        n_layers=40,
        n_heads=40,
        n_kv_heads=10,
        vocab_size=32000,
        max_seq_len=2048,
    ),
}


def get_model_config(name: str) -> ModelConfig:
    """根据名称获取预定义的模型配置"""
    if name in MODEL_CONFIGS:
        return MODEL_CONFIGS[name]
    else:
        raise ValueError(f"未知的模型配置: {name}。可用配置: {list(MODEL_CONFIGS.keys())}")


if __name__ == "__main__":
    # 测试配置
    print("=== 模型配置示例 ===")
    for name, config in MODEL_CONFIGS.items():
        total_params = (
            config.vocab_size * config.dim +  # embedding
            config.n_layers * (
                config.dim * config.dim * 3 +  # attention
                config.dim * config.hidden_dim * 2  # ffn
            )
        )
        print(f"{name}: {total_params / 1e9:.2f}B 参数")
