"""
config.py - 系统配置文件
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch


@dataclass
class ModelConfig:
    """模型配置"""
    # 模型基础配置
    model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct"
    cache_dir: str = "./models/transformers_cache"

    # 添加本地模型路径配置
    local_model_path: Optional[str] = "models/transformers_cache/models--Qwen--Qwen2.5-3B-Instruct"
    use_local: bool = True

    # 设备和精度配置
    device: Optional[str] = "cuda"
    dtype: Optional[str] = None

    # 量化配置
    load_in_4bit: bool = False
    load_in_8bit: bool = True

    # 模型加载配置
    trust_remote_code: bool = True
    attn_implementation: Optional[str] = None
    use_cache = False
    # 其他配置
    verbose: bool = False

    def get_torch_dtype(self):
        """将字符串dtype转换为torch.dtype"""
        if self.dtype is None:
            return None
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return dtype_map.get(self.dtype.lower(), None)


@dataclass
class TrainingConfig:
    """训练配置"""
    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_steps: int = 50
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 1
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    seed: int = 42

    # 添加训练相关的额外配置
    max_grad_norm: float = 1.0
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = True
    label_smoothing_factor: float = 0.0
    report_to: List[str] = field(default_factory=lambda: ["swanlab"])
    ddp_find_unused_parameters: bool = False


@dataclass
class LoRAConfig:
    """LoRA配置 - 针对Qwen2.5模型优化"""
    r: int = 32
    lora_alpha: int = 64
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    modules_to_save: Optional[List[str]] = None


@dataclass
class RAGConfig:
    """RAG配置"""
    # 嵌入模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cuda"
    embedding_batch_size: int = 32

    # 向量数据库配置
    vector_dim: int = 512
    index_type: str = "IVF"
    nprobe: int = 10
    nlist: int = 100

    # 文档处理配置
    chunk_size: int = 500
    chunk_overlap: int = 100
    separator: str = "\n"

    # 检索配置
    top_k: int = 5
    similarity_threshold: float = 0.7
    use_rerank: bool = True
    rerank_model: Optional[str] = "BAAI/bge-reranker-base"

    # 存储配置
    vector_store_path: str = "./data/vector_stores"
    persist_directory: str = "./data/chroma_db"


@dataclass
class GenerationConfig:
    """生成配置"""
    # 基础生成参数
    max_new_tokens: int = 1024
    min_new_tokens: int = 10

    # 采样参数
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

    # 控制参数
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3

    # 束搜索参数
    num_beams: int = 1
    num_beam_groups: int = 1
    diversity_penalty: float = 0.0
    early_stopping: bool = False

    # 停止条件
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    max_length: Optional[int] = None

    # 流式输出
    use_streaming: bool = False


@dataclass
class DataConfig:
    """数据配置"""
    # 数据目录
    novel_data_dir: str = "./data/novels"
    dataset_dir: str = "./data/datasets"
    knowledge_base_dir: str = "./data/knowledge_bases"
    processed_data_dir: str = "./data/processed"
    cache_dir: str = "./data/cache"

    # 数据处理参数
    max_samples_per_novel: int = 1000
    max_length: int = 2048
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # 数据增强
    use_data_augmentation: bool = True
    augmentation_factor: int = 2
    data_augment_ratio: float = 0.2  # 从PostTrainingConfig移过来

    # 数据格式
    input_format: str = "text"
    output_format: str = "json"

    # 上下文配置（从PostTrainingConfig移过来）
    use_context: bool = True
    context_length: int = 200


@dataclass
class PostTrainingConfig:
    """后训练配置"""
    # 基础配置
    experiment_name: str = ""

    # SFT配置
    sft_enabled: bool = True
    sft_epochs: int = 3
    sft_batch_size: int = 1
    sft_learning_rate: float = 2e-5
    sft_use_lora: bool = True
    sft_lora_r: int = 32
    sft_lora_alpha: int = 64

    # DPO配置
    dpo_enabled: bool = True
    dpo_epochs: int = 2
    dpo_batch_size: int = 1
    dpo_learning_rate: float = 1e-6
    dpo_beta: float = 0.1
    dpo_num_candidates: int = 2

    # 评估配置
    eval_enabled: bool = True
    eval_test_size: int = 100
    eval_metrics: List[str] = field(default_factory=lambda: ["perplexity", "bleu", "diversity", "style"])

    # 实验配置
    max_samples_per_style: int = 1000
    parallel_workers: int = 4


@dataclass
class SystemConfig:
    """系统总配置"""
    # 项目信息
    project_name: str = "novel-rag-qwen"

    # 子配置
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    post_training: PostTrainingConfig = field(default_factory=PostTrainingConfig)  # 新增

    # 风格配置
    styles: List[str] = field(default_factory=lambda: [
        "仙侠", "武侠", "玄幻", "都市", "科幻", "历史", "悬疑"
    ])

    # 日志配置
    log_level: str = "INFO"
    log_file: str = "logs/novel_rag.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    use_wandb: bool = False
    wandb_project: str = "novel-rag"

    # 硬件配置
    use_gpu: bool = True
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    num_workers: int = 4
    pin_memory: bool = True

    # 环境配置
    random_seed: int = 42
    deterministic: bool = True

    # 优化配置
    compile_model: bool = False
    use_deepspeed: bool = False
    deepspeed_config: Optional[str] = None

    def __post_init__(self):
        """初始化后处理"""
        # 设置experiment_name
        if not self.post_training.experiment_name:
            from datetime import datetime
            self.post_training.experiment_name = f"{self.project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 创建必要的目录
        for dir_path in [
            self.data.novel_data_dir,
            self.data.dataset_dir,
            self.data.knowledge_base_dir,
            self.data.processed_data_dir,
            self.data.cache_dir,
            self.model.cache_dir,
            self.training.output_dir,
            self.rag.vector_store_path,
            Path(self.log_file).parent
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def save(self, path: str):
        """保存配置到JSON文件"""
        import json
        from dataclasses import asdict

        def convert_types(obj):
            """转换特殊类型"""
            if isinstance(obj, Path):
                return str(obj)
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            return obj

        config_dict = asdict(self)
        # 移除torch.dtype等不能序列化的对象
        if 'dtype' in config_dict.get('model', {}):
            dtype_value = config_dict['model']['dtype']
            if dtype_value is not None and not isinstance(dtype_value, str):
                config_dict['model']['dtype'] = str(dtype_value).replace('torch.', '')

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2, default=convert_types)

        print(f"配置已保存到: {path}")

    @classmethod
    def load(cls, path: str):
        """从JSON文件加载配置"""
        import json

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 递归创建嵌套的dataclass实例
        def create_dataclass(config_class, config_data):
            if not isinstance(config_data, dict):
                return config_data

            # 处理子配置
            for key, value in config_data.items():
                if key == 'model' and isinstance(value, dict):
                    config_data[key] = ModelConfig(**value)
                elif key == 'training' and isinstance(value, dict):
                    config_data[key] = TrainingConfig(**value)
                elif key == 'lora' and isinstance(value, dict):
                    config_data[key] = LoRAConfig(**value)
                elif key == 'rag' and isinstance(value, dict):
                    config_data[key] = RAGConfig(**value)
                elif key == 'generation' and isinstance(value, dict):
                    config_data[key] = GenerationConfig(**value)
                elif key == 'data' and isinstance(value, dict):
                    config_data[key] = DataConfig(**value)
                elif key == 'post_training' and isinstance(value, dict):  # 新增
                    config_data[key] = PostTrainingConfig(**value)

            return config_class(**config_data)

        config = create_dataclass(cls, data)
        print(f"配置已从 {path} 加载")
        return config

    def get_model_loader_kwargs(self) -> dict:
        """获取ModelLoader初始化参数"""
        return {
            'model_name_or_path': self.model.model_name_or_path,
            'device': self.model.device,
            'dtype': self.model.get_torch_dtype(),
            'load_in_8bit': self.model.load_in_8bit,
            'load_in_4bit': self.model.load_in_4bit,
            'trust_remote_code': self.model.trust_remote_code,
            'attn_implementation': self.model.attn_implementation,
            'cache_dir': self.model.cache_dir,
            'verbose': self.model.verbose
        }
    
    def get_generation_kwargs(self) -> dict:
        """获取生成参数"""
        return {
            'max_new_tokens': self.generation.max_new_tokens,
            'temperature': self.generation.temperature,
            'top_p': self.generation.top_p,
            'top_k': self.generation.top_k,
            'do_sample': self.generation.do_sample,
            'repetition_penalty': self.generation.repetition_penalty,
            'length_penalty': self.generation.length_penalty,
            'num_beams': self.generation.num_beams,
            'early_stopping': self.generation.early_stopping,
            'pad_token_id': self.generation.pad_token_id,
            'eos_token_id': self.generation.eos_token_id,
        }


# 创建默认配置实例
default_config = SystemConfig()

# 使用示例
if __name__ == "__main__":
    # 创建配置
    config = SystemConfig()
    
    # 修改特定配置
    config.model.device = "cuda"
    config.model.dtype = "float16"
    config.model.load_in_4bit = True
    config.generation.max_new_tokens = 2048
    
    # 保存配置
    config.save("config.json")
    
    # 加载配置
    loaded_config = SystemConfig.load("config.json")
    
    # 获取模型加载器参数
    model_kwargs = loaded_config.get_model_loader_kwargs()
    print("ModelLoader参数:", model_kwargs)
    
    # 获取生成参数
    gen_kwargs = loaded_config.get_generation_kwargs()
    print("生成参数:", gen_kwargs)