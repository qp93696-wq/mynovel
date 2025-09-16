"""
training/dpo_trainer.py - DPO训练器
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from loguru import logger
from peft import LoraConfig, get_peft_model


class DPOTrainer:
    """DPO训练器"""
    
    def __init__(self, config, model_name_or_path: str):
        self.config = config
        self.model_name = model_name_or_path
        
        # 加载模型和分词器
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=getattr(config, 'use_4bit', False),
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            ) if getattr(config, 'use_4bit', False) else None
        )
        
        self.ref_model = None  

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # LoRA配置
        if config.use_lora:
            peft_config = LoraConfig(
                r=config.lora.r,
                lora_alpha=config.lora.lora_alpha,
                target_modules=self._find_target_modules(),
                lora_dropout=config.lora.lora_dropout,
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            logger.info(f"LoRA配置: r={config.lora.r}, alpha={config.lora.lora_alpha}")
    
    def prepare_dataset(self, preference_data: List) -> Dataset:
        """准备DPO数据集"""
        dataset_dict = {
            "prompt": [],
            "chosen": [],
            "rejected": []
        }
        
        for data in preference_data:
            dataset_dict["prompt"].append(data.prompt)
            dataset_dict["chosen"].append(data.chosen)
            dataset_dict["rejected"].append(data.rejected)
        
        return Dataset.from_dict(dataset_dict)
    
    def _find_target_modules(self) -> List[str]:
        """自动检测LoRA目标模块"""
        import re

         # 常见的目标模块名称模式
        patterns = [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',  # 注意力层
            'gate_proj', 'up_proj', 'down_proj',      # FFN层（LLaMA类）
            'c_attn', 'c_proj', 'c_fc',               # GPT类
            'query', 'key', 'value',                   # BERT类
            'dense', 'dense_h_to_4h', 'dense_4h_to_h' # 其他
        ]

        target_modules = set()
        for name, module in self.model.named_modules():
            # 检查是否是线性层
            if isinstance(module, nn.Linear):
                # 提取模块名的最后一部分
                module_name = name.split('.')[-1]
                
                # 检查是否匹配已知模式
                for pattern in patterns:
                    if pattern in module_name:
                        target_modules.add(module_name)
                        break
        
        # 如果没找到，使用默认值
        if not target_modules:
            logger.warning("未能自动检测目标模块，使用默认值")
            target_modules = {"q_proj", "v_proj"}
        
        logger.info(f"检测到的LoRA目标模块: {target_modules}")
        return list(target_modules)

    def train(self, preference_data: List):
        """使用TRL的DPO训练"""
        logger.info("开始DPO训练...")
        
        # 准备数据集
        dataset = self.prepare_dataset(preference_data)

        dpo_training_config = getattr(self.config.training, 'dpo', {})
        output_dir = f"{self.config.training.output_dir}/dpo_final"
        
        # DPO配置
        training_args = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=getattr(self.config.training, 'dpo_epochs', 2),
            per_device_train_batch_size=getattr(self.config.training, 'dpo_batch_size', 1),
            learning_rate=getattr(self.config.training, 'dpo_learning_rate', 1e-5),
            beta=getattr(self.config.training, 'dpo_beta', 0.1),
            
            # 显存优化设置
            gradient_accumulation_steps=getattr(self.config.training, 'gradient_accumulation_steps', 8),
            gradient_checkpointing=getattr(self.config.training, 'gradient_checkpointing', True),
            fp16=getattr(self.config.training, 'fp16', True),
            optim="paged_adamw_8bit",  # 8bit优化器
            
            # 训练设置
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            
            # DPO特定设置
            max_prompt_length=256,
            max_length=512,
            
            # 可选：生成评估
            generate_during_eval=False,  # 节省显存
        )
        
        # 创建DPO训练器
        dpo_trainer = DPOTrainer(
            model=self.model,
            ref_model=None,  # 关键：让DPOTrainer自动管理
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=training_args,
        )
        
        # 训练
        train_result = dpo_trainer.train()
        
        # 保存模型（补充1：保存训练好的模型）
        logger.info("保存模型...")
        
        # 保存完整模型
        final_model_path = f"{output_dir}/final_model"
        dpo_trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        # 如果使用LoRA，也保存LoRA权重
        if getattr(self.config, 'use_lora', True):
            lora_weights_path = f"{output_dir}/lora_weights"
            self.model.save_pretrained(lora_weights_path)
            logger.info(f"LoRA权重已保存到: {lora_weights_path}")
        
        # 保存训练指标
        metrics_path = f"{output_dir}/training_metrics.json"
        with open(metrics_path, 'w') as f:
            import json
            json.dump(train_result.metrics, f, indent=2)
        
        logger.success(f"DPO训练完成！模型已保存到: {final_model_path}")
        
        return dpo_trainer
    
    def merge_and_save(self, output_path: str):
        """
        合并LoRA权重并保存完整模型（可选功能）
        
        Args:
            output_path: 输出路径
        """
        if hasattr(self.model, 'merge_and_unload'):
            logger.info("合并LoRA权重...")
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            logger.success(f"合并后的模型已保存到: {output_path}")
        else:
            logger.warning("当前模型不支持合并操作")