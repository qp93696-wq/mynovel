"""
train/sft_trainer.py - 监督微调训练器
支持LoRA、QLoRA、全参数微调
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import json

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset, DatasetDict
from loguru import logger
import wandb
from tqdm import tqdm
import numpy as np


class SFTTrainer:
    """监督微调训练器"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_config = None
        
        # 设置环境变量
        os.environ["WANDB_PROJECT"] = f"{config.project_name}_sft"
        os.environ["WANDB_LOG_MODEL"] = "false"
        
    def setup_model_and_tokenizer(self):
        """设置模型和分词器"""
        logger.info(f"加载模型: {self.config.model.base_model}")
        
        # 量化配置
        quantization_config = None
        if self.config.model.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.config.model.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
            )
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model,
            trust_remote_code=self.config.model.trust_remote_code,
            use_fast=True
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=self.config.model.trust_remote_code,
            torch_dtype=torch.bfloat16 if self.config.training.bf16 else torch.float16,
            use_flash_attention_2=self.config.model.use_flash_attention
        )
        
        # 准备模型进行k-bit训练
        if quantization_config:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config.training.gradient_checkpointing
            )
        
        # 设置LoRA
        self._setup_lora()
        
        logger.success("模型和分词器设置完成")
    
    def _setup_lora(self):
        """设置LoRA"""
        # 自动检测目标模块
        target_modules = self._find_target_modules()
        
        logger.info(f"LoRA目标模块: {target_modules}")
        
        self.peft_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.config.lora.lora_dropout,
            bias=self.config.lora.bias,
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()
    
    def _find_target_modules(self) -> List[str]:
        """自动查找目标模块"""
        # 常见的目标模块名称模式
        patterns = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                   'gate_proj', 'up_proj', 'down_proj',
                   'c_attn', 'c_proj', 'c_fc']
        
        target_modules = set()
        
        for name, module in self.model.named_modules():
            # 检查是否是线性层
            if isinstance(module, nn.Linear):
                # 检查名称是否匹配模式
                for pattern in patterns:
                    if pattern in name:
                        # 提取模块名的最后一部分
                        module_name = name.split('.')[-1]
                        target_modules.add(module_name)
        
        # 如果没找到，使用默认值
        if not target_modules:
            target_modules = self.config.lora.target_modules
        
        return list(target_modules)
    
    def prepare_dataset(self, train_examples, val_examples):
        """准备数据集"""
        logger.info("准备数据集...")
        max_length = self.tokenizer.model_max_length or 512

        def preprocess_function(examples):
            """预处理函数"""
            model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
            
            for i in range(len(examples['instruction'])):
                instruction = examples['instruction'][i]
                input_text = examples['input'][i]
                output = examples['output'][i]

                # 1. 构建提示和完整文本
                if input_text:
                    prompt = f"{instruction}\n输入：{input_text}\n输出："
                else:
                    prompt = f"{instruction}\n输出："
            
                full_text = prompt + output + self.tokenizer.eos_token

                # 2. 分词
                prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
                full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids

                # 3. 截断
                if len(full_ids) > max_length:
                    full_ids = full_ids[:max_length]

                # 4. 创建标签，将提示部分掩码
                labels = full_ids.copy()
                prompt_len = len(prompt_ids)
                labels[:prompt_len] = [-100] * prompt_len
                
                # 5. Padding (由DataCollator处理会更高效，但这里为了清晰也演示一下)
                padding_len = max_length - len(full_ids)
                input_ids = full_ids + [self.tokenizer.pad_token_id] * padding_len
                attention_mask = [1] * len(full_ids) + [0] * padding_len
                labels = labels + [-100] * padding_len # labels也需要padding

                model_inputs["input_ids"].append(input_ids)
                model_inputs["attention_mask"].append(attention_mask)
                model_inputs["labels"].append(labels)

            return model_inputs

        # 转换为Dataset
        train_dataset = Dataset.from_list([ex.to_dict() for ex in train_examples])
        val_dataset = Dataset.from_list([ex.to_dict() for ex in val_examples])
        
        # 预处理
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="处理训练集"
        )
        val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="处理验证集"
    )
    
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset, resume_from_checkpoint=None):
        """训练模型"""
        logger.info("开始训练...")
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.config.training.output_dir,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            warmup_steps=self.config.training.warmup_steps,
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            eval_steps=self.config.training.eval_steps,
            save_total_limit=self.config.training.save_total_limit,
            load_best_model_at_end=self.config.training.load_best_model_at_end,
            metric_for_best_model=self.config.training.metric_for_best_model,
            greater_is_better=self.config.training.greater_is_better,
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            optim=self.config.training.optim,
            seed=self.config.training.seed,
            report_to="wandb",
            run_name=f"{self.config.project_name}_sft",
            evaluation_strategy="steps",
            save_strategy="steps",
            logging_first_step=True,
            remove_unused_columns=False,
            label_names=["labels"]
        )
        
        # 数据收集器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # 回调函数
        callbacks = [
            EarlyStoppingCallback(early_stopping_patience=3),
            ProgressCallback()
        ]
        
        # 训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=callbacks
        )
        
        # 开始训练
        if resume_from_checkpoint:
            logger.info(f"从检查点恢复: {resume_from_checkpoint}")
        
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # 保存最终模型
        trainer.save_model(f"{self.config.training.output_dir}/final_model")
        self.tokenizer.save_pretrained(f"{self.config.training.output_dir}/final_model")
        
        logger.success("训练完成！")
        
        return trainer
    
    def evaluate(self, eval_dataset):
        """评估模型"""
        logger.info("评估模型...")
        
        # 创建评估用的Trainer
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        # 评估
        metrics = trainer.evaluate()
        
        # 计算困惑度
        metrics["perplexity"] = np.exp(metrics["eval_loss"])
        
        logger.info(f"评估结果: {metrics}")
        
        return metrics


class ProgressCallback(TrainerCallback):
    """进度回调"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # 格式化日志
            log_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                 for k, v in logs.items()])
            logger.info(f"Step {state.global_step}: {log_str}")
