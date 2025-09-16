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

        import os
        import torch
        import gc
        
        # 清理显存
        torch.cuda.empty_cache()
        gc.collect()
        # 检查是否使用本地模型
        model_path = self.config.model.model_name_or_path
        
        # 如果是 Qwen 模型，尝试使用本地路径
        if "Qwen2.5-3B-Instruct" in model_path:
            local_path = "D:/Project/novel/models/transformers_cache/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
            if os.path.exists(local_path):
                logger.info(f"使用本地模型: {local_path}")
                model_path = local_path
                # 设置离线模式
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                os.environ['HF_DATASETS_OFFLINE'] = '1'

        logger.info(f"加载模型: {self.config.model.model_name_or_path}")
        
        from transformers import BitsAndBytesConfig
    
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        logger.info("使用 4bit 量化加载模型以节省显存")

        # 量化配置
        ##quantization_config = None
        '''if self.config.model.load_in_4bit:
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
            )'''
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.config.model.trust_remote_code,
            use_fast=True,
            local_files_only=True
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        device = "cuda" if torch.cuda.is_available() else "cpu"


        ''' # 量化配置
        if self.config.model.load_in_4bit and device == "cuda":
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,  # 改为 float16
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            # 使用量化加载
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                dtype=torch.float16,
                device_map="auto",  # 量化时可以用 auto
                trust_remote_code=self.config.model.trust_remote_code,
                local_files_only=True
            )
            
            # 准备模型进行量化训练
            from peft import prepare_model_for_kbit_training
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config.training.gradient_checkpointing
            )
            
        elif self.config.model.load_in_8bit and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=self.config.model.trust_remote_code,
                dtype=torch.float16,
                local_files_only=True,
                low_cpu_mem_usage=True
            )
            
            # 8bit 训练准备
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config.training.gradient_checkpointing
            )
            
        else:
            # 非量化加载 - 避免 meta tensor 问题
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=self.config.model.trust_remote_code,
                local_files_only=True,
                low_cpu_mem_usage=True
            )
            
            # 手动移动到设备
            self.model = self.model.to(device)
            
            # 如果没有量化但启用了梯度检查点
            if self.config.training.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()'''
        # 加载量化模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.float16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            max_memory={0: "7GB"},  # 限制最大显存使用
        )
        
        # 准备量化训练
        from peft import prepare_model_for_kbit_training
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=True  # 必须启用
        )
        
        # 禁用缓存（与梯度检查点不兼容）
        self.model.config.use_cache = False
        
        # 设置LoRA
        self._setup_lora_optimized()
        
        logger.success("模型和分词器设置完成")
    
    def _setup_lora_optimized(self):
        """优化的 LoRA 设置（更小的参数）"""
        from peft import LoraConfig, get_peft_model, TaskType
        
        # 使用更小的 LoRA 参数
        self.peft_config = LoraConfig(
            r=8,  # 降低 rank（原来是 16）
            lora_alpha=16,  # 降低 alpha（原来是 32）
            target_modules=["q_proj", "v_proj"],  # 只针对部分模块
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()

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
        max_length = self.tokenizer.model_max_length or 256

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
            
                 # 截断输出以适应最大长度
                prompt_tokens = self.tokenizer(prompt, truncation=False, add_special_tokens=False)['input_ids']
                max_output_length = max_length - len(prompt_tokens) - 10  # 预留一些空间

                if max_output_length <= 0:
                    # 如果提示太长，截断提示
                    prompt = instruction[:100] + "\n输出："
                    prompt_tokens = self.tokenizer(prompt, truncation=False, add_special_tokens=False)['input_ids']
                    max_output_length = max_length - len(prompt_tokens) - 10
                
                # 截断输出
                output_truncated = output[:max_output_length * 2]  # 粗略估计字符数
                full_text = prompt + output_truncated + self.tokenizer.eos_token

                # 2. 分词
                tokenized = self.tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                add_special_tokens=True
                )

                # 创建标签
                prompt_ids = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
                labels = tokenized['input_ids'].copy()
                #prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
                #full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids

                '''# 3. 截断
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
                model_inputs["labels"].append(labels)'''

                # 掩码提示部分
                for j in range(min(len(prompt_ids), len(labels))):
                    labels[j] = -100
            
                model_inputs["input_ids"].append(tokenized['input_ids'])
                model_inputs["attention_mask"].append(tokenized['attention_mask'])
                model_inputs["labels"].append(labels)

            return model_inputs

        # 转换为Dataset
        train_dataset = Dataset.from_list([ex.to_dict() for ex in train_examples])
        val_dataset = Dataset.from_list([ex.to_dict() for ex in val_examples])
        
        # 预处理
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            batch_size=4,
            remove_columns=train_dataset.column_names,
            desc="处理训练集"
        )
        val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=4,
        remove_columns=val_dataset.column_names,
        desc="处理验证集"
    )
    
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset, resume_from_checkpoint=None):
        """训练模型"""
        import gc
        import torch

        # 清理内存
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("开始训练...")
        
        # 判断是否有验证集
        do_eval = val_dataset is not None and len(val_dataset) > 0

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
            eval_steps=self.config.training.eval_steps if do_eval else None,
            save_total_limit=self.config.training.save_total_limit,
            load_best_model_at_end=self.config.training.load_best_model_at_end,
            metric_for_best_model=self.config.training.metric_for_best_model,
            greater_is_better=self.config.training.greater_is_better,
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            optim=self.config.training.optim,
            seed=self.config.training.seed,
            report_to="none",
            dataloader_pin_memory = False,
            max_grad_norm=0.3,
            run_name=f"{self.config.project_name}_sft",
            eval_strategy="steps" if do_eval else "no",
            save_strategy="steps",
            do_eval=do_eval,
            logging_first_step=True,
            remove_unused_columns=False,
            label_names=["labels"],
        )
        
        '''# 根据是否有验证集设置评估相关参数
        if do_eval:
            # 有验证集时的设置
            training_args.update({
                "do_eval": True,
                "evaluation_strategy": "steps",  # 或者 "epoch"
                "eval_steps": 500,  # 每500步评估一次
                "save_strategy": "steps",
                "load_best_model_at_end": False,  # 显存不足时不加载最佳模型
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            })
        else:
            # 没有验证集时的设置
            training_args.update({
                "do_eval": False,
                "evaluation_strategy": "no",  # 关键：设置为 "no"
                "save_strategy": "steps",
                # 不设置评估相关的参数
            })
         # 创建训练参数
        training_args = TrainingArguments(**training_args)'''

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
            eval_dataset=val_dataset if do_eval else None,
            data_collator=data_collator,
            callbacks=[]
        )
        # 清理内存
        gc.collect()
        torch.cuda.empty_cache()

        # 开始训练
        try:
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("显存不足，尝试清理缓存后重试...")
                torch.cuda.empty_cache()
                # 可以考虑进一步减少batch size或序列长度
                raise
        
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
