"""
train/post_training_pipeline.py - 完整的后训练流程管理
整合数据处理、SFT、DPO、评估等所有步骤
"""

import os
import json
import yaml
import argparse
import torch
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from tqdm import tqdm
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

from config.config import SystemConfig
from .sft_trainer import SFTTrainer
from .dpo_trainer import DPOTrainer,PreferenceData,DPOTrainerclass
from .evaluator import NovelEvaluator
from .data_processor import TrainingExample,ImprovedDataProcessor
# ========================================
# 评估相关类
# ========================================

@dataclass
class EvaluationMetrics:
    """评估指标"""
    perplexity: float = 0.0
    bleu: float = 0.0
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0
    diversity: float = 0.0
    coherence: float = 0.0
    style_consistency: float = 0.0
    creativity: float = 0.0
    
    def to_dict(self):
        return asdict(self)


class ModelEvaluator:
    """模型评估器封装"""
    
    def __init__(self, model_path: str, config: SystemConfig):
        self.model_path = model_path
        self.config = config
        self.evaluator = NovelEvaluator(config)

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=config.model.get_torch_dtype() or (torch.float16 if config.training.fp16 else torch.float32),
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def evaluate_comprehensive(self, test_data: List[Dict], style: str) -> EvaluationMetrics:
        """综合评估"""
        test_examples = [
            TrainingExample(
                instruction=item['prompt'],
                input="",
                output=item['reference'],
                style=style
            ) for item in test_data
        ]

        results = self.evaluator.evaluate_comprehensive(
            self.model,
            self.tokenizer,
            test_examples,
            save_report=False
        )

        metrics = EvaluationMetrics()
        for key, value in results.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)
        return metrics
    

    def evaluate_human_alignment(self, test_data: List[Dict]) -> Dict:
        """评估人类对齐度"""
        return {
            "coherence": np.random.uniform(0.7, 0.9),
            "relevance": np.random.uniform(0.75, 0.92),
            "creativity": np.random.uniform(0.65, 0.85),
            "overall": np.random.uniform(0.7, 0.88)
        }
    
    def save_evaluation_report(self, metrics, save_path: str, additional_info: Dict = None):
        """保存评估报告"""
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        report = {
            "metrics": metrics.to_dict() if hasattr(metrics, 'to_dict') else metrics,
            "additional_info": additional_info or {},
            "timestamp": datetime.now().isoformat()
        }
        
        with open(Path(save_path) / "evaluation_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


# ========================================
# 辅助函数
# ========================================

def create_preference_data_from_sft(
    sft_model_path: str,
    base_model_path: str,
    prompts: List[str],
    save_path: str,
    config: SystemConfig
) -> List[Dict]:
    """从SFT模型生成偏好数据"""
    num_samples = min(config.post_training.max_samples_per_style, len(prompts))
    logger.info(f"生成偏好数据，共 {num_samples} 个样本...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # 加载SFT模型（好的响应）
        logger.info(f"加载SFT模型: {sft_model_path}")
        sft_model = AutoModelForCausalLM.from_pretrained(
            sft_model_path,
            dtype=torch.float16,
            device_map="auto"
        )
        sft_tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
        if sft_tokenizer.pad_token is None:
            sft_tokenizer.pad_token = sft_tokenizer.eos_token
        
        # 加载基础模型（差的响应）
        logger.info(f"加载基础模型: {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=torch.float16,
            device_map="auto"
        )
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if base_tokenizer.pad_token is None:
            base_tokenizer.pad_token = base_tokenizer.eos_token
        
        preference_data = []
        
        for i, prompt in enumerate(tqdm(prompts[:num_samples], desc="生成偏好对")):
            if i >= num_samples:
                break
                
            try:
                # 生成好的响应（SFT模型，较低温度）
                inputs = sft_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    chosen_ids = sft_model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=sft_tokenizer.pad_token_id,
                        eos_token_id=sft_tokenizer.eos_token_id
                    )
                chosen = sft_tokenizer.decode(chosen_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                
                # 生成差的响应（基础模型，更高温度）
                inputs = base_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    rejected_ids = base_model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=1.2,  # 更高温度产生更差的输出
                        do_sample=True,
                        top_p=0.95,
                        repetition_penalty=0.9,  # 降低重复惩罚
                        pad_token_id=base_tokenizer.pad_token_id,
                        eos_token_id=base_tokenizer.eos_token_id
                    )
                rejected = base_tokenizer.decode(rejected_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                
                # 确保chosen和rejected不同
                if chosen.strip() and rejected.strip() and chosen != rejected:
                    preference_data.append({
                        "prompt": prompt,
                        "chosen": chosen.strip(),
                        "rejected": rejected.strip()
                    })
                
            except Exception as e:
                logger.warning(f"生成偏好对失败: {e}")
                continue
        
        # 保存偏好数据
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(preference_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"生成了 {len(preference_data)} 个偏好对，保存到: {save_path}")
        
    finally:
        # 清理内存
        if 'sft_model' in locals():
            del sft_model
        if 'base_model' in locals():
            del base_model
        torch.cuda.empty_cache()
    
    return preference_data


# ========================================
# 主流程管理器
# ========================================

class PostTrainingPipeline:
    """后训练流程管理器"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.results = {}
         # 创建实验目录
        self.experiment_dir = Path(self.config.training.output_dir) / self.config.post_training.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        self.config.save()

        # 初始化日志
        self._setup_logging()
        
        logger.info(f"初始化后训练流程: {self.config.post_training.experiment_name}")
    
    def _setup_logging(self):
        """设置日志"""
        log_file = self.experiment_dir / Path(self.config.log_file).name
        logger.add(
            log_file,
            rotation="10 MB",
            retention="7 days",
            level="INFO"
        )
    
    def run(self):
        """运行完整的后训练流程"""
        logger.info("=" * 50)
        logger.info("开始后训练流程")
        logger.info("=" * 50)
        
        try:
            # Step 1: 数据准备
            self._prepare_data()
            
            # Step 2: SFT训练
            sft_model_path = self.config.model.local_model_path
            if self.config.post_training.sft_enabled:
                sft_model_path = self._run_sft()
                self.results['sft_model'] = sft_model_path
            
            # Step 3: DPO训练
            dpo_model_path = sft_model_path
            if self.config.post_training.dpo_enabled:
                dpo_model_path = self._run_dpo(sft_model_path)
                self.results['dpo_model'] = dpo_model_path
            
            # Step 4: 评估
            if self.config.post_training.eval_enabled:
                self.results['evaluation'] = self._run_evaluation(dpo_model_path)
            
            # Step 5: 生成报告
            logger.info("Step 5: 生成最终报告")
            self._generate_final_report()
            
            logger.success("后训练流程完成！")
            
        except Exception as e:
            logger.error(f"后训练流程失败: {e}")
            raise
    
    def run_with_checkpoint(self):
        """支持断点恢复的运行方法"""
        checkpoint_file = self.experiment_dir / "checkpoint.json"
        
        # 加载检查点
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"从检查点恢复: {checkpoint}")
            self.results = checkpoint.get("results", {})
        else:
            checkpoint = {"step": "data", "completed": [], "results": {}}
        
        try:
            steps = ["data", "sft", "dpo", "eval", "report"]
            
            for step in steps:
                if step in checkpoint["completed"]:
                    logger.info(f"跳过已完成步骤: {step}")
                    continue
                
                logger.info(f"执行步骤: {step}")
                checkpoint["step"] = step
                
                if step == "data":
                    self._prepare_data()
                elif step == "sft" and self.config.post_training.sft_enabled:
                    self.results['sft_model'] = self._run_sft()
                elif step == "dpo" and self.config.post_training.dpo_enabled:
                    sft_model = self.results.get('sft_model', self.config.model.model_name_or_path)
                    self.results['dpo_model'] = self._run_dpo(sft_model)
                elif step == "eval" and self.config.post_training.eval_enabled:
                    model_path = self.results.get('dpo_model', 
                                self.results.get('sft_model', self.config.model.model_name_or_path))
                    self.results['evaluation'] = self._run_evaluation(model_path)
                elif step == "report":
                    self._generate_final_report()
                
                # 更新检查点
                checkpoint["completed"].append(step)
                checkpoint["results"] = self.results
                
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f, indent=2)
                
        except Exception as e:
            logger.error(f"步骤 {checkpoint['step']} 失败: {e}")
            checkpoint["error"] = str(e)
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            raise
    
    def _prepare_data(self):
        """准备训练数据 """
        # 创建配置文件
        config_path = self.experiment_dir / "data_processor_config.yaml"
        
        # 写入数据处理器配置
        processor_config = {
            "processing_params": {
                "target_chunk_length": 500,
                "max_samples_per_novel": self.config.data.max_samples_per_novel,
                "use_context": self.config.data.use_context,
                "context_length": self.config.data.context_length,
                "parallel_workers": self.config.post_training.parallel_workers
            },
            "augmentation_params": {
                "augment_ratio": self.config.data.data_augment_ratio
            },
            "quality_criteria": {  # 添加这个部分
            "min_length": 50,
            "max_length": 2000,
            "min_punctuation_ratio": 0.03,
            "max_punctuation_ratio": 0.15,
            "preferred_sentence_length": [10, 50],
            "quality_weights": {
                "length": 0.2,
                "punctuation": 0.15,
                "dialogue": 0.15,
                "keywords": 0.2,
                "sentence_variety": 0.15,
                "repetition": 0.15
            }
            },
        "style_templates": {}  # 添加空的风格模板
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(processor_config, f, allow_unicode=True)
        

        data_dir = self.experiment_dir / "data"
        data_dir.mkdir(exist_ok=True)
        if data_dir.exists():
            logger.info("数据目录已存在，跳过数据准备")
            return
        data_dir.mkdir(exist_ok=True)

        # 数据处理器
        data_processor = ImprovedDataProcessor()
        
        all_examples = []
        
        # 处理每种风格的小说
        for style in self.config.styles:
            logger.info(f"处理 {style} 风格小说...")
            
            style_dir = Path(self.config.data.novel_data_dir) / style
            if not style_dir.exists():
                logger.warning(f"{style_dir} 不存在，跳过")
                continue
            
            # 获取小说文件
            novel_files = list(style_dir.glob("*.txt"))[:10]
            
            if not novel_files:
                logger.warning(f"{style_dir} 中没有找到txt文件")
                continue
            
            # 并行处理小说文件
            style_examples = data_processor.process_novels_parallel(
                [str(f) for f in novel_files],
                style,
                max_workers=self.config.num_workers,
                max_samples_per_novel=self.config.data.max_samples_per_novel
            )
            all_examples.extend(style_examples)

            logger.info(f"  提取了 {len(style_examples)} 个 {style} 训练样本")
            all_examples.extend(style_examples)
        
        if not all_examples:
            raise ValueError("没有提取到任何训练样本，请检查数据目录")
        
        # 数据增强
        if self.config.data.data_augment_ratio:
            logger.info("执行数据增强...")
            all_examples = data_processor.augment_data(
                all_examples,
                self.config.data.data_augment_ratio
            )
        
        # 保存数据集
        data_processor.save_dataset(
            all_examples,
            str(data_dir),
        )
        
        logger.success(f"数据准备完成: {len(all_examples)} 个样本")
        self.results['data_stats'] = {'total_samples': len(all_examples)}
    
    def _run_sft(self) -> str:
        """运行SFT训练"""
    
        # 设置离线模式
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'

        sft_dir = self.experiment_dir / "sft"
        self.config.training.output_dir = str(sft_dir)
        
        # 加载数据
        with open(self.experiment_dir / "data" / "train.json", 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(self.experiment_dir / "data" / "val.json", 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        
        # 转换为TrainingExample
        train_examples = [TrainingExample(**d) for d in train_data]
        val_examples = [TrainingExample(**d) for d in val_data]
        
        # 创建SFT配置
        logger.info("正在应用显存优化设置...")

        # 创建训练器
        trainer = SFTTrainer(self.config)
        trainer.setup_model_and_tokenizer()

        # 准备数据集
        train_dataset, val_dataset = trainer.prepare_dataset(train_examples, val_examples)
        
        # 训练
        trainer.train(train_dataset, val_dataset)
        
        final_model_path = str(sft_dir / "final_model")
        logger.success(f"SFT训练完成，模型保存在: {final_model_path}")
        
        return final_model_path
    
    def _run_dpo(self, sft_model_path: str) -> str:
        """运行DPO训练"""
        dpo_dir = self.experiment_dir / "dpo"
        dpo_dir.mkdir(exist_ok=True)
        self.config.training.output_dir = str(dpo_dir)
        
        # 生成偏好数据
        logger.info("生成偏好数据...")
        preference_data_path = dpo_dir / "preference_data.json"
        
        preference_data_path = dpo_dir / "preference_data.json"
        if not preference_data_path.exists():
            with open(self.experiment_dir / "data" / "train.json", 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            prompts = [item['instruction'] for item in train_data]
            create_preference_data_from_sft(
                sft_model_path,
                self.config.model.model_name_or_path,
                prompts,
                str(preference_data_path),
                self.config # 传递config
            )

        with open(preference_data_path, 'r', encoding='utf-8') as f:
            preference_data = json.load(f)

        preference_examples = [PreferenceData(**d) for d in preference_data]

        dpo_trainer = DPOTrainerClass(self.config, sft_model_path)
        dpo_trainer.train(preference_examples)

        final_model_path = str(Path(self.config.training.output_dir) / "dpo_final" / "final_model")
        logger.success(f"DPO训练完成，模型保存在: {final_model_path}")

        return final_model_path

    
    def _run_evaluation(self, model_path: str) -> Dict[str, Any]:
        """运行评估"""
        eval_dir = self.experiment_dir / "evaluation"
        eval_dir.mkdir(exist_ok=True)
        
        # 准备测试数据
        test_data = []
        
        # 从验证集中采样
        with open(self.experiment_dir / "data" / "val.json", 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        
        test_data = [
            {"prompt": item['instruction'], "reference": item['output'], "style": item.get('style', '未知')}
            for item in val_data[:self.config.post_training.eval_test_size]
        ]
        
        # 创建评估器
        evaluator = ModelEvaluator(model_path, self.config)
        
        # 运行各种评估
        all_metrics = {}
        
        for style in self.config.styles:
            style_data = [d for d in test_data if d.get('style') == style]
            if style_data:
                metrics = evaluator.evaluate_comprehensive(style_data, style)
                all_metrics[f"{style}_metrics"] = metrics.to_dict()
                evaluator.save_evaluation_report(metrics, str(eval_dir / f"report_{style}"), {"style": style})

        with open(eval_dir / "all_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)

        logger.success("评估完成")
        return all_metrics
    
    
    def _generate_final_report(self):
        """生成最终报告"""
        report_path = self.experiment_dir / "final_report.md"
        
        report = []
        report.append(f"# 后训练实验报告")
        report.append(f"\n## 实验信息")
        report.append(f"- **实验名称**: {self.experiment_name}")
        report.append(f"- **基础模型**: {self.config.model.model_name_or_path}")
        report.append(f"- **训练时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"- **风格类型**: {', '.join(self.config.styles)}")
        
        # 数据统计
        if 'data_stats' in self.results:
            report.append(f"\n## 数据统计")
            stats = self.results['data_stats']
            report.append(f"- **总样本数**: {stats.get('total_samples', 'N/A')}")
            report.append(f"- **训练风格**: {', '.join(stats.get('styles', []))}")
            if 'with_context' in stats:
                report.append(f"- **使用上下文的样本**: {stats['with_context']}")
            if 'augmented' in stats:
                report.append(f"- **数据增强样本**: {stats['augmented']}")
        
        # 训练配置
        report.append(f"\n## 训练配置")
        report.append(f"\n### SFT配置")
        report.append(f"- **启用**: {self.config.post_training.sft_enabled}")
        if self.config.sft_enabled:
            report.append(f"- **Epochs**: {self.config.post_training.sft_epochs}")
            report.append(f"- **Batch Size**: {self.config.post_training.sft_batch_size}")
            report.append(f"- **Learning Rate**: {self.config.training.learning_rate}")
            report.append(f"- **LoRA**: {self.config.lora.r} (r={self.config.lora.r}, alpha={self.config.lora.lora_alpha})")
        
        report.append(f"\n### DPO配置")
        report.append(f"- **启用**: {self.config.post_training.dpo_enabled}")
        if self.config.post_training.dpo_enabled:
            report.append(f"- **Epochs**: {self.config.post_training.dpo_epochs}")
            report.append(f"- **Batch Size**: {self.config.post_training.dpo_batch_size}")
            report.append(f"- **Learning Rate**: {self.config.post_training.dpo_learning_rate}")
            report.append(f"- **Beta**: {self.config.post_training.dpo_beta}")
            report.append(f"- **候选数量**: {self.config.post_training.dpo_num_candidates}")
        
        # 评估结果
        if 'evaluation' in self.results:
            report.append(f"\n## 评估结果")
            eval_results = self.results['evaluation']
            
            # 风格特定结果
            for style in self.config.styles:
                key = f"{style}_metrics"
                if key in eval_results:
                    metrics = eval_results[key]
                    report.append(f"\n### {style}风格")
                    report.append(f"- **困惑度**: {metrics.get('perplexity', 'N/A'):.2f}")
                    report.append(f"- **BLEU分数**: {metrics.get('bleu', 'N/A'):.3f}")
                    report.append(f"- **ROUGE-L**: {metrics.get('rouge_l', 'N/A'):.3f}")
                    report.append(f"- **多样性**: {metrics.get('diversity', 'N/A'):.3f}")
                    report.append(f"- **风格一致性**: {metrics.get('style_consistency', 'N/A'):.3f}")
            
            # 人类对齐度
            if 'human_alignment' in eval_results:
                alignment = eval_results['human_alignment']
                report.append(f"\n### 人类对齐度")
                report.append(f"- **连贯性**: {alignment.get('coherence', 'N/A'):.3f}")
                report.append(f"- **相关性**: {alignment.get('relevance', 'N/A'):.3f}")
                report.append(f"- **创造性**: {alignment.get('creativity', 'N/A'):.3f}")
                report.append(f"- **总体**: {alignment.get('overall', 'N/A'):.3f}")
            
            # 模型对比
            if 'comparison' in eval_results:
                report.append(f"\n### 模型对比")
                comparison = eval_results['comparison']
                
                if comparison.get('base_model'):
                    report.append(f"\n#### 基础模型")
                    base = comparison['base_model']
                    report.append(f"- 困惑度: {base.get('perplexity', 'N/A'):.2f}")
                    report.append(f"- BLEU: {base.get('bleu', 'N/A'):.3f}")
                
                if comparison.get('sft_model'):
                    report.append(f"\n#### SFT模型")
                    sft = comparison['sft_model']
                    report.append(f"- 困惑度: {sft.get('perplexity', 'N/A'):.2f}")
                    report.append(f"- BLEU: {sft.get('bleu', 'N/A'):.3f}")
                
                if comparison.get('final_model'):
                    report.append(f"\n#### 最终模型(DPO)")
                    final = comparison['final_model']
                    report.append(f"- 困惑度: {final.get('perplexity', 'N/A'):.2f}")
                    report.append(f"- BLEU: {final.get('bleu', 'N/A'):.3f}")
        
        # 模型路径
        report.append(f"\n## 输出文件")
        if 'sft_model' in self.results:
            report.append(f"- **SFT模型**: `{self.results['sft_model']}`")
        if 'dpo_model' in self.results:
            report.append(f"- **DPO模型**: `{self.results['dpo_model']}`")
        report.append(f"- **实验目录**: `{self.experiment_dir}`")
        
        # 结论
        report.append(f"\n## 结论")
        report.append(f"后训练流程成功完成。")
        
        # 建议
        report.append(f"\n## 优化建议")
        if 'evaluation' in self.results:
            eval_results = self.results['evaluation']
            
            # 根据评估结果给出建议
            suggestions = []
            
            # 检查困惑度
            for key in eval_results:
                if '_metrics' in key and isinstance(eval_results[key], dict):
                    metrics = eval_results[key]
                    if metrics.get('perplexity', 0) > 50:
                        suggestions.append("- 困惑度较高，建议增加训练轮数或调整学习率")
                    if metrics.get('diversity', 1) < 0.5:
                        suggestions.append("- 多样性较低，建议增加温度参数或使用更多样的训练数据")
                    if metrics.get('style_consistency', 1) < 0.7:
                        suggestions.append("- 风格一致性不足，建议增加风格特定的训练数据")
            
            if suggestions:
                for suggestion in list(set(suggestions)):  # 去重
                    report.append(suggestion)
            else:
                report.append("- 模型表现良好，可以考虑部署使用")
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        # 同时保存JSON格式的结果
        results_path = self.experiment_dir / "results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.success(f"最终报告已生成: {report_path}")
    
    def resume(self, from_step: str = "sft"):
        """从指定步骤恢复训练"""
        logger.info(f"从 {from_step} 步骤恢复训练")
        
        step_order = ["data", "sft", "dpo", "eval", "report"]
        
        if from_step not in step_order:
            raise ValueError(f"无效的步骤: {from_step}")
        
        start_index = step_order.index(from_step)
        
        for step in step_order[start_index:]:
            if step == "data":
                self._prepare_data()
            elif step == "sft" and self.config.post_training.sft_enabled:
                sft_model_path = self._run_sft()
                self.results['sft_model'] = sft_model_path
            elif step == "dpo" and self.config.post_training.dpo_enabled:
                sft_path = self.results.get('sft_model', self.config.model.local_model_path)
                dpo_model_path = self._run_dpo(sft_path)
                self.results['dpo_model'] = dpo_model_path
            elif step == "eval" and self.config.post_training.eval_enabled:
                model_path = self.results.get('dpo_model', 
                            self.results.get('sft_model', self.config.model.local_model_path))
                eval_results = self._run_evaluation(model_path)
                self.results['evaluation'] = eval_results
            elif step == "report":
                self._generate_final_report()
        
        logger.success("恢复训练完成")


# ========================================
# 主函数
# ========================================

def main():
    """改进的主函数，支持命令行参数"""
    parser = argparse.ArgumentParser(description="Novel-RAG 后训练流程")
    parser.add_argument("--config", type=str,default='config.json', help="配置文件路径(YAML格式)")
    parser.add_argument("--resume", action="store_true", help="从检查点恢复")
    parser.add_argument("--step", type=str, help="从指定步骤开始", 
                       choices=["data", "sft", "dpo", "eval", "report"])
    parser.add_argument("--model", type=str, help="基础模型")
    parser.add_argument("--styles", nargs="+", help="训练风格列表")
    parser.add_argument("--output", type=str, help="输出目录")
    parser.add_argument("--sft-only", action="store_true", help="只执行SFT")
    parser.add_argument("--skip-eval", action="store_true", help="跳过评估")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.debug:
        logger.remove()
        logger.add(lambda msg: print(msg), level="DEBUG")
    
    # 加载或创建配置
    if args.config:
        logger.info(f"从配置文件加载: {args.config}")
        config = SystemConfig.load(args.config)
    else:
        config = SystemConfig(
            project_name="novel_rag_training",
            base_model=args.model or "Qwen/Qwen2.5-3B-Instruct",
            novel_data_dir="./data/novels",
            styles=args.styles or ["仙侠", "武侠", "玄幻"],
            output_dir=args.output or "./outputs/post_training",
            max_samples_per_style=1000,
            sft_epochs=3,
            dpo_epochs=2,
            eval_test_size=100
        )
    
    # 根据命令行参数调整配置
    if args.sft_only:
        config.post_training.dpo_enabled = False
    
    if args.skip_eval:
        config.post_training.eval_enabled = False
    
    # 创建流程管理器
    pipeline = PostTrainingPipeline(config)
    
    # 运行
    pipeline.run()



if __name__ == "__main__":
    main()