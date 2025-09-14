"""
train/post_training_pipeline.py - 完整的后训练流程管理
整合数据处理、SFT、DPO、评估等所有步骤
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
import shutil
from loguru import logger

# 导入各个模块
from .data_processor import DataProcessor, TrainingExample, PreferenceData
from .sft_trainer import SFTTrainer, SFTConfig
from .dpo_trainer import DPOTrainer, DPOConfig, create_preference_data_from_sft
from .evaluator import ModelEvaluator, EvaluationMetrics


@dataclass
class PostTrainingConfig:
    """后训练完整配置"""
    # 基础配置
    project_name: str = "novel_rag_post_training"
    base_model: str = "Qwen/Qwen-1_8B-Chat"
    output_dir: str = "./outputs/post_training"
    
    # 数据配置
    novel_data_dir: str = "./data/novels"
    styles: List[str] = field(default_factory=lambda: ["仙侠", "武侠", "玄幻"])
    max_samples_per_style: int = 1000
    data_augment_ratio: float = 0.3
    
    # SFT配置
    sft_enabled: bool = True
    sft_epochs: int = 3
    sft_batch_size: int = 4
    sft_learning_rate: float = 5e-5
    sft_use_lora: bool = True
    sft_lora_r: int = 16
    
    # DPO配置
    dpo_enabled: bool = True
    dpo_epochs: int = 2
    dpo_batch_size: int = 2
    dpo_learning_rate: float = 1e-6
    dpo_beta: float = 0.1
    
    # 评估配置
    eval_enabled: bool = True
    eval_test_size: int = 100
    eval_metrics: List[str] = field(default_factory=lambda: ["perplexity", "bleu", "diversity", "style"])
    
    # 实验配置
    experiment_name: str = ""
    seed: int = 42
    use_wandb: bool = False
    wandb_project: str = "novel-rag-post-training"
    
    # 硬件配置
    device: str = "cuda"
    fp16: bool = True
    gradient_checkpointing: bool = True
    
    def __post_init__(self):
        """后处理"""
        if not self.experiment_name:
            self.experiment_name = f"{self.project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 创建输出目录
        self.experiment_dir = Path(self.output_dir) / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, path: Optional[str] = None):
        """保存配置"""
        save_path = path or self.experiment_dir / "config.yaml"
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(self), f, allow_unicode=True, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str):
        """加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return cls(**config_dict)


class PostTrainingPipeline:
    """后训练流程管理器"""
    
    def __init__(self, config: PostTrainingConfig):
        self.config = config
        self.results = {}
        
        # 初始化日志
        self._setup_logging()
        
        # 保存配置
        self.config.save()
        
        logger.info(f"初始化后训练流程: {self.config.experiment_name}")
    
    def _setup_logging(self):
        """设置日志"""
        log_file = self.config.experiment_dir / "training.log"
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
            if not (self.config.experiment_dir / "data").exists():
                logger.info("Step 1: 数据准备")
                self._prepare_data()
            else:
                logger.info("Step 1: 跳过数据准备（已存在）")
            
            # Step 2: SFT训练
            if self.config.sft_enabled:
                logger.info("Step 2: SFT监督微调")
                sft_model_path = self._run_sft()
                self.results['sft_model'] = sft_model_path
            else:
                logger.info("Step 2: 跳过SFT训练")
                sft_model_path = self.config.base_model
            
            # Step 3: DPO训练
            if self.config.dpo_enabled:
                logger.info("Step 3: DPO偏好优化")
                dpo_model_path = self._run_dpo(sft_model_path)
                self.results['dpo_model'] = dpo_model_path
            else:
                logger.info("Step 3: 跳过DPO训练")
                dpo_model_path = sft_model_path
            
            # Step 4: 评估
            if self.config.eval_enabled:
                logger.info("Step 4: 模型评估")
                eval_results = self._run_evaluation(dpo_model_path)
                self.results['evaluation'] = eval_results
            else:
                logger.info("Step 4: 跳过评估")
            
            # Step 5: 生成报告
            logger.info("Step 5: 生成最终报告")
            self._generate_final_report()
            
            logger.success("后训练流程完成！")
            
        except Exception as e:
            logger.error(f"后训练流程失败: {e}")
            raise
    
    def _prepare_data(self):
        """准备训练数据"""
        data_processor = DataProcessor(self.config)
        data_dir = self.config.experiment_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        all_examples = []
        
        # 处理每种风格的小说
        for style in self.config.styles:
            logger.info(f"处理 {style} 风格小说...")
            
            style_dir = Path(self.config.novel_data_dir) / style
            if not style_dir.exists():
                logger.warning(f"{style_dir} 不存在，跳过")
                continue
            
            # 获取小说文件
            novel_files = list(style_dir.glob("*.txt"))[:10]  # 限制数量
            
            style_examples = []
            for novel_file in novel_files:
                examples = data_processor.process_novel_to_training_data(
                    str(novel_file),
                    style,
                    max_samples=self.config.max_samples_per_style // len(novel_files)
                )
                style_examples.extend(examples)
            
            logger.info(f"  提取了 {len(style_examples)} 个 {style} 训练样本")
            all_examples.extend(style_examples)
        
        # 数据增强
        if self.config.data_augment_ratio > 0:
            logger.info("执行数据增强...")
            all_examples = data_processor.augment_data(
                all_examples,
                self.config.data_augment_ratio
            )
        
        # 保存数据集
        data_processor.save_dataset(
            all_examples,
            str(data_dir),
            split_ratio=0.9
        )
        
        logger.success(f"数据准备完成: {len(all_examples)} 个样本")
        self.results['data_stats'] = {
            'total_samples': len(all_examples),
            'styles': self.config.styles
        }
    
    def _run_sft(self) -> str:
        """运行SFT训练"""
        sft_dir = self.config.experiment_dir / "sft"
        
        # SFT配置
        sft_config = SFTConfig(
            model_name=self.config.base_model,
            use_lora=self.config.sft_use_lora,
            lora_r=self.config.sft_lora_r,
            num_epochs=self.config.sft_epochs,
            batch_size=self.config.sft_batch_size,
            learning_rate=self.config.sft_learning_rate,
            train_data_path=str(self.config.experiment_dir / "data" / "train.json"),
            val_data_path=str(self.config.experiment_dir / "data" / "val.json"),
            output_dir=str(sft_dir),
            fp16=self.config.fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            seed=self.config.seed,
            use_wandb=self.config.use_wandb,
            wandb_project=self.config.wandb_project
        )
        
        # 创建训练器并训练
        trainer = SFTTrainer(sft_config)
        train_result = trainer.train()
        
        # 评估
        eval_result = trainer.evaluate()
        
        # 生成样本测试
        test_prompts = [
            "写一段仙侠小说的开头：",
            "描述一个武侠高手的出场：",
            "创作一段玄幻世界的场景："
        ]
        
        samples = []
        for prompt in test_prompts:
            sample = trainer.generate_sample(prompt)
            samples.append({"prompt": prompt, "generated": sample})
            logger.info(f"样本: {prompt}\n生成: {sample[:100]}...")
        
        # 保存结果
        sft_results = {
            "train_loss": train_result.training_loss,
            "eval_metrics": eval_result,
            "samples": samples
        }
        
        with open(sft_dir / "results.json", 'w', encoding='utf-8') as f:
            json.dump(sft_results, f, ensure_ascii=False, indent=2)
        
        logger.success(f"SFT训练完成，模型保存在: {sft_dir}")
        return str(sft_dir / "checkpoint-final")
    
    def _run_dpo(self, sft_model_path: str) -> str:
        """运行DPO训练"""
        dpo_dir = self.config.experiment_dir / "dpo"
        dpo_dir.mkdir(exist_ok=True)
        
        # 首先生成偏好数据
        logger.info("生成偏好数据...")
        preference_data_path = dpo_dir / "preference_data.json"
        
        if not preference_data_path.exists():
            # 加载训练数据获取prompts
            with open(self.config.experiment_dir / "data" / "train.json", 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            
            prompts = [item['instruction'] for item in train_data[:500]]  # 限制数量
            
            # 生成偏好数据
            preference_data = create_preference_data_from_sft(
                sft_model_path,
                self.config.base_model,
                prompts,
                str(preference_data_path),
                num_samples=min(len(prompts), 500)
            )
        
        # 分割偏好数据
        with open(preference_data_path, 'r', encoding='utf-8') as f:
            preference_data = json.load(f)
        
        split_point = int(len(preference_data) * 0.9)
        train_pref = preference_data[:split_point]
        val_pref = preference_data[split_point:]
        
        train_pref_path = dpo_dir / "preference_train.json"
        val_pref_path = dpo_dir / "preference_val.json"
        
        with open(train_pref_path, 'w', encoding='utf-8') as f:
            json.dump(train_pref, f, ensure_ascii=False, indent=2)
        with open(val_pref_path, 'w', encoding='utf-8') as f:
            json.dump(val_pref, f, ensure_ascii=False, indent=2)
        
        # DPO配置
        dpo_config = DPOConfig(
            model_name=sft_model_path,
            ref_model_name=self.config.base_model,
            beta=self.config.dpo_beta,
            num_epochs=self.config.dpo_epochs,
            batch_size=self.config.dpo_batch_size,
            learning_rate=self.config.dpo_learning_rate,
            train_data_path=str(train_pref_path),
            val_data_path=str(val_pref_path),
            output_dir=str(dpo_dir),
            fp16=self.config.fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            seed=self.config.seed,
            use_wandb=self.config.use_wandb,
            wandb_project=self.config.wandb_project
        )
        
        # 创建训练器并训练
        trainer = DPOTrainer(dpo_config)
        trainer.train()
        
        # 生成对比样本
        test_prompts = [
            "写一段仙侠小说的战斗场景：",
            "描述一个修仙者突破境界的过程："
        ]
        
        comparisons = []
        for prompt in test_prompts:
            comparison = trainer.generate_comparison(prompt)
            comparisons.append(comparison)
            logger.info(f"DPO对比:\n原始: {comparison['reference_output'][:100]}...\n优化: {comparison['trained_output'][:100]}...")
        
        # 保存结果
        dpo_results = {
            "comparisons": comparisons,
            "config": asdict(dpo_config)
        }
        
        with open(dpo_dir / "results.json", 'w', encoding='utf-8') as f:
            json.dump(dpo_results, f, ensure_ascii=False, indent=2)
        
        logger.success(f"DPO训练完成，模型保存在: {dpo_dir}")
        return str(dpo_dir / "final_model")
    
    def _run_evaluation(self, model_path: str) -> Dict[str, Any]:
        """运行评估"""
        eval_dir = self.config.experiment_dir / "evaluation"
        eval_dir.mkdir(exist_ok=True)
        
        # 准备测试数据
        test_data = []
        
        # 从验证集中采样
        with open(self.config.experiment_dir / "data" / "val.json", 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        
        for item in val_data[:self.config.eval_test_size]:
            test_data.append({
                "prompt": item['instruction'],
                "reference": item['output'],
                "style": item.get('style', '未知')
            })
        
        # 创建评估器
        evaluator = ModelEvaluator(model_path)
        
        # 运行各种评估
        all_metrics = {}
        
        # 1. 综合评估
        logger.info("运行综合评估...")
        for style in self.config.styles:
            style_data = [d for d in test_data if d.get('style') == style]
            if style_data:
                metrics = evaluator.evaluate_comprehensive(style_data, style)
                all_metrics[f"{style}_metrics"] = metrics.to_dict()
                
                # 保存风格特定的报告
                evaluator.save_evaluation_report(
                    metrics,
                    str(eval_dir / f"report_{style}"),
                    additional_info={"style": style, "test_size": len(style_data)}
                )
        
        # 2. 人类对齐度评估
        logger.info("评估人类对齐度...")
        alignment_results = evaluator.evaluate_human_alignment(test_data[:50])
        all_metrics['human_alignment'] = alignment_results
        
        # 3. 对比评估（如果有基线模型）
        if self.config.sft_enabled and self.config.dpo_enabled:
            logger.info("运行对比评估...")
            
            # 评估基础模型
            base_evaluator = ModelEvaluator(self.config.base_model)
            base_metrics = base_evaluator.evaluate_comprehensive(test_data[:50], "混合")
            
            # 评估SFT模型
            sft_path = self.results.get('sft_model')
            if sft_path and Path(sft_path).exists():
                sft_evaluator = ModelEvaluator(sft_path)
                sft_metrics = sft_evaluator.evaluate_comprehensive(test_data[:50], "混合")
            else:
                sft_metrics = None
            
            # 当前模型（DPO后）
            current_metrics = evaluator.evaluate_comprehensive(test_data[:50], "混合")
            
            # 对比结果
            comparison = {
                "base_model": base_metrics.to_dict() if base_metrics else None,
                "sft_model": sft_metrics.to_dict() if sft_metrics else None,
                "final_model": current_metrics.to_dict()
            }
            all_metrics['comparison'] = comparison
            
            # 生成对比图表
            self._generate_comparison_chart(comparison, eval_dir / "comparison.png")
        
        # 保存所有评估结果
        with open(eval_dir / "all_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        
        logger.success("评估完成")
        return all_metrics
    
    def _generate_comparison_chart(self, comparison: Dict, save_path: Path):
        """生成对比图表"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        models = []
        metrics_data = {}
        
        if comparison.get('base_model'):
            models.append('Base')
            metrics_data['Base'] = comparison['base_model']
        
        if comparison.get('sft_model'):
            models.append('SFT')
            metrics_data['SFT'] = comparison['sft_model']
        
        if comparison.get('final_model'):
            models.append('DPO')
            metrics_data['DPO'] = comparison['final_model']
        
        # 指标对比
        metric_names = ['perplexity', 'bleu_score', 'distinct_2']
        metric_labels = ['Perplexity\n(lower is better)', 'BLEU Score\n(higher is better)', 'Distinct-2\n(higher is better)']
        
        for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
            ax = axes[idx]
            values = [metrics_data[model].get(metric, 0) for model in models]
            
            colors = ['#ff7f0e', '#2ca02c', '#1f77b4'][:len(models)]
            bars = ax.bar(models, values, color=colors)
            
            ax.set_ylabel('Score')
            ax.set_title(label)
            ax.grid(axis='y', alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}',
                       ha='center', va='bottom')
        
        plt.suptitle('Model Comparison', fontsize=16, y=1.05)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_final_report(self):
        """生成最终报告"""
        report_path = self.config.experiment_dir / "final_report.md"
        
        report = []
        report.append(f"# 后训练实验报告")
        report.append(f"\n## 实验信息")
        report.append(f"- **实验名称**: {self.config.experiment_name}")
        report.append(f"- **基础模型**: {self.config.base_model}")
        report.append(f"- **训练时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"- **风格类型**: {', '.join(self.config.styles)}")
        
        # 数据统计
        if 'data_stats' in self.results:
            report.append(f"\n## 数据统计")
            stats = self.results['data_stats']
            report.append(f"- **总样本数**: {stats['total_samples']}")
            report.append(f"- **训练风格**: {', '.join(stats['styles'])}")
        
        # 训练配置
        report.append(f"\n## 训练配置")
        report.append(f"\n### SFT配置")
        report.append(f"- **启用**: {self.config.sft_enabled}")
        if self.config.sft_enabled:
            report.append(f"- **Epochs**: {self.config.sft_epochs}")
            report.append(f"- **Batch Size**: {self.config.sft_batch_size}")
            report.append(f"- **Learning Rate**: {self.config.sft_learning_rate}")
            report.append(f"- **LoRA**: {self.config.sft_use_lora} (r={self.config.sft_lora_r})")
        
        report.append(f"\n### DPO配置")
        report.append(f"- **启用**: {self.config.dpo_enabled}")
        if self.config.dpo_enabled:
            report.append(f"- **Epochs**: {self.config.dpo_epochs}")
            report.append(f"- **Batch Size**: {self.config.dpo_batch_size}")
            report.append(f"- **Learning Rate**: {self.config.dpo_learning_rate}")
            report.append(f"- **Beta**: {self.config.dpo_beta}")
        
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
                    report.append(f"- **BLEU分数**: {metrics.get('bleu_score', 'N/A'):.3f}")
                    report.append(f"- **Distinct-2**: {metrics.get('distinct_2', 'N/A'):.3f}")
                    report.append(f"- **风格一致性**: {metrics.get('style_consistency', 'N/A'):.3f}")
            
            # 人类对齐度
            if 'human_alignment' in eval_results:
                alignment = eval_results['human_alignment']
                report.append(f"\n### 人类对齐度")
                report.append(f"- **连贯性**: {alignment.get('coherence', 'N/A'):.3f}")
                report.append(f"- **相关性**: {alignment.get('relevance', 'N/A'):.3f}")
                report.append(f"- **创造性**: {alignment.get('creativity', 'N/A'):.3f}")
                report.append(f"- **总体**: {alignment.get('overall', 'N/A'):.3f}")
        
        # 模型路径
        report.append(f"\n## 输出文件")
        if 'sft_model' in self.results:
            report.append(f"- **SFT模型**: `{self.results['sft_model']}`")
        if 'dpo_model' in self.results:
            report.append(f"- **DPO模型**: `{self.results['dpo_model']}`")
        
        # 结论
        report.append(f"\n## 结论")
        report.append(f"后训练流程成功完成。最终模型保存在: `{self.config.experiment_dir}`")
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        # 同时保存JSON格式的结果
        results_path = self.config.experiment_dir / "results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.success(f"最终报告已生成: {report_path}")
    
    def resume(self, from_step: str = "sft"):
        """从指定步骤恢复训练"""
        logger.info(f"从 {from_step} 步骤恢复训练")
        
        if from_step == "data":
            self._prepare_data()
            from_step = "sft"
        
        if from_step == "sft":
            sft_model_path = self._run_sft()
            self.results['sft_model'] = sft_model_path
            from_step = "dpo"
        else:
            # 尝试找到已有的SFT模型
            sft_dir = self.config.experiment_dir / "sft"
            if sft_dir.exists():
                sft_model_path = str(sft_dir / "checkpoint-final")
                self.results['sft_model'] = sft_model_path
            else:
                sft_model_path = self.config.base_model
        
        if from_step == "dpo" and self.config.dpo_enabled:
            dpo_model_path = self._run_dpo(sft_model_path)
            self.results['dpo_model'] = dpo_model_path
            from_step = "eval"
        else:
            # 尝试找到已有的DPO模型
            dpo_dir = self.config.experiment_dir / "dpo"
            if dpo_dir.exists():
                dpo_model_path = str(dpo_dir / "final_model")
                self.results['dpo_model'] = dpo_model_path
            else:
                dpo_model_path = sft_model_path
        
        if from_step == "eval" and self.config.eval_enabled:
            eval_results = self._run_evaluation(dpo_model_path)
            self.results['evaluation'] = eval_results
        
        self._generate_final_report()


def main():
    """主函数"""
    # 创建配置
    config = PostTrainingConfig(
        project_name="novel_rag_test",
        base_model="Qwen/Qwen-1_8B-Chat",
        novel_data_dir="./data/novels",
        styles=["仙侠", "武侠"],
        max_samples_per_style=500,
        sft_epochs=2,
        dpo_epochs=1,
        eval_test_size=50,
        use_wandb=False
    )
    
    # 创建流程管理器
    pipeline = PostTrainingPipeline(config)
    
    # 运行完整流程
    pipeline.run()
    
    # 或者从某个步骤恢复
    # pipeline.resume(from_step="dpo")


if __name__ == "__main__":
    main()