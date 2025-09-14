"""
training/evaluator.py - 模型评估器
实现多维度评估指标
"""

import torch
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict
import jieba
import torch.nn.functional as F
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

class NovelEvaluator:
    """小说生成评估器"""
    
    def __init__(self, config):
        self.config = config
        self.rouge = Rouge()
        self.smoothing = SmoothingFunction()
        
    def evaluate_comprehensive(
        self,
        model,
        tokenizer,
        test_examples,
        save_report=True
    ) -> Dict[str, Any]:
        """全面评估"""
        logger.info("开始综合评估...")
        
        results = {
            "perplexity": [],
            "bleu": [],
            "rouge": {"rouge-1": [], "rouge-2": [], "rouge-l": []},
            "diversity": [],
            "coherence": [],
            "style_consistency": [],
            "creativity": []
        }
        
        max_eval_samples = min(len(test_examples), 100)

        for i, example in enumerate(test_examples[:max_eval_samples]):
            try:
                # 生成文本
                generated = self._generate_text(model, tokenizer, example.instruction)
                
                # 计算各项指标
                results["perplexity"].append(
                    self._calculate_perplexity(model, tokenizer, example.output)
                )
                results["bleu"].append(
                    self._calculate_bleu(generated, example.output)
                )
                
                rouge_scores = self._calculate_rouge(generated, example.output)
                for key in rouge_scores:
                    results["rouge"][key].append(rouge_scores[key])
                
                results["diversity"].append(
                    self._calculate_diversity(generated)
                )
                results["coherence"].append(
                    self._calculate_coherence(generated)
                )
                results["style_consistency"].append(
                    self._calculate_style_consistency(generated, example.style)
                )
                results["creativity"].append(
                    self._calculate_creativity(generated)
                )
                
                # 进度提示
                if (i + 1) % 20 == 0:
                    logger.info(f"已评估 {i + 1}/{max_eval_samples} 个样本")
                    
            except Exception as e:
                logger.warning(f"评估样本 {i} 时出错: {e}")
                continue
        
        # 计算平均值
        final_results = {}
        for key in ["perplexity", "bleu", "diversity", "coherence", "style_consistency", "creativity"]:
            values = results[key]
            final_results[key] = np.mean(values) if values else 0.0
        
        # ROUGE scores
        for rouge_key in ["rouge-1", "rouge-2", "rouge-l"]:
            values = results["rouge"][rouge_key]
            final_results[rouge_key] = np.mean(values) if values else 0.0

        # 生成报告
        if save_report:
            self._generate_report(final_results)
        
        logger.success(f"评估完成: {final_results}")
        
        return final_results
    
    def _generate_text(self, model, tokenizer, prompt):
        """生成文本"""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,  
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return generated
    
    def _calculate_perplexity(self, model, tokenizer, text):
        """计算困惑度"""
        if not text:
            return float('inf')
            
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        encodings = {k: v.to(model.device) for k, v in encodings.items()}
        
        with torch.no_grad():
            outputs = model(**encodings)
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            else:
                # 手动计算loss
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = encodings['input_ids'][..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1)
                )
        
        perplexity = torch.exp(loss).item()
        # 防止inf值
        return min(perplexity, 1e10)
    
    def _calculate_bleu(self, generated, reference):
        """计算BLEU分数"""
        if not generated or not reference:
            return 0.0
            
        gen_tokens = list(jieba.cut(generated))
        ref_tokens = list(jieba.cut(reference))
        
        if not gen_tokens or not ref_tokens:
            return 0.0
        
        return sentence_bleu([ref_tokens], gen_tokens, 
                            smoothing_function=self.smoothing.method1)
    
    def _calculate_rouge(self, generated, reference):
        """计算ROUGE分数"""
        try:
            if not generated or not reference:
                return {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}
                
            scores = self.rouge.get_scores(generated, reference)[0]
            return {
                "rouge-1": scores["rouge-1"]["f"],
                "rouge-2": scores["rouge-2"]["f"],
                "rouge-l": scores["rouge-l"]["f"]
            }
        except Exception as e:
            logger.debug(f"ROUGE计算失败: {e}")
            return {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}
        
    def _calculate_diversity(self, text):
        """计算词汇多样性"""
        tokens = list(jieba.cut(text))
        if not tokens:
            return 0
        
        unique_tokens = set(tokens)
        bigrams = set(zip(tokens[:-1], tokens[1:])) if len(tokens) > 1 else set()
        
        diversity_1 = len(unique_tokens) / len(tokens)
        diversity_2 = len(bigrams) / max(len(tokens) - 1, 1) if len(tokens) > 1 else 0
        
        return (diversity_1 + diversity_2) / 2
    
    def _calculate_coherence(self, text):
        """计算连贯性"""
        sentences = [s.strip() for s in text.split('。') if s.strip()]
        if len(sentences) < 2:
            return 1.0
        
        coherence_scores = []
        for i in range(len(sentences) - 1):
            if sentences[i] and sentences[i+1]:
                # 计算相邻句子的词汇重叠度
                tokens1 = set(jieba.cut(sentences[i]))
                tokens2 = set(jieba.cut(sentences[i+1]))
                
                if tokens1 and tokens2:
                    overlap = len(tokens1 & tokens2) / min(len(tokens1), len(tokens2))
                    coherence_scores.append(overlap)
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def _calculate_style_consistency(self, text, style):
        """计算风格一致性"""
        style_keywords = {
            "仙侠": ["修炼", "灵气", "仙", "道", "法宝", "神通", "飞升", "渡劫"],
            "武侠": ["江湖", "侠", "武功", "内力", "剑", "刀", "掌法", "轻功"],
            "玄幻": ["魔法", "斗气", "异界", "龙", "魔兽", "血脉", "契约", "元素"],
            "都市": ["城市", "公司", "现代", "科技", "都市", "生活", "手机", "汽车"],
            "科幻": ["科技", "未来", "太空", "机器人", "星际", "宇宙", "AI", "基因"]
        }
        
        keywords = style_keywords.get(style, [])
        if not keywords:
            return 0.5
        
        text_lower = text.lower()
        keyword_count = sum(1 for kw in keywords if kw in text_lower)
        
        return min(keyword_count / len(keywords), 1.0)
    
    def _calculate_creativity(self, text):
        """计算创造性（基于句式和用词）"""
        sentences = [s.strip() for s in text.split('。') if s.strip()]
        
        # 句长变化
        sent_lengths = [len(s) for s in sentences if s]
        length_variance = np.var(sent_lengths) if len(sent_lengths) > 1 else 0
        
        # 词汇丰富度
        tokens = list(jieba.cut(text))
        vocab_richness = len(set(tokens)) / max(len(tokens), 1) if tokens else 0
        
        # 修辞手法检测（完整列表）
        rhetorical_markers = [
            "如", "似", "像", "犹如", "仿佛", "宛若", "好像",
            "一般", "似的", "如同", "好似", "恰似"
        ]
        rhetorical_count = sum(1 for marker in rhetorical_markers if marker in text)
        
        # 综合评分
        creativity_score = (
            min(length_variance / 100, 1.0) * 0.3 +  # 句长变化权重
            vocab_richness * 0.4 +                    # 词汇丰富度权重
            min(rhetorical_count / 5, 1.0) * 0.3      # 修辞手法权重
        )
        
        return creativity_score
    
    def _generate_report(self, results: Dict[str, float]):
        """生成评估报告"""
        report_dir = Path(self.config.training.output_dir) / "evaluation"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文本报告
        report_path = report_dir / "evaluation_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("小说生成模型评估报告\n")
            f.write("=" * 50 + "\n\n")
            
            for metric, value in results.items():
                f.write(f"{metric:20s}: {value:.4f}\n")
        
        # 保存JSON格式
        json_path = report_dir / "evaluation_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 生成可视化报告
        try:
            self._plot_results(results, report_dir)
        except Exception as e:
            logger.warning(f"生成可视化报告失败: {e}")
        
        logger.info(f"评估报告已保存到: {report_dir}")
    
    def _plot_results(self, results: Dict[str, float], save_dir: Path):
        """绘制评估结果"""
        plt.figure(figsize=(14, 6))
        
        # 柱状图
        plt.subplot(1, 2, 1)
        metrics = list(results.keys())
        values = list(results.values())
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(metrics)))
        bars = plt.bar(metrics, values, color=colors)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Score')
        plt.title('Evaluation Metrics')
        plt.ylim(0, max(values) * 1.2 if values else 1)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 雷达图
        plt.subplot(1, 2, 2, projection='polar')
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values_normalized = [v / (max(values) if values else 1) for v in values]
        
        angles += angles[:1]
        values_normalized += values_normalized[:1]
        
        plt.plot(angles, values_normalized, 'o-', linewidth=2, color='#1f77b4')
        plt.fill(angles, values_normalized, alpha=0.25, color='#1f77b4')
        plt.xticks(angles[:-1], metrics, size=8)
        plt.title('Metrics Radar Chart')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'evaluation_plots.png', dpi=100, bbox_inches='tight')
        plt.close()