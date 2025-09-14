"""
train/data_processor.py
主要改进：
1. 配置外部化
2. 上下文利用
3. 批量生成优化
4. 参数化配置
5. 质量评估增强
"""

import os
import json
import yaml
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import hashlib
from loguru import logger
import jieba
import jieba.analyse
import numpy as np
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Protocol, TypedDict
from typing_extensions import NotRequired

class StyleTemplate(TypedDict):
    """风格模板类型"""
    instruction: str
    keywords: List[str]

class QualityCriteria(TypedDict):
    """质量标准类型"""
    min_length: int
    max_length: int
    min_punctuation_ratio: float
    max_punctuation_ratio: float
    preferred_sentence_length: Tuple[int, int]
    quality_weights: Dict[str, float]

@dataclass
class TrainingExample:
    """训练样本"""
    instruction: str
    input: str
    output: str
    style: str
    quality_score: float = 1.0
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def to_prompt(self) -> str:
        """生成训练提示"""
        if self.input:
            return f"{self.instruction}\n输入：{self.input}\n输出："
        return f"{self.instruction}\n输出："


@dataclass
class PreferenceData:
    """偏好数据（用于DPO/RLHF）"""
    prompt: str
    chosen: str
    rejected: str
    style: str
    chosen_score: float = 1.0
    rejected_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


class DataProcessorConfig:
    """数据处理器配置类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径，支持JSON或YAML格式
        """
        self.config_path = config_path or "config/data_processor_config.yaml"
        self.config:Dict[str,Any] = self._load_config()
        
        # 加载各项配置
        self.style_templates: Dict[str, List[StyleTemplate]] = self.config.get("style_templates", {})
        self.quality_criteria: QualityCriteria = self.config.get("quality_criteria", {})
        self.processing_params: Dict[str, Any] = self.config.get("processing_params", {})
        self.augmentation_params: Dict[str, Any] = self.config.get("augmentation_params", {})
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        config_path = Path(self.config_path)
        
        # 如果配置文件不存在，使用默认配置
        if not config_path.exists():
            logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
            return self._get_default_config()
        
        # 根据扩展名选择加载方式
        if config_path.suffix == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            logger.error(f"不支持的配置文件格式: {config_path.suffix}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "style_templates": {
                "仙侠": [
                    {"instruction": "描写一个修仙者突破境界的场景", "keywords": ["突破", "境界", "灵气"]},
                    {"instruction": "创作一段仙界的环境描写", "keywords": ["仙界", "灵气", "飘渺"]}
                ],
                "武侠": [
                    {"instruction": "描写一场江湖恩怨的对决", "keywords": ["江湖", "恩怨", "决斗"]},
                    {"instruction": "创作一个侠客行走江湖的场景", "keywords": ["侠客", "江湖", "行侠仗义"]}
                ]
            },
            "quality_criteria": {
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
            "processing_params": {
                "target_chunk_length": 500,
                "max_samples_per_novel": 1000,
                "use_context": True,
                "context_length": 200
            },
            "augmentation_params": {
                "augment_ratio": 0.3,
                "instruction_rewrite_prob": 0.5,
                "truncate_prob": 0.3,
                "context_add_prob": 0.3
            }
        }
    
    def save_config(self, path: str = None):
        """保存配置到文件"""
        save_path = Path(path or self.config_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.suffix == '.json':
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        elif save_path.suffix in ['.yaml', '.yml']:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
        
        logger.info(f"配置已保存到: {save_path}")


class ImprovedDataProcessor:
    """改进版数据处理器"""

    def __init__(self, config_path: str = None):
        """
        初始化数据处理器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = DataProcessorConfig(config_path)
        
        # 初始化jieba
        self._init_jieba()
        
        # 数据统计
        self.stats = defaultdict(int)
        
        # 初始化质量评估器（可选：使用预训练模型）
        self.quality_evaluator = None
        self._init_quality_evaluator()

    def _init_jieba(self):
        """初始化jieba分词"""
        # 从配置加载自定义词典
        custom_words = self.config.config.get("custom_words", [
            '修炼', '突破', '灵气', '真元', '法力', '神识', '元神',
            '剑意', '刀意', '道心', '心魔', '天劫', '飞升',
            '功法', '秘籍', '法宝', '灵宝', '仙器', '神器'
        ])
        for word in custom_words:
            jieba.add_word(word)
    
    def _init_quality_evaluator(self):
        """初始化质量评估器（可选）"""
        use_model_evaluator = self.config.config.get("use_model_evaluator", False)
        
        if use_model_evaluator:
            try:
                # 可以使用轻量级分类器或语言模型困惑度
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                
                model_name = self.config.config.get("evaluator_model", "bert-base-chinese")
                self.quality_evaluator = {
                    "tokenizer": AutoTokenizer.from_pretrained(model_name),
                    "model": AutoModelForSequenceClassification.from_pretrained(model_name)
                }
                logger.info(f"使用模型评估器: {model_name}")
            except Exception as e:
                logger.warning(f"模型评估器加载失败，使用规则评估: {e}")
                self.quality_evaluator = None

    def process_novel_to_training_data(
            self,
            novel_path: str,
            style: str,
            max_samples: Optional[int] = None,
            use_context: Optional[bool] = None
    ) -> List[TrainingExample]:
        """
        将小说转换为训练数据
        
        Args:
            novel_path: 小说文件路径
            style: 风格类型
            max_samples: 最大样本数
            use_context: 是否使用上下文
        """
        examples = []
        
        # 从配置获取参数
        max_samples = max_samples or self.config.processing_params.get("max_samples_per_novel", 1000)
        use_context = use_context if use_context is not None else self.config.processing_params.get("use_context", True)
        context_length = self.config.processing_params.get("context_length", 200)

        try:
            # 读取小说内容
            with open(novel_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 智能分割章节和段落
            chunks = self._smart_split_text(
                content, 
                self.config.processing_params.get("target_chunk_length", 500)
            )

            # 获取风格模板
            templates = self.config.style_templates.get(style, [])
            if not templates:
                logger.warning(f"未找到风格 {style} 的模板，使用默认模板")
                templates = [{"instruction": f"创作一段{style}风格的内容", "keywords": []}]

            # 处理每个文本块
            for i, chunk in enumerate(chunks):
                if i >= max_samples:
                    break

                # 评估文本质量
                quality_score = self._evaluate_text_quality_enhanced(chunk, style)

                # 只保留高质量文本
                if quality_score < 0.5:
                    self.stats["low_quality_filtered"] += 1
                    continue

                # 准备上下文（使用前一个块）
                context = ""
                if use_context and i > 0:
                    prev_chunk = chunks[i-1]
                    # 截取上下文到指定长度
                    context = prev_chunk[-context_length:] if len(prev_chunk) > context_length else prev_chunk
                    context = self._clean_text(context)

                # 为不同类型的文本选择合适的指令
                instruction_data = self._select_instruction_for_text(chunk, templates)

                # 创建训练样本
                example = TrainingExample(
                    instruction=instruction_data["instruction"],
                    input=context,  # 使用上下文
                    output=self._clean_text(chunk),
                    style=style,
                    quality_score=quality_score,
                    source=Path(novel_path).stem,
                    metadata={
                        "chunk_index": i,
                        "length": len(chunk),
                        "has_context": bool(context),
                        "keywords": instruction_data.get("keywords", []),
                        "type": self._detect_text_type(chunk)
                    }
                )
                examples.append(example)
                self.stats["samples_created"] += 1

            logger.info(f"从《{Path(novel_path).stem}》提取了 {len(examples)} 个训练样本")

        except Exception as e:
            logger.error(f"处理小说失败 {novel_path}: {e}")

        return examples

    def _evaluate_text_quality_enhanced(self, text: str, style: str) -> float:
        """
        增强版文本质量评估
        
        结合规则评估和模型评估（如果可用）
        """
        # 基础规则评估
        rule_score = self._evaluate_text_quality_rules(text, style)
        
        # 如果有模型评估器，结合模型评分
        if self.quality_evaluator:
            try:
                model_score = self._evaluate_text_quality_model(text)
                # 加权平均
                final_score = 0.7 * rule_score + 0.3 * model_score
            except Exception as e:
                logger.debug(f"模型评估失败，使用规则评分: {e}")
                final_score = rule_score
        else:
            final_score = rule_score
        
        return min(max(final_score, 0.1), 1.0)

    def _evaluate_text_quality_rules(self, text: str, style: str) -> float:
        """基于规则的文本质量评估"""
        scores = {}
        weights = self.config.quality_criteria.get("quality_weights", {})
        
        # 长度评分
        length = len(text)
        min_len = self.config.quality_criteria["min_length"]
        max_len = self.config.quality_criteria["max_length"]
        
        if length < min_len:
            scores["length"] = 0.3
        elif length > max_len:
            scores["length"] = 0.7
        else:
            # 线性插值
            scores["length"] = 0.7 + 0.3 * (length - min_len) / (max_len - min_len)
        
        # 标点符号评分
        punctuation = len([c for c in text if c in '，。！？；：""''…'])
        punct_ratio = punctuation / max(length, 1)
        min_punct = self.config.quality_criteria["min_punctuation_ratio"]
        max_punct = self.config.quality_criteria["max_punctuation_ratio"]
        
        if punct_ratio < min_punct:
            scores["punctuation"] = 0.6
        elif punct_ratio > max_punct:
            scores["punctuation"] = 0.8
        else:
            scores["punctuation"] = 1.0
        
        # 对话评分
        dialogue_markers = ['说道', '道：', '问道', '笑道', '冷道', '喝道']
        dialogue_count = sum(text.count(marker) for marker in dialogue_markers)
        scores["dialogue"] = min(1.0, 0.7 + dialogue_count * 0.1)
        
        # 风格关键词评分
        style_keywords = []
        for template in self.config.style_templates.get(style, []):
            style_keywords.extend(template.get("keywords", []))
        
        keyword_count = sum(text.count(kw) for kw in style_keywords)
        scores["keywords"] = min(1.0, 0.6 + keyword_count * 0.1)
        
        # 句子多样性评分
        sentences = re.split(r'[。！？]', text)
        if len(sentences) > 1:
            sent_lengths = [len(s) for s in sentences if s]
            length_variance = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
            scores["sentence_variety"] = min(1.0, 0.5 + length_variance / 20)
        else:
            scores["sentence_variety"] = 0.5
        
        # 重复性检测
        if self._has_repetition(text):
            scores["repetition"] = 0.3
        else:
            scores["repetition"] = 1.0
        
        # 加权计算最终分数
        final_score = 0
        total_weight = 0
        
        for key, score in scores.items():
            weight = weights.get(key, 1.0 / len(scores))
            final_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            final_score /= total_weight
        
        return final_score

    def _evaluate_text_quality_model(self, text: str) -> float:
        """使用模型评估文本质量"""
        if not self.quality_evaluator:
            return 0.5
        
        tokenizer = self.quality_evaluator["tokenizer"]
        model = self.quality_evaluator["model"]
        
        # 编码文本
        inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        
        # 获取模型预测
        with torch.no_grad():
            outputs = model(**inputs)
            # 假设模型输出是质量分数的logits
            score = torch.sigmoid(outputs.logits[0, 0]).item()
        
        return score

    def create_preference_data_batch(
            self,
            examples: List[TrainingExample],
            model,
            tokenizer,
            max_preference_pairs: int = 100,
            batch_size: int = 4,
            num_candidates: int = 4
    ) -> List[PreferenceData]:
        """
        批量创建偏好数据
        
        Args:
            examples: 训练样本列表
            model: 生成模型
            tokenizer: 分词器
            max_preference_pairs: 最大偏好对数量
            batch_size: 批处理大小
            num_candidates: 每个prompt生成的候选数量
        """
        preference_data = []
        
        # 限制处理数量
        examples_to_process = examples[:max_preference_pairs]
        
        logger.info(f"开始批量生成偏好数据，共 {len(examples_to_process)} 个样本")
        
        # 批量处理
        for i in tqdm(range(0, len(examples_to_process), batch_size), desc="生成偏好对"):
            batch_examples = examples_to_process[i:i + batch_size]
            
            # 准备批量prompts
            prompts = [ex.to_prompt() for ex in batch_examples]
            
            # 批量生成候选
            try:
                all_candidates = self._batch_generate_texts(
                    model, tokenizer, prompts, 
                    num_return_sequences=num_candidates,
                    max_length=512
                )
                
                # 处理每个样本的候选
                for j, example in enumerate(batch_examples):
                    candidates = all_candidates[j * num_candidates:(j + 1) * num_candidates]
                    
                    # 评估候选质量
                    scored_candidates = []
                    for candidate in candidates:
                        quality = self._evaluate_text_quality_enhanced(candidate, example.style)
                        scored_candidates.append((candidate, quality))
                    
                    # 添加原始输出
                    scored_candidates.append((example.output, example.quality_score))
                    
                    # 排序并选择最好和最差的
                    scored_candidates.sort(key=lambda x: x[1], reverse=True)
                    
                    if len(scored_candidates) >= 2:
                        chosen = scored_candidates[0][0]
                        rejected = scored_candidates[-1][0]
                        
                        if chosen != rejected:
                            preference_data.append(PreferenceData(
                                prompt=prompts[j],
                                chosen=chosen,
                                rejected=rejected,
                                style=example.style,
                                chosen_score=scored_candidates[0][1],
                                rejected_score=scored_candidates[-1][1],
                                metadata=example.metadata
                            ))
                            
            except Exception as e:
                logger.warning(f"批次生成失败: {e}")
                continue
        
        logger.info(f"生成了 {len(preference_data)} 个偏好对")
        return preference_data

    def _batch_generate_texts(
            self,
            model,
            tokenizer,
            prompts: List[str],
            num_return_sequences: int = 4,
            max_length: int = 512
    ) -> List[str]:
        """
        批量生成文本 (性能优化版)
        
        一次调用 model.generate 生成所有候选，以最大化GPU效率
        """
        # 批量编码
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # 一次性生成所有候选！
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                # 使用不同的采样策略来增加多样性
                temperature=0.8,  # 适中的温度
                top_p=0.95,
                top_k=50,  # 添加top_k以增加多样性控制
                do_sample=True,
                num_return_sequences=num_return_sequences,  # 核心：每个prompt生成多个序列
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # 添加多样性惩罚以确保生成的候选不会太相似
                diversity_penalty=0.5 if hasattr(model.config, 'diversity_penalty') else 0.0,
                repetition_penalty=1.1
            )
        
        # 批量解码
        # outputs shape: [batch_size * num_return_sequences, sequence_length]
        generated_texts = tokenizer.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return generated_texts

    def _smart_split_text(self, text: str, target_length: int = 500) -> List[str]:
        """智能分割文本"""
        chunks = []

        # 首先按章节分割
        chapter_pattern = r'第[零一二三四五六七八九十百千万\d]+章.*?\n'
        chapters = re.split(chapter_pattern, text)

        for chapter in chapters:
            if not chapter.strip():
                continue

            # 按段落分割
            paragraphs = chapter.split('\n\n')

            current_chunk = []
            current_length = 0

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                para_length = len(para)

                # 如果单个段落太长，进一步分割
                if para_length > target_length * 2:
                    sentences = re.split(r'[。！？]', para)
                    for sent in sentences:
                        if sent:
                            chunks.append(sent + '。')

                # 如果当前块太长，保存并开始新块
                elif current_length + para_length > target_length:
                    if current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [para]
                    current_length = para_length

                else:
                    current_chunk.append(para)
                    current_length += para_length

            # 保存最后一个块
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))

        return chunks


    # 在 ImprovedDataProcessor 类中添加（约第600行后）

def process_novels_parallel(
    self,
    novel_paths: List[str],
    styles: Union[str, List[str], Dict[str, str]],
    max_workers: Optional[int] = None,
    max_samples_per_novel: Optional[int] = None
) -> List[TrainingExample]:
    """
    并行处理多个小说文件
    
    Args:
        novel_paths: 小说文件路径列表
        styles: 风格（字符串、列表或路径到风格的映射）
        max_workers: 最大工作线程数
        max_samples_per_novel: 每本小说的最大样本数
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # 处理风格参数
    if isinstance(styles, str):
        style_map = {path: styles for path in novel_paths}
    elif isinstance(styles, list):
        style_map = dict(zip(novel_paths, styles))
    else:
        style_map = styles
    
    # 设置工作线程数
    max_workers = max_workers or self.config.processing_params.get("parallel_workers", 4)
    
    all_examples = []
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_novel = {
            executor.submit(
                self.process_novel_to_training_data,
                novel_path,
                style_map.get(novel_path, "玄幻"),
                max_samples_per_novel
            ): novel_path
            for novel_path in novel_paths
        }
        
        # 显示进度条
        with tqdm(total=len(novel_paths), desc="处理小说") as pbar:
            for future in as_completed(future_to_novel):
                novel_path = future_to_novel[future]
                try:
                    examples = future.result()
                    all_examples.extend(examples)
                    logger.info(f"成功处理: {Path(novel_path).stem}, 获得 {len(examples)} 个样本")
                except Exception as e:
                    logger.error(f"处理失败 {novel_path}: {e}")
                finally:
                    pbar.update(1)
    
    logger.success(f"并行处理完成，共获得 {len(all_examples)} 个训练样本")
    return all_examples

    def _has_repetition(self, text: str, threshold: int = 3) -> bool:
        """检测文本是否有过多重复"""
        words = list(jieba.cut(text))
        trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]

        trigram_counts = defaultdict(int)
        for trigram in trigrams:
            trigram_counts[trigram] += 1

        return any(count > threshold for count in trigram_counts.values())

    def _select_instruction_for_text(
            self,
            text: str,
            templates: List[Dict[str, str]]
    ) -> Dict[str, str]:
        """根据文本内容选择合适的指令"""
        if not templates:
            return {"instruction": "创作一段小说内容", "keywords": []}
        
        # 提取文本关键词
        text_keywords = jieba.analyse.extract_tags(text, topK=10)

        # 计算每个模板的匹配度
        best_match = templates[0]
        best_score = 0

        for template in templates:
            template_keywords = template.get("keywords", [])
            score = len(set(text_keywords) & set(template_keywords))

            for kw in template_keywords:
                if kw in text:
                    score += 1

            if score > best_score:
                best_score = score
                best_match = template

        if best_score == 0:
            best_match = random.choice(templates)

        return best_match

    def _detect_text_type(self, text: str) -> str:
        """检测文本类型"""
        if any(marker in text for marker in ['说道', '道：', '问道']):
            return "dialogue"
        elif any(marker in text for marker in ['突然', '忽然', '猛地', '一声']):
            return "action"
        elif len(text) > 200 and '的' in text and '了' in text:
            return "description"
        else:
            return "narrative"

    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 1. 标准化换行
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # 2. 移除段落首尾空白
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        text = '\n\n'.join(paragraphs)
        
        # 3. 移除多余空格
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 4. 移除特殊字符
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\n\s，。！？；：""''（）《》【】…—～·]', '', text)
        
        return text.strip()

    # 保留原有的其他方法...
    def augment_data(
            self,
            examples: List[TrainingExample],
            augment_ratio: Optional[float] = None
    ) -> List[TrainingExample]:
        """数据增强"""
        augment_ratio = augment_ratio or self.config.augmentation_params.get("augment_ratio", 0.3)
        augmented = []

        for example in examples:
            augmented.append(example)

            if random.random() < augment_ratio:
                aug_params = self.config.augmentation_params
                
                if random.random() < aug_params.get("instruction_rewrite_prob", 0.5):
                    augmented.append(self._augment_instruction(example))

                if random.random() < aug_params.get("truncate_prob", 0.3) and len(example.output) > 200:
                    augmented.append(self._augment_truncate(example))

                if random.random() < aug_params.get("context_add_prob", 0.3):
                    augmented.append(self._augment_context(example))

        logger.info(f"数据增强: {len(examples)} -> {len(augmented)}")
        return augmented

    def _augment_instruction(self, example: TrainingExample) -> TrainingExample:
        """改写指令"""
        instruction_variants = self.config.config.get("instruction_variants", {
            "描写": ["创作", "写一段", "生成"],
            "创作": ["描写", "写出", "构思"],
            "写": ["创作", "描述", "生成"]
        })

        new_instruction = example.instruction
        for key, values in instruction_variants.items():
            if key in new_instruction:
                new_instruction = new_instruction.replace(key, random.choice(values))
                break

        return TrainingExample(
            instruction=new_instruction,
            input=example.input,
            output=example.output,
            style=example.style,
            quality_score=example.quality_score * 0.95,
            source=example.source + "_aug",
            metadata={**example.metadata, "augment_type": "instruction"}
        )

    def _augment_truncate(self, example: TrainingExample) -> TrainingExample:
        """截断输出用于续写训练"""
        truncate_point = random.randint(100, len(example.output) - 50)
        text = example.output[:truncate_point]
        
        last_period = max(text.rfind('。'), text.rfind('！'), text.rfind('？'))
        if last_period > 50:
            text = text[:last_period + 1]

        return TrainingExample(
            instruction=f"续写下面的内容：\n{text}",
            input="",
            output=example.output[len(text):],
            style=example.style,
            quality_score=example.quality_score * 0.9,
            source=example.source + "_aug",
            metadata={**example.metadata, "augment_type": "continuation"}
        )

    def _augment_context(self, example: TrainingExample) -> TrainingExample:
        """添加上下文"""
        context_templates = self.config.config.get("context_templates", [
            f"在{example.style}风格的小说中，",
            f"按照{example.style}的写作风格，",
            f"参考经典{example.style}小说的风格，"
        ])

        return TrainingExample(
            instruction=random.choice(context_templates) + example.instruction,
            input=example.input,
            output=example.output,
            style=example.style,
            quality_score=example.quality_score,
            source=example.source + "_aug",
            metadata={**example.metadata, "augment_type": "context"}
        )

    def save_dataset(
            self,
            examples: List[TrainingExample],
            save_path: str,
            split_ratio: float = 0.9
    ):
        """保存数据集"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # 打乱数据
        random.shuffle(examples)

        # 分割训练集和验证集
        split_point = int(len(examples) * split_ratio)
        train_data = examples[:split_point]
        val_data = examples[split_point:]

        # 保存训练集
        train_file = save_path / "train.json"
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump([ex.to_dict() for ex in train_data], f, ensure_ascii=False, indent=2)

        # 保存验证集
        val_file = save_path / "val.json"
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump([ex.to_dict() for ex in val_data], f, ensure_ascii=False, indent=2)

        # 保存数据集统计信息
        stats = {
            "total_examples": len(examples),
            "train_examples": len(train_data),
            "val_examples": len(val_data),
            "styles": defaultdict(int),
            "quality_distribution": {
                "high": len([e for e in examples if e.quality_score > 0.8]),
                "medium": len([e for e in examples if 0.5 <= e.quality_score <= 0.8]),
                "low": len([e for e in examples if e.quality_score < 0.5])
            },
            "sources": defaultdict(int),
            "avg_output_length": np.mean([len(e.output) for e in examples]),
            "context_usage": len([e for e in examples if e.metadata.get("has_context", False)])
        }

        for ex in examples:
            stats["styles"][ex.style] += 1
            stats["sources"][ex.source] += 1

        stats_file = save_path / "stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.success(f"数据集已保存到 {save_path}")
        logger.info(f"训练集: {len(train_data)} 样本")
        logger.info(f"验证集: {len(val_data)} 样本")
        logger.info(f"使用上下文: {stats['context_usage']} 样本")
        logger.info(f"风格分布: {dict(stats['styles'])}")
        logger.info(f"质量分布: {stats['quality_distribution']}")

    def load_dataset(self, dataset_path: str) -> Tuple[List[TrainingExample], List[TrainingExample]]:
        """加载数据集"""
        dataset_path = Path(dataset_path)

        # 加载训练集
        train_file = dataset_path / "train.json"
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        train_examples = [TrainingExample(**d) for d in train_data]

        # 加载验证集
        val_file = dataset_path / "val.json"
        with open(val_file, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        val_examples = [TrainingExample(**d) for d in val_data]

        logger.info(f"加载数据集: 训练集 {len(train_examples)} 样本, 验证集 {len(val_examples)} 样本")

        return train_examples, val_examples


# ========================================
# 配置文件示例: config/data_processor_config.yaml
# ========================================

EXAMPLE_CONFIG_YAML = """
# 数据处理器配置文件
# config/data_processor_config.yaml

# 风格模板配置
style_templates:
  仙侠:
    - instruction: "描写一个修仙者突破境界的场景"
      keywords: ["突破", "境界", "灵气", "修为", "瓶颈"]
    - instruction: "创作一段仙界的环境描写"
      keywords: ["仙界", "灵气", "飘渺", "仙山", "云雾"]
    - instruction: "写一个炼丹或炼器的过程"
      keywords: ["炼丹", "炼器", "火候", "丹炉", "法宝"]
    - instruction: "描述一场修仙者之间的战斗"
      keywords: ["法术", "飞剑", "神通", "斗法", "真元"]
      
  武侠:
    - instruction: "描写一场江湖恩怨的对决"
      keywords: ["江湖", "恩怨", "决斗", "比武", "仇恨"]
    - instruction: "创作一个侠客行走江湖的场景"
      keywords: ["侠客", "江湖", "行侠仗义", "大侠", "义气"]
    - instruction: "写一段武功秘籍的描述"
      keywords: ["秘籍", "武功", "内力", "招式", "心法"]
      
  玄幻:
    - instruction: "描写主角觉醒特殊体质的场景"
      keywords: ["觉醒", "体质", "天赋", "血脉", "力量"]
    - instruction: "创作一个异世界的种族描述"
      keywords: ["种族", "异界", "文明", "魔族", "精灵"]
      
  都市:
    - instruction: "描写一个都市异能者的日常"
      keywords: ["异能", "都市", "隐藏", "现代", "超能力"]
    - instruction: "创作一段商业竞争的场景"
      keywords: ["商业", "竞争", "谈判", "公司", "商战"]
      
  科幻:
    - instruction: "描写一个未来世界的场景"
      keywords: ["未来", "科技", "文明", "星际", "AI"]
    - instruction: "创作一段星际旅行的过程"
      keywords: ["星际", "宇宙", "航行", "飞船", "星球"]

# 质量评估标准
quality_criteria:
  min_length: 50
  max_length: 2000
  min_punctuation_ratio: 0.03
  max_punctuation_ratio: 0.15
  preferred_sentence_length: [10, 50]
  
  # 各项评分权重
  quality_weights:
    length: 0.2
    punctuation: 0.15
    dialogue: 0.15
    keywords: 0.2
    sentence_variety: 0.15
    repetition: 0.15

# 处理参数
processing_params:
  target_chunk_length: 500
  max_samples_per_novel: 1000
  use_context: true  # 启用上下文
  context_length: 200  # 上下文长度

# 增强参数
augmentation_params:
  augment_ratio: 0.3
  instruction_rewrite_prob: 0.5
  truncate_prob: 0.3
  context_add_prob: 0.3

# 自定义词典
custom_words:
  - 修炼
  - 突破
  - 灵气
  - 真元
  - 法力
  - 神识
  - 元神
  - 剑意
  - 刀意
  - 道心
  - 心魔
  - 天劫
  - 飞升

# 指令改写词典
instruction_variants:
  描写: ["创作", "写一段", "生成", "构思"]
  创作: ["描写", "写出", "构思", "编写"]
  写: ["创作", "描述", "生成", "撰写"]

# 上下文模板
context_templates:
  - "在{style}风格的小说中，"
  - "按照{style}的写作风格，"
  - "参考经典{style}小说的风格，"
  - "以{style}小说的笔法，"

# 模型评估器配置（可选）
use_model_evaluator: false  # 是否使用模型评估
evaluator_model: "bert-base-chinese"  # 评估模型名称
"""


# ========================================
# 使用示例
# ========================================

def example_usage():
    """使用示例"""
    from pathlib import Path
    
    # 1. 创建配置文件
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "data_processor_config.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(EXAMPLE_CONFIG_YAML)
    
    print(f"配置文件已创建: {config_file}")
    
    # 2. 初始化改进版数据处理器
    processor = ImprovedDataProcessor(config_path=str(config_file))
    
    # 3. 处理小说数据（使用上下文）
    novel_path = "data/novels/仙侠/example.txt"
    examples = processor.process_novel_to_training_data(
        novel_path=novel_path,
        style="仙侠",
        max_samples=500,
        use_context=True  # 启用上下文
    )
    
    print(f"生成了 {len(examples)} 个训练样本")
    
    # 检查上下文使用情况
    context_count = sum(1 for ex in examples if ex.input)
    print(f"其中 {context_count} 个样本包含上下文")
    
    # 4. 批量生成偏好数据（如果有模型）
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        preference_data = processor.create_preference_data_batch(
            examples=examples,
            model=model,
            tokenizer=tokenizer,
            max_preference_pairs=50,  # 明确的数量限制
            batch_size=4,  # 批处理大小
            num_candidates=4  # 每个prompt的候选数
        )
        
        print(f"批量生成了 {len(preference_data)} 个偏好对")
        
    except ImportError:
        print("跳过偏好数据生成（需要安装transformers）")
    
    # 5. 数据增强
    augmented_examples = processor.augment_data(examples)
    print(f"数据增强后: {len(augmented_examples)} 个样本")
    
    # 6. 保存数据集
    processor.save_dataset(augmented_examples, "data/datasets/enhanced")
    
    # 7. 查看统计信息
    print("\n处理统计:")
    for key, value in processor.stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    example_usage()