"""
generation/rag_generator.py - 基于RAG的小说生成器（优化版）
支持流式输出和对话历史
"""

import torch
from typing import List, Dict, Any, Optional, Generator, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from queue import Queue
from dataclasses import dataclass, field
from collections import deque
from loguru import logger
import time

from rag.knowledge_base import NovelKnowledgeBase


@dataclass
class GenerationHistory:
    """生成历史记录"""
    role: str  # "user" or "assistant"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class RAGNovelGenerator:
    """基于RAG的小说生成器（增强版）"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        knowledge_base: Optional[NovelKnowledgeBase] = None,
        device: str = None,
        max_history: int = 10  # 最大历史记录数
    ):
        """
        初始化生成器
        
        Args:
            model_name: 生成模型名称
            knowledge_base: 知识库实例
            device: 设备
            max_history: 最大保存的历史记录数
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.knowledge_base = knowledge_base
        self.max_history = max_history
        
        # 历史记录
        self.history: deque[GenerationHistory] = deque(maxlen=max_history * 2)
        
        # 加载模型和分词器
        self._load_model()
    
    def _load_model(self):
        """加载生成模型"""   
        import os

        '''if self.model_name == "Qwen/Qwen2.5-3B-Instruct":
            local_path = "./models/transformers_cache/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
            if os.path.exists(local_path):
                self.model_name = local_path
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                '''
        logger.info(f"加载生成模型: {self.model_name}")
        
        if "Qwen2.5" in self.model_name and os.path.isdir(self.model_name):
            config_path = os.path.join(self.model_name, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                if 'model_type' not in config:
                    config['model_type'] = 'qwen2'
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(config, f, ensure_ascii=False, indent=2)
                        logger.info("已自动修复模型配置")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 根据显存选择加载方式
        if self.device == 'cuda':
            # 尝试4bit量化
            try:
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                logger.info("使用4bit量化加载模型")
                
            except Exception as e:
                logger.warning(f"4bit量化失败: {e}, 尝试普通加载")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float32,
                trust_remote_code=True,
            ).to(self.device)
        
        self.model.eval()
        logger.success("模型加载完成")
    
    def generate_stream(
        self,
        prompt: str,
        style: Optional[str] = None,
        use_rag: bool = True,
        use_history: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1
    ) -> Generator[str, None, None]:
        """
        流式生成小说内容
        
        Args:
            prompt: 生成提示
            style: 风格
            use_rag: 是否使用RAG
            use_history: 是否使用历史记录
            max_new_tokens: 最大生成长度
            temperature: 温度
            top_p: Top-p采样
            top_k: Top-k采样
            repetition_penalty: 重复惩罚
            
        Yields:
            生成的文本片段
        """
        # 保存用户输入到历史
        if use_history:
            self.history.append(GenerationHistory(
                role="user",
                content=prompt,
                metadata={"style": style}
            ))
        
        # RAG增强和历史整合
        if use_rag and self.knowledge_base:
            augmented_prompt = self._augment_prompt_with_history(
                prompt, style, use_history
            )
        else:
            augmented_prompt = self._format_prompt_with_history(
                prompt, style, use_history
            )
        
        # 分词
        inputs = self.tokenizer(
            augmented_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # 创建流式输出器
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=60.0
        )
        
        # 生成参数
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=streamer
        )
        
        # 在后台线程中运行生成
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # 收集完整的生成内容（用于保存到历史）
        generated_text = []
        
        # 流式输出
        for new_text in streamer:
            if new_text:
                generated_text.append(new_text)
                yield new_text
        
        # 等待生成完成
        thread.join()
        
        # 保存助手回复到历史
        if use_history:
            full_response = ''.join(generated_text)
            self.history.append(GenerationHistory(
                role="assistant",
                content=full_response,
                metadata={"style": style}
            ))
    
    def generate(
        self,
        prompt: str,
        style: Optional[str] = None,
        use_rag: bool = True,
        use_history: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1
    ) -> str:
        """
        生成小说内容（非流式，兼容旧接口）
        
        Args:
            prompt: 生成提示
            style: 风格
            use_rag: 是否使用RAG
            use_history: 是否使用历史记录
            其他参数同generate_stream
            
        Returns:
            完整的生成文本
        """
        # 收集流式输出
        generated_parts = []
        for text in self.generate_stream(
            prompt=prompt,
            style=style,
            use_rag=use_rag,
            use_history=use_history,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        ):
            generated_parts.append(text)
        
        return ''.join(generated_parts)
    
    def _augment_prompt_with_history(
        self,
        prompt: str,
        style: Optional[str] = None,
        use_history: bool = True
    ) -> str:
        """
        使用RAG和历史记录增强提示
        
        Args:
            prompt: 原始提示
            style: 风格
            use_history: 是否包含历史
            
        Returns:
            增强后的提示
        """
        # 检索相关内容
        search_results = self.knowledge_base.search(prompt, top_k=3, style=style)
        
        # 构建增强提示
        context_texts = []
        for result in search_results:
            # 截取相关片段
            content = result['content'][:300]
            context_texts.append(f"参考片段：{content}")
        
        # 构建历史上下文
        history_text = ""
        if use_history and len(self.history) > 1:  # 至少有之前的交互
            history_text = self._format_history()
        
        # 组合提示
        if context_texts:
            augmented = f"""你是一位专业的{style if style else ''}小说作家。

参考内容：
{chr(10).join(context_texts)}

{history_text}

请基于以上参考内容的风格和写作手法，创作以下内容：
{prompt}

要求：
1. 保持与参考内容风格一致
2. 与之前的内容保持连贯性
3. 情节连贯，逻辑合理
4. 语言生动，富有感染力

创作内容："""
        else:
            augmented = self._format_prompt_with_history(prompt, style, use_history)
        
        return augmented
    
    def _format_prompt_with_history(
        self,
        prompt: str,
        style: Optional[str] = None,
        use_history: bool = True
    ) -> str:
        """
        格式化提示（包含历史，无RAG）
        
        Args:
            prompt: 原始提示
            style: 风格
            use_history: 是否包含历史
            
        Returns:
            格式化的提示
        """
        # 构建历史上下文
        history_text = ""
        if use_history and len(self.history) > 1:
            history_text = self._format_history()
        
        return f"""你是一位专业的{style if style else ''}小说作家。

{history_text}

请创作以下内容：
{prompt}

要求：
1. 符合{style if style else '小说'}风格
2. 与之前的内容保持连贯
3. 情节生动有趣
4. 人物形象鲜明

创作内容："""
    
    def _format_history(self, max_history_chars: int = 1500) -> str:
        """
        格式化历史记录
        
        Args:
            max_history_chars: 最大历史字符数
            
        Returns:
            格式化的历史文本
        """
        if not self.history:
            return ""
        
        history_parts = []
        total_chars = 0
        
        # 从最近的历史开始（倒序）
        for item in reversed(self.history):
            # 跳过当前的用户输入（已在prompt中）
            if item == self.history[-1] and item.role == "user":
                continue
            
            # 格式化历史项
            if item.role == "user":
                formatted = f"用户要求：{item.content[:200]}..."
            else:
                formatted = f"已创作内容：{item.content[:300]}..."
            
            # 检查长度限制
            if total_chars + len(formatted) > max_history_chars:
                break
            
            history_parts.append(formatted)
            total_chars += len(formatted)
        
        if history_parts:
            # 反转回正确的时间顺序
            history_parts.reverse()
            return "之前的创作历史：\n" + "\n".join(history_parts) + "\n"
        
        return ""
    
    def clear_history(self):
        """清空历史记录"""
        self.history.clear()
        logger.info("历史记录已清空")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        获取历史记录
        
        Returns:
            历史记录列表
        """
        return [
            {
                "role": item.role,
                "content": item.content,
                "metadata": item.metadata,
                "timestamp": item.timestamp
            }
            for item in self.history
        ]
    
    def generate_continuation_stream(
        self,
        context: str,
        style: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        流式续写小说
        
        Args:
            context: 上下文
            style: 风格
            **kwargs: 生成参数
            
        Yields:
            续写内容片段
        """
        prompt = f"续写下面的内容：\n{context}\n\n续写："
        yield from self.generate_stream(prompt, style, **kwargs)
    
    def generate_chapter_stream(
        self,
        chapter_title: str,
        chapter_outline: str,
        style: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        流式生成章节
        
        Args:
            chapter_title: 章节标题
            chapter_outline: 章节大纲
            style: 风格
            **kwargs: 生成参数
            
        Yields:
            章节内容片段
        """
        prompt = f"""章节标题：{chapter_title}
章节大纲：{chapter_outline}

请根据以上信息创作完整的章节内容。"""
        
        yield from self.generate_stream(prompt, style, **kwargs)