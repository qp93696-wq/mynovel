"""
models/model_loader.py 
"""

import os
import json
from pathlib import Path
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from typing import Optional, Dict, Any, List, Union, Iterator
from threading import Thread


class ModelLoader:
    """Qwen2.5模型加载器 - 使用Transformers库"""
    
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = True,
        attn_implementation: Optional[str] = None,
        max_memory: Optional[Dict[int, str]] = None,
        cache_dir: Optional[str] = "./models/transformers_cache",
        verbose: bool = False
    ):
        """
        初始化Qwen2.5模型加载器
        
        Args:
            model_name_or_path: HuggingFace模型ID或本地路径
            device: 设备类型 ('cuda', 'cpu', 'mps' 或 None自动选择)
            dtype: 数据类型 (torch.float16, torch.bfloat16, torch.float32 或 None自动选择)
            load_in_8bit: 是否使用8位量化加载
            load_in_4bit: 是否使用4位量化加载
            trust_remote_code: 是否信任远程代码
            attn_implementation: 是否使用Flash Attention
            max_memory: GPU内存分配字典
            cache_dir: 模型缓存目录
            verbose: 是否输出详细信息
        """
        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = attn_implementation
        self.verbose = verbose
        
        # 设备配置
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
            
        # 数据类型配置
        if dtype is None:
            if self.device == "cuda":
                self.dtype = torch.float16  # GPU默认使用fp16
            else:
                self.dtype = torch.float32  # CPU使用fp32
        else:
            self.dtype = dtype
            
        # 量化配置
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.max_memory = max_memory
        
        # 模型和分词器
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
    
    def load_model(self):
        """加载Qwen2.5模型和分词器"""

        import os
    
        # 检查本地缓存
        if "Qwen2.5-3B-Instruct" in self.model_name_or_path:
            # 构建本地路径
            local_path = os.path.join(
                self.cache_dir,
                "models--Qwen--Qwen2.5-3B-Instruct",
                "snapshots",
                "aa8e72537993ba99e69dfaafa59ed015b17504d1"
            )
            
            # 检查路径是否存在
            if os.path.exists(local_path):
                logger.info(f"发现本地模型，使用: {local_path}")
                self.model_name_or_path = local_path
                # 设置离线模式
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                os.environ['HF_DATASETS_OFFLINE'] = '1'
        logger.info(f"正在加载模型: {self.model_name_or_path}")
        logger.info(f"设备: {self.device}, 数据类型: {self.dtype}")
        
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.trust_remote_code,
                cache_dir=self.cache_dir
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 配置模型加载参数
            model_kwargs = {
                "trust_remote_code": self.trust_remote_code,
                "cache_dir": self.cache_dir,
                "torch_dtype": self.dtype,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            # 量化配置
            if self.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
                logger.info("使用8位量化加载模型")
            elif self.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
                logger.info("使用4位量化加载模型")
            
            # Flash Attention配置
            if self.attn_implementation:
                model_kwargs["attn_implementation"] = self.attn_implementation
                logger.info(f"使用Attention实现: {self.attn_implementation}")
            
            # 内存配置
            if self.max_memory:
                model_kwargs["max_memory"] = self.max_memory
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                **model_kwargs
            )
            
            # 如果不使用device_map="auto"，手动移动到设备
            if self.device != "cuda" or "device_map" not in model_kwargs:
                self.model = self.model.to(self.device)
            
            # 设置为评估模式
            self.model.eval()
            
            logger.success("模型加载成功")
            self._print_model_info()
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self._log_install_help()
            raise
    
    def _print_model_info(self):
        """打印模型信息"""
        if self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"模型参数量: {total_params / 1e9:.2f}B")
            logger.info(f"设备: {self.device}")
            logger.info(f"数据类型: {self.dtype}")
            
            if self.device == "cuda":
                logger.info(f"GPU: {torch.cuda.get_device_name()}")
                logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
    
    def _log_install_help(self):
        """打印安装帮助信息"""
        logger.info("--- Transformers 安装帮助 ---")
        logger.info("基础安装:")
        logger.info("  pip install transformers torch accelerate")
        logger.info("量化支持:")
        logger.info("  pip install bitsandbytes")
        logger.info("Flash Attention (可选，提升性能):")
        logger.info("  pip install flash-attn --no-build-isolation")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        """
        对话接口
        
        Args:
            messages: 对话消息列表
            stream: 是否流式输出
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: nucleus采样参数
            do_sample: 是否采样
            **kwargs: 其他生成参数
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("模型未加载，请先调用load_model()方法")
        
        # 将消息转换为模型输入格式
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码输入
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # 生成参数
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **kwargs
        }
        
        if stream:
            return self._stream_generate(inputs, generation_kwargs)
        else:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # 解码输出，只返回新生成的部分
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            return response.strip()
    
    def _stream_generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, Any]
    ) -> Iterator[str]:
        """流式生成"""
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generation_kwargs["streamer"] = streamer
        
        # 在线程中运行生成
        thread = Thread(
            target=self.model.generate,
            kwargs={**inputs, **generation_kwargs}
        )
        thread.start()
        
        # 逐步输出生成的文本
        for text in streamer:
            yield text
        
        thread.join()
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "你是一位乐于助人的AI助手。",
        stream: bool = False,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        """
        单轮对话的便捷封装
        
        Args:
            prompt: 用户输入
            system_prompt: 系统提示
            stream: 是否流式输出
            **kwargs: 其他生成参数
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        return self.chat(messages, stream=stream, **kwargs)
    
    def get_embeddings(self, text: str) -> List[float]:
        """
        获取文本嵌入
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("模型未加载，请先调用load_model()方法")
        
        # 编码文本
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 获取模型输出
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # 使用最后一层的平均池化作为嵌入
        last_hidden_state = outputs.hidden_states[-1]
        embeddings = torch.mean(last_hidden_state, dim=1)
        
        return embeddings[0].cpu().numpy().tolist()


# 使用示例
if __name__ == "__main__":
    # 创建模型加载器
    loader = ModelLoader(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        device="cuda",  # 或 "cpu", "mps"
        dtype=torch.float16,  # 使用fp16节省内存
        load_in_4bit=False,  # 可选：4位量化
        verbose=True
    )
    
    # 加载模型
    loader.load_model()
    
    # 单轮对话
    response = loader.generate(
        prompt="请介绍一下Python编程语言",
        temperature=0.7,
        max_new_tokens=512
    )
    print(response)
    
    # 多轮对话
    messages = [
        {"role": "system", "content": "你是一位Python编程专家"},
        {"role": "user", "content": "什么是装饰器？"},
    ]
    response = loader.chat(messages)
    print(response)
    
    # 流式输出
    for chunk in loader.generate("写一个关于春天的诗", stream=True):
        print(chunk, end="", flush=True)