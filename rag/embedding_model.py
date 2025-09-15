"""
rag/embedding_model.py - 文本嵌入模型
"""

import torch
import numpy as np
from typing import List, Union, Optional 
from sentence_transformers import SentenceTransformer
from loguru import logger
from tqdm import tqdm


class EmbeddingModel:
    """文本嵌入模型封装"""
    
    def __init__(self,
                model_name: str = "BAAI/bge-small-zh-v1.5",
                device: str = None,
                use_compile: bool = False,  # 是否使用torch.compile
                compile_mode: str = "reduce-overhead" 
                ):
        """
        初始化嵌入模型
        
        Args:
            model_name: 模型名称
                推荐模型：
                - BAAI/bge-small-zh-v1.5 (快速，384维)
                - BAAI/bge-base-zh-v1.5 (平衡，768维)
                - BAAI/bge-large-zh-v1.5 (精准，1024维)
            device: 设备类型
            use_compile: 是否使用torch.compile加速（None为自动检测）
            compile_mode: 编译模式 ("default", "reduce-overhead", "max-autotune")
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"加载嵌入模型: {model_name} on {self.device}")
        
        # 加载模型
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.eval()
        
        self._apply_compile_optimization(use_compile, compile_mode)

        # 获取嵌入维度
        test_embedding = self.model.encode(["测试"], convert_to_numpy=True)
        self.embedding_dim = test_embedding.shape[1]
        
        logger.success(f"嵌入模型加载完成，维度: {self.embedding_dim}")
    

    def _apply_compile_optimization(self, use_compile: Optional[bool], compile_mode: str):
        """应用torch.compile优化"""
        # 自动检测是否应该使用compile
        if use_compile is None:
            # PyTorch 2.0+ 且使用GPU时默认启用
            use_compile = (
                hasattr(torch, 'compile') and 
                torch.cuda.is_available() and 
                self.device != 'cpu'
            )
        
        if use_compile:
            try:
                if hasattr(torch, 'compile'):
                    logger.info(f"应用torch.compile优化 (mode={compile_mode})")
                    
                    # 获取底层的transformer模型
                    if hasattr(self.model[0], 'auto_model'):
                        # 对transformer模型应用compile
                        original_model = self.model[0].auto_model
                        compiled_model = torch.compile(
                            original_model,
                            mode=compile_mode,
                            backend="inductor",  # 使用inductor后端
                            fullgraph=False,  # 允许部分图编译
                            dynamic=True  # 支持动态形状
                        )
                        self.model[0].auto_model = compiled_model
                        
                        # 预热编译
                        self._warmup_compile()
                        
                        logger.success("torch.compile优化应用成功")
                    else:
                        logger.warning("无法访问底层模型，跳过compile优化")
                else:
                    logger.info("当前PyTorch版本不支持compile")
                    
            except Exception as e:
                logger.warning(f"torch.compile优化失败，使用原始模型: {e}")


    def _warmup_compile(self):
        """预热编译缓存"""
        try:
            logger.debug("预热torch.compile...")
            # 使用不同长度的输入预热
            warmup_texts = [
                "短文本",
                "这是一个中等长度的测试文本用于预热编译器缓存",
                "这是一个更长的测试文本，" * 10
            ]
            _ = self.model.encode(warmup_texts, convert_to_numpy=True, show_progress_bar=False)
            logger.debug("预热完成")
        except Exception as e:
            logger.debug(f"预热失败（不影响使用）: {e}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True,
        convert_to_tensor: bool = False,  # 新增：是否返回tensor
        precision: str = "float32"  # 新增：精度控制
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        编码文本为嵌入向量（优化版）
        
        Args:
            texts: 文本或文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条
            normalize: 是否L2归一化
            convert_to_tensor: 是否返回PyTorch tensor
            precision: 精度 ("float32", "float16", "bfloat16")
            
        Returns:
            嵌入向量矩阵
        """
        # 确保输入是列表
        if isinstance(texts, str):
            texts = [texts]
        
        # 根据设备和任务规模动态调整batch_size
        if self.device == 'cuda' and len(texts) > 1000:
            # 大规模任务时增加batch_size
            batch_size = min(batch_size * 2, 256)
        
        # 设置精度
        original_dtype = None
        if precision != "float32" and self.device == 'cuda':
            original_dtype = torch.get_default_dtype()
            if precision == "float16":
                torch.set_default_dtype(torch.float16)
            elif precision == "bfloat16" and torch.cuda.is_bf16_supported():
                torch.set_default_dtype(torch.bfloat16)
        
        try:
            # 编码
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=not convert_to_tensor,
                normalize_embeddings=normalize,
                device=self.device
            )
            
            # 如果需要额外的归一化（某些模型可能需要）
            if normalize and convert_to_tensor and isinstance(embeddings, torch.Tensor):
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
        finally:
            # 恢复原始精度
            if original_dtype is not None:
                torch.set_default_dtype(original_dtype)
        
        return embeddings
    
    def encode_batch_multiprocess(
        self,
        texts: List[str],
        batch_size: int = 32,
        num_workers: int = 4,
        normalize: bool = True
    ) -> np.ndarray:
        """
        使用多进程批量编码（适合超大规模数据）
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            num_workers: 工作进程数
            normalize: 是否归一化
            
        Returns:
            嵌入向量矩阵
        """
        from multiprocessing import Pool
        import math
        
        # 分割数据
        chunk_size = math.ceil(len(texts) / num_workers)
        text_chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
        
        # 多进程处理
        with Pool(num_workers) as pool:
            results = pool.starmap(
                self._encode_chunk,
                [(chunk, batch_size, normalize) for chunk in text_chunks]
            )
        
        # 合并结果
        embeddings = np.vstack(results)
        return embeddings
    
    def _encode_chunk(self, texts: List[str], batch_size: int, normalize: bool) -> np.ndarray:
        """编码文本块（用于多进程）"""
        return self.encode(texts, batch_size=batch_size, show_progress=False, normalize=normalize)
    
    def encode_queries(self, queries: Union[str, List[str]], **kwargs) -> np.ndarray:
        """编码查询文本（可能使用特殊的查询前缀）"""
        if isinstance(queries, str):
            queries = [queries]
        
        # BGE模型需要添加查询前缀
        if "bge" in self.model_name.lower():
            queries = [f"为这个句子生成表示以用于检索相关文章：{q}" for q in queries]
        
        # 查询通常较短，可以使用更高的精度
        kwargs.setdefault('precision', 'float32')
        
        return self.encode(queries, **kwargs)
    
    def encode_documents(
        self, 
        documents: Union[str, List[str]], 
        **kwargs
    ) -> np.ndarray:
        """编码文档文本（优化大批量处理）"""
        # 文档编码可以使用较低精度以节省内存
        kwargs.setdefault('precision', 'float16' if self.device == 'cuda' else 'float32')
        kwargs.setdefault('batch_size', 64)  # 文档通常可以用更大的batch
        
        return self.encode(documents, **kwargs)