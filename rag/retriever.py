"""
rag/retriever.py - RAG检索器
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from loguru import logger
import jieba
from .embedding_model import EmbeddingModel
from .faiss_vector_store import FAISSVectorStore, Document


class RAGRetriever:
    """RAG检索器"""
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: FAISSVectorStore,
        rerank: bool = True,
        hybrid_search: bool = True
    ):
        """
        初始化检索器
        
        Args:
            embedding_model: 嵌入模型
            vector_store: 向量存储
            rerank: 是否启用重排序
            hybrid_search: 是否启用混合检索（向量+关键词）
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.rerank = rerank
        self.hybrid_search = hybrid_search
        
        # 初始化jieba
        self._init_jieba()
    
    def _init_jieba(self):
        """初始化jieba分词"""
        # 添加小说常用词
        novel_words = [
            '修炼', '突破', '灵气', '真元', '法力', '神识',
            '剑意', '刀意', '道心', '心魔', '天劫', '飞升'
        ]
        for word in novel_words:
            jieba.add_word(word)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        rerank_top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回数量
            filters: 元数据过滤条件
            rerank_top_k: 重排序候选数量
            
        Returns:
            检索结果列表
        """
        # 生成查询向量
        query_embedding = self.embedding_model.encode_queries(query, show_progress=False)
        
        # 向量检索
        if self.rerank and rerank_top_k:
            search_k = rerank_top_k
        else:
            search_k = top_k * 2 if self.hybrid_search else top_k
        
        vector_results = self.vector_store.search(
            query_embedding[0],
            top_k=search_k,
            filter_metadata=filters
        )
        
        # 混合检索（添加关键词匹配）
        if self.hybrid_search:
            vector_results = self._hybrid_score(query, vector_results)
        
        # 重排序
        if self.rerank:
            vector_results = self._rerank_results(query, vector_results, top_k)
        
        # 格式化输出
        results = []
        for doc, score in vector_results[:top_k]:
            results.append({
                'id': doc.id,
                'content': doc.content,
                'metadata': doc.metadata,
                'score': score
            })
        
        return results
    
    def _hybrid_score(
        self,
        query: str,
        vector_results: List[Tuple[Document, float]],
        keyword_weight: float = 0.3
    ) -> List[Tuple[Document, float]]:
        """
        混合评分（向量相似度 + 关键词匹配）
        
        Args:
            query: 查询文本
            vector_results: 向量检索结果
            keyword_weight: 关键词权重
            
        Returns:
            重新评分的结果
        """
        # 提取查询关键词
        query_tokens = set(jieba.cut_for_search(query))
        
        hybrid_results = []
        for doc, vec_score in vector_results:
            # 计算关键词匹配分数
            doc_tokens = set(jieba.cut_for_search(doc.content[:500]))  # 只看前500字
            
            # Jaccard相似度
            intersection = len(query_tokens & doc_tokens)
            union = len(query_tokens | doc_tokens)
            keyword_score = intersection / max(union, 1)
            
            # 混合分数
            final_score = (1 - keyword_weight) * vec_score + keyword_weight * keyword_score
            hybrid_results.append((doc, final_score))
        
        # 重新排序
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        
        return hybrid_results
    
    def _rerank_results(
        self,
        query: str,
        results: List[Tuple[Document, float]],
        top_k: int
    ) -> List[Tuple[Document, float]]:
        """
        重排序结果
        
        Args:
            query: 查询文本
            results: 初始检索结果
            top_k: 最终返回数量
            
        Returns:
            重排序后的结果
        """
        if not results:
            return results
        
        # 简单的重排序策略：结合多个因素
        reranked = []
        
        for doc, initial_score in results:
            # 1. 长度惩罚（过短或过长的文档降权）
            doc_length = len(doc.content)
            length_score = 1.0
            if doc_length < 100:
                length_score = 0.7
            elif doc_length > 2000:
                length_score = 0.9
            
            # 2. 查询词位置（查询词出现在开头加分）
            position_score = 1.0
            query_tokens = list(jieba.cut(query))
            for token in query_tokens:
                if token in doc.content[:100]:
                    position_score = 1.2
                    break
            
            # 3. 风格一致性（如果有风格元数据）
            style_score = 1.0
            if 'style' in doc.metadata:
                # 检查查询中是否包含风格关键词
                style = doc.metadata['style']
                if style in query:
                    style_score = 1.3
            
            # 综合评分
            final_score = initial_score * length_score * position_score * style_score
            reranked.append((doc, final_score))
        
        # 排序并返回
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked[:top_k]
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        rerank_top_k: Optional[int] = None
        ) -> List[List[Dict[str, Any]]]:
        """
        批量检索（真正的批处理优化）
    
        Args:
            queries: 查询列表
            top_k: 每个查询返回的数量
            filters: 元数据过滤条件
            rerank_top_k: 重排序候选数量
        
        Returns:
            检索结果列表的列表
        """
        if not queries:
            return []
        
        # 1. 批量生成查询向量（一次性编码所有查询）
        logger.info(f"批量编码 {len(queries)} 个查询...")
        query_embeddings = self.embedding_model.encode_queries(queries, show_progress=True)
        
        # 2. 确定搜索数量
        if self.rerank and rerank_top_k:
            search_k = rerank_top_k
        else:
            search_k = top_k * 2 if self.hybrid_search else top_k
        
        # 3. 批量向量检索
        all_results = []
        
        # 如果FAISS支持批量搜索，使用批量搜索
        if hasattr(self.vector_store.index, 'search') and len(queries) > 1:
            # 批量搜索所有查询向量
            all_results = self._batch_vector_search(
                query_embeddings, 
                search_k, 
                filters
            )
        else:
            # 退回到循环搜索（但嵌入已经批量完成）
            for query_embedding in query_embeddings:
                results = self.vector_store.search(
                    query_embedding,
                    top_k=search_k,
                    filter_metadata=filters
                )
                all_results.append(results)
        
        # 4. 批量后处理（混合检索和重排序）
        final_results = []
        for i, (query, vector_results) in enumerate(zip(queries, all_results)):
            # 混合检索
            if self.hybrid_search:
                vector_results = self._hybrid_score(query, vector_results)
            
            # 重排序
            if self.rerank:
                vector_results = self._rerank_results(query, vector_results, top_k)
            
            # 格式化输出
            results = []
            for doc, score in vector_results[:top_k]:
                results.append({
                    'id': doc.id,
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'score': score
                })
            final_results.append(results)
        
        logger.info(f"批量检索完成，每个查询返回 {top_k} 个结果")
        return final_results
    
    def _batch_vector_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
        ) -> List[List[Tuple[Document, float]]]:
        """
        批量向量搜索（利用FAISS的批量搜索能力）
        
        Args:
            query_embeddings: 查询向量矩阵 [n_queries, embedding_dim]
            top_k: 返回数量
            filters: 元数据过滤条件
            
        Returns:
            每个查询的检索结果列表
        """
        if self.vector_store.index.ntotal == 0:
            logger.warning("索引为空")
            return [[] for _ in range(len(query_embeddings))]
        
        # 准备查询向量
        query_vecs = query_embeddings.astype('float32')
        
        # 归一化（如果使用余弦相似度）
        if self.vector_store.metric == "cosine":
            import faiss
            faiss.normalize_L2(query_vecs)
        
        # 设置搜索参数（针对IVF索引）
        if hasattr(self.vector_store.index, 'nprobe'):
            self.vector_store.index.nprobe = 10
        
        # 批量搜索（FAISS原生支持）
        search_k = min(top_k * 2, self.vector_store.index.ntotal) if filters else top_k
        scores_batch, indices_batch = self.vector_store.index.search(query_vecs, search_k)
        
        # 整理结果
        all_results = []
        for scores, indices in zip(scores_batch, indices_batch):
            results = []
            for score, idx in zip(scores, indices):
                if idx < 0 or idx >= len(self.vector_store.documents):
                    continue
                
                doc = self.vector_store.documents[idx]
                
                # 元数据过滤
                if filters:
                    match = all(
                        doc.metadata.get(key) == value
                        for key, value in filters.items()
                    )
                    if not match:
                        continue
                
                results.append((doc, float(score)))
                
                if len(results) >= top_k:
                    break
            
            all_results.append(results)
        
        return all_results


    def retrieve_with_cache(
        self,
        query: str,
        top_k: int = 5,
        use_cache: bool = True,
        cache_ttl: int = 3600,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        带缓存的检索（避免重复查询）
        
        Args:
            query: 查询文本
            top_k: 返回数量
            use_cache: 是否使用缓存
            cache_ttl: 缓存过期时间（秒）
            **kwargs: 其他检索参数
            
        Returns:
            检索结果列表
        """
        if not hasattr(self, '_cache'):
            self._cache = {}
            self._cache_time = {}
        
        # 生成缓存键
        import hashlib
        import time
        cache_key = hashlib.md5(f"{query}_{top_k}_{kwargs}".encode()).hexdigest()
        
        # 检查缓存
        if use_cache and cache_key in self._cache:
            if time.time() - self._cache_time[cache_key] < cache_ttl:
                logger.debug(f"使用缓存结果: {cache_key[:8]}")
                return self._cache[cache_key]
        
        # 执行检索
        results = self.retrieve(query, top_k, **kwargs)
        
        # 更新缓存
        if use_cache:
            self._cache[cache_key] = results
            self._cache_time[cache_key] = time.time()
            
            # 清理过期缓存
            current_time = time.time()
            expired_keys = [
                k for k, t in self._cache_time.items() 
                if current_time - t > cache_ttl
            ]
            for k in expired_keys:
                del self._cache[k]
                del self._cache_time[k]
        
        return results