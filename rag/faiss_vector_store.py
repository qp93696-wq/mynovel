"""
rag/faiss_vector_store.py - FAISS向量存储实现
"""

import faiss
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from loguru import logger
import hashlib


@dataclass
class Document:
    """文档数据结构"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if not self.id:
            # 生成唯一ID
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:16]
            self.id = f"doc_{content_hash}"
    
    def to_dict(self) -> Dict:
        """转换为字典（不包含embedding）"""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata
        }


class FAISSVectorStore:
    """FAISS向量存储"""
    
    def __init__(
        self,
        embedding_dim: int = 768,
        index_type: str = "IVF",
        metric: str = "cosine",
        use_gpu: bool = False
    ):
        """
        初始化FAISS向量存储
        
        Args:
            embedding_dim: 向量维度
            index_type: 索引类型 (Flat/IVF/HNSW/LSH)
            metric: 距离度量 (cosine/l2/ip)
            use_gpu: 是否使用GPU
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        
        # 文档存储
        self.documents: List[Document] = []
        self.doc_map: Dict[str, int] = {}  # doc_id -> index
        
        # 初始化索引
        self.index = self._create_index()
        
        logger.info(f"FAISS向量存储初始化: dim={embedding_dim}, type={index_type}, metric={metric}")
    
    def _create_index(self) -> faiss.Index:
        """创建FAISS索引"""
        # 选择度量类型
        if self.metric == "cosine":
            metric_type = faiss.METRIC_INNER_PRODUCT
        elif self.metric == "l2":
            metric_type = faiss.METRIC_L2
        else:
            metric_type = faiss.METRIC_INNER_PRODUCT
        
        if self.index_type == "Flat":
            # 精确搜索
            if metric_type == faiss.METRIC_INNER_PRODUCT:
                index = faiss.IndexFlatIP(self.embedding_dim)
            else:
                index = faiss.IndexFlatL2(self.embedding_dim)
                
        elif self.index_type == "IVF":
            # 倒排文件索引（适合中大规模）
            nlist = 100  # 聚类中心数
            quantizer = faiss.IndexFlatIP(self.embedding_dim) if metric_type == faiss.METRIC_INNER_PRODUCT \
                       else faiss.IndexFlatL2(self.embedding_dim)
            
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, metric_type)
            
        elif self.index_type == "HNSW":
            # 分层可导航小世界图（适合大规模）
            M = 32  # 每个点的连接数
            index = faiss.IndexHNSWFlat(self.embedding_dim, M, metric_type)
            
        elif self.index_type == "LSH":
            # 局部敏感哈希（适合超大规模）
            nbits = self.embedding_dim * 4  # 哈希位数
            index = faiss.IndexLSH(self.embedding_dim, nbits)
        
        else:
            # 默认使用Flat
            index = faiss.IndexFlatIP(self.embedding_dim)
        
        # GPU加速
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("使用GPU加速的FAISS索引")
            except Exception as e:
                logger.warning(f"GPU加速失败，使用CPU: {e}")
                self.use_gpu = False
        
        return index
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: np.ndarray,
        update_if_exists: bool = False
        ) -> int:
        """
        添加文档到向量库
    
        Args:
            documents: 文档列表
            embeddings: 嵌入向量矩阵
            update_if_exists: 如果文档已存在是否更新（已废弃，将在未来版本移除）
        
        Returns:
            添加的文档数量
        """
        assert len(documents) == len(embeddings), "文档数量和嵌入数量不匹配"
    
        # 如果设置了update_if_exists，给出警告
        if update_if_exists:
            logger.warning(
                "update_if_exists参数已废弃。FAISS不支持高效的向量更新。"
                "如需更新文档，请先调用delete_documents()删除旧文档，再添加新文档。"
            )
    
        # 准备添加的文档和嵌入
        docs_to_add = []
        embeds_to_add = []
        skipped_docs = []
    
        for doc, embed in zip(documents, embeddings):
            if doc.id in self.doc_map:
                # 文档已存在，跳过添加
                skipped_docs.append(doc.id)
                logger.debug(f"文档 {doc.id} 已存在，跳过添加")
            else:
                docs_to_add.append(doc)
                embeds_to_add.append(embed)
    
        # 如果有文档被跳过，给出提示
        if skipped_docs:
            logger.info(
                f"跳过了 {len(skipped_docs)} 个已存在的文档。"
                f"如需更新这些文档，请先使用delete_documents()删除它们。"
            )
    
        if not docs_to_add:
            return 0
    
        # 归一化（如果使用余弦相似度）
        if self.metric == "cosine":
            embeds_array = np.array(embeds_to_add).astype('float32')
            faiss.normalize_L2(embeds_array)
        else:
            embeds_array = np.array(embeds_to_add).astype('float32')
    
        # 训练索引（如果需要）
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            if len(embeds_array) >= 100:
                logger.info("训练FAISS索引...")
                self.index.train(embeds_array)
            else:
                # 使用随机向量训练
                random_vecs = np.random.randn(100, self.embedding_dim).astype('float32')
                if self.metric == "cosine":
                    faiss.normalize_L2(random_vecs)
                self.index.train(random_vecs)
    
        # 添加到索引
        start_idx = len(self.documents)
        self.index.add(embeds_array)
    
        # 更新文档存储
        for i, doc in enumerate(docs_to_add):
            doc.embedding = embeds_array[i]
            self.documents.append(doc)
            self.doc_map[doc.id] = start_idx + i
    
        logger.info(f"添加了 {len(docs_to_add)} 个文档，总文档数: {len(self.documents)}")
    
        return len(docs_to_add)


    def update_documents(
        self,
        documents: List[Document],
        embeddings: np.ndarray
    ) -> Tuple[int, int]:
        """
        更新文档（通过删除旧文档并添加新文档实现）
    
        Args:
            documents: 要更新的文档列表
            embeddings: 新的嵌入向量矩阵
        
        Returns:
            (删除的文档数, 添加的文档数)
        """
        # 收集需要更新的文档ID
        doc_ids_to_update = [doc.id for doc in documents if doc.id in self.doc_map]
    
        if not doc_ids_to_update:
            logger.info("没有找到需要更新的文档，将作为新文档添加")
            added = self.add_documents(documents, embeddings)
            return 0, added
    
        # 删除旧文档
        deleted = self.delete_documents(doc_ids_to_update)
    
        # 添加新文档
        added = self.add_documents(documents, embeddings)
    
        logger.info(f"更新完成：删除了 {deleted} 个旧文档，添加了 {added} 个新文档")
    
        return deleted, added
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        搜索相似文档
        
        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            threshold: 相似度阈值
            filter_metadata: 元数据过滤条件
            
        Returns:
            [(文档, 相似度分数)]
        """
        if self.index.ntotal == 0:
            logger.warning("索引为空")
            return []
        
        # 准备查询向量
        query_vec = query_embedding.reshape(1, -1).astype('float32')
        
        # 归一化（如果使用余弦相似度）
        if self.metric == "cosine":
            faiss.normalize_L2(query_vec)
        
        # 搜索
        search_k = min(top_k * 2, self.index.ntotal) if filter_metadata else top_k
        
        # 设置搜索参数（针对IVF索引）
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = 10  # 搜索10个聚类中心
        
        scores, indices = self.index.search(query_vec, search_k)
        
        # 整理结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            
            doc = self.documents[idx]
            
            # 元数据过滤
            if filter_metadata:
                match = all(
                    doc.metadata.get(key) == value
                    for key, value in filter_metadata.items()
                )
                if not match:
                    continue
            
            # 相似度阈值过滤
            if score >= threshold:
                results.append((doc, float(score)))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def delete_documents(self, doc_ids: List[str]) -> int:
        """
        删除文档（需要重建索引）
        
        Args:
            doc_ids: 要删除的文档ID列表
            
        Returns:
            删除的文档数量
        """
        # 找出要保留的文档
        keep_indices = []
        keep_docs = []
        keep_embeddings = []
        
        for i, doc in enumerate(self.documents):
            if doc.id not in doc_ids:
                keep_indices.append(i)
                keep_docs.append(doc)
                if doc.embedding is not None:
                    keep_embeddings.append(doc.embedding)
        
        if len(keep_docs) == len(self.documents):
            logger.warning("没有找到要删除的文档")
            return 0
        
        # 重建索引
        deleted_count = len(self.documents) - len(keep_docs)
        
        self.documents = keep_docs
        self.doc_map = {doc.id: i for i, doc in enumerate(self.documents)}
        
        # 重建FAISS索引
        self.index = self._create_index()
        
        if keep_embeddings:
            embeds_array = np.array(keep_embeddings).astype('float32')
            if self.metric == "cosine":
                faiss.normalize_L2(embeds_array)
            
            # 重新训练和添加
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                self.index.train(embeds_array)
            self.index.add(embeds_array)
        
        logger.info(f"删除了 {deleted_count} 个文档，剩余 {len(self.documents)} 个")
        
        return deleted_count
    
    def save(self, path: str):
        """
        保存向量库到文件
        
        Args:
            path: 保存路径（不含扩展名）
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存FAISS索引
        faiss.write_index(self.index, str(path) + ".faiss")
        
        # 保存文档（不包含embedding以节省空间）
        docs_data = {
            "documents": [doc.to_dict() for doc in self.documents],
            "doc_map": self.doc_map,
            "config": {
                "embedding_dim": self.embedding_dim,
                "index_type": self.index_type,
                "metric": self.metric
            }
        }
        
        with open(str(path) + ".pkl", 'wb') as f:
            pickle.dump(docs_data, f)
        
        logger.info(f"向量库已保存到: {path}")
    
    def load(self, path: str):
        """
        从文件加载向量库
        
        Args:
            path: 加载路径（不含扩展名）
        """
        path = Path(path)
        
        # 加载FAISS索引
        self.index = faiss.read_index(str(path) + ".faiss")
        
        # 加载文档
        with open(str(path) + ".pkl", 'rb') as f:
            docs_data = pickle.load(f)
        
        self.documents = [Document(**doc) for doc in docs_data["documents"]]
        self.doc_map = docs_data["doc_map"]
        
        # 恢复配置
        config = docs_data.get("config", {})
        self.embedding_dim = config.get("embedding_dim", self.embedding_dim)
        self.index_type = config.get("index_type", self.index_type)
        self.metric = config.get("metric", self.metric)
        
        logger.info(f"向量库已加载: {len(self.documents)} 个文档")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "metric": self.metric,
            "use_gpu": self.use_gpu
        }
        
        # 统计元数据
        if self.documents:
            metadata_keys = set()
            for doc in self.documents:
                metadata_keys.update(doc.metadata.keys())
            stats["metadata_keys"] = list(metadata_keys)
        
        return stats
