"""
rag/knowledge_base.py - 知识库管理
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import re
from loguru import logger
from tqdm import tqdm

from .embedding_model import EmbeddingModel
from .faiss_vector_store import FAISSVectorStore, Document
from .retriever import RAGRetriever


class NovelKnowledgeBase:
    """小说知识库管理"""
    
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-small-zh-v1.5",
        vector_store_path: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        """
        初始化知识库
        
        Args:
            embedding_model_name: 嵌入模型名称
            vector_store_path: 向量库保存路径
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store_path = vector_store_path
        
        # 初始化嵌入模型
        self.embedding_model = EmbeddingModel(embedding_model_name)
        
        # 初始化向量存储
        self.vector_store = FAISSVectorStore(
            embedding_dim=self.embedding_model.embedding_dim,
            index_type="IVF",
            metric="cosine"
        )
        
        # 加载已有向量库
        if vector_store_path and Path(vector_store_path + ".faiss").exists():
            self.load()
        
        # 初始化检索器
        self.retriever = RAGRetriever(
            self.embedding_model,
            self.vector_store,
            rerank=True,
            hybrid_search=True
        )
    
    def add_novel(
        self,
        novel_path: str,
        style: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        添加小说到知识库
        
        Args:
            novel_path: 小说文件路径
            style: 小说风格
            metadata: 额外元数据
            
        Returns:
            添加的文档数量
        """
        logger.info(f"添加小说: {novel_path}")
        
        # 读取小说内容
        with open(novel_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 分割成文本块
        chunks = self._split_text(content)
        
        # 创建文档
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = {
                "source": Path(novel_path).stem,
                "style": style,
                "chunk_index": i,
                "chunk_size": len(chunk)
            }
            if metadata:
                doc_metadata.update(metadata)
            
            doc = Document(
                id=f"{Path(novel_path).stem}_{i}",
                content=chunk,
                metadata=doc_metadata
            )
            documents.append(doc)
        
        # 生成嵌入
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode_documents(
            texts,
            batch_size=32,
            show_progress=True
        )
        
        # 添加到向量库
        added = self.vector_store.add_documents(documents, embeddings)
        
        logger.success(f"添加了 {added} 个文本块")
        
        return added
    
    def add_novels_batch(
        self,
        novel_dir: str,
        style_mapping: Optional[Dict[str, str]] = None
    ) -> int:
        """
        批量添加小说
        
        Args:
            novel_dir: 小说目录
            style_mapping: 文件名到风格的映射
            
        Returns:
            总添加文档数
        """
        novel_dir = Path(novel_dir)
        novel_files = list(novel_dir.glob("*.txt"))
        
        logger.info(f"批量添加 {len(novel_files)} 个小说文件")
        
        total_added = 0
        for novel_file in tqdm(novel_files, desc="添加小说"):
            # 确定风格
            if style_mapping and novel_file.stem in style_mapping:
                style = style_mapping[novel_file.stem]
            else:
                # 根据目录名推断风格
                style = novel_file.parent.name
            
            added = self.add_novel(str(novel_file), style)
            total_added += added
        
        logger.success(f"批量添加完成，共 {total_added} 个文档")
        
        # 自动保存
        if self.vector_store_path:
            self.save()
        
        return total_added
    
    def _split_text(self, text: str) -> List[str]:
        """
        智能文本分割
        
        Args:
            text: 原始文本
            
        Returns:
            文本块列表
        """
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
                if para_length > self.chunk_size * 2:
                    # 按句子分割
                    sentences = re.split(r'[。！？]', para)
                    for sent in sentences:
                        if sent:
                            if current_length + len(sent) > self.chunk_size:
                                if current_chunk:
                                    chunks.append('\n'.join(current_chunk))
                                current_chunk = [sent + '。']
                                current_length = len(sent)
                            else:
                                current_chunk.append(sent + '。')
                                current_length += len(sent)
                
                # 如果当前块太长，保存并开始新块
                elif current_length + para_length > self.chunk_size:
                    if current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                    
                    # 保留重叠部分
                    if self.chunk_overlap > 0 and current_chunk:
                        overlap_text = current_chunk[-1][-self.chunk_overlap:]
                        current_chunk = [overlap_text, para]
                        current_length = len(overlap_text) + para_length
                    else:
                        current_chunk = [para]
                        current_length = para_length
                
                else:
                    current_chunk.append(para)
                    current_length += para_length
            
            # 保存最后一个块
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        style: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索知识库
        
        Args:
            query: 查询文本
            top_k: 返回数量
            style: 风格过滤
            
        Returns:
            搜索结果
        """
        filters = {"style": style} if style else None
        return self.retriever.retrieve(query, top_k, filters)
    
    def save(self):
        """保存知识库"""
        if self.vector_store_path:
            self.vector_store.save(self.vector_store_path)
            logger.info(f"知识库已保存到: {self.vector_store_path}")
    
    def load(self):
        """加载知识库"""
        if self.vector_store_path:
            self.vector_store.load(self.vector_store_path)
            logger.info(f"知识库已加载")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.vector_store.get_statistics()
        
        # 添加风格统计
        if stats["total_documents"] > 0:
            style_counts = {}
            for doc in self.vector_store.documents:
                style = doc.metadata.get("style", "unknown")
                style_counts[style] = style_counts.get(style, 0) + 1
            stats["style_distribution"] = style_counts
        
        return stats
