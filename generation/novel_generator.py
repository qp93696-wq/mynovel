"""
novel_generator.py - 长篇小说生成系统
解决了长期一致性、动态节奏、智能解析等问题
"""

import os
import json
import time
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from loguru import logger
from tqdm import tqdm
import numpy as np

# 导入必要的组件
from generation.rag_generator import RAGNovelGenerator
from rag.knowledge_base import NovelKnowledgeBase
from rag.faiss_vector_store import Document


# ========================================
# 数据结构定义
# ========================================

class ChapterLength(Enum):
    """章节长度枚举"""
    SHORT = "short"      # 2000-3000字
    MEDIUM = "medium"    # 4000-5000字
    LONG = "long"        # 6000-8000字
    EPIC = "epic"        # 10000+字（关键剧情）

@dataclass
class Character:
    """角色定义"""
    name: str
    role: str  # 主角/配角/反派
    personality: List[str]
    background: str
    relationships: Dict[str, str]  # 与其他角色的关系
    development_arc: str  # 角色成长弧线
    first_appearance: int = 1  # 首次出现的章节
    
@dataclass
class PlotPoint:
    """情节点"""
    chapter: int
    event: str
    importance: str  # high/medium/low
    foreshadowing: List[str] = field(default_factory=list)  # 伏笔
    callbacks: List[int] = field(default_factory=list)  # 回应章节

@dataclass
class WorldSetting:
    """世界观设定"""
    name: str
    description: str
    rules: List[str]  # 世界规则
    locations: Dict[str, str]  # 重要地点
    power_system: str  # 力量体系
    
@dataclass
class NovelOutline:
    """增强版小说大纲"""
    title: str
    genre: str
    theme: str  # 主题
    world_setting: WorldSetting
    characters: List[Character]
    plot_points: List[PlotPoint]
    chapter_count: int
    estimated_words: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Chapter:
    """增强版章节"""
    number: int
    title: str
    outline: str
    length_type: ChapterLength
    target_words: int
    key_events: List[str]
    appearing_characters: List[str]
    emotional_tone: str  # 情感基调
    pacing: str  # fast/medium/slow
    content: str = ""
    actual_words: int = 0
    quality_score: float = 0.0


# ========================================
# 核心生成器
# ========================================

class EnhancedNovelGenerator:
    """增强版长篇小说生成器"""
    
    def __init__(
        self, 
        model_path: str, 
        style: str = "仙侠",
        use_rag: bool = True,
        enable_reflection: bool = True
    ):
        self.model_path = model_path
        self.style = style
        self.use_rag = use_rag
        self.enable_reflection = enable_reflection
        self.progress_callback = None
        
        # 初始化RAG知识库
        if use_rag:
            self.knowledge_base = NovelKnowledgeBase(
                embedding_model_name="BAAI/bge-small-zh-v1.5",
                vector_store_path=f"./data/novel_kb_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                chunk_size=300,
                chunk_overlap=50
            )
        else:
            self.knowledge_base = None
        
        # 初始化生成器
        self.generator = RAGNovelGenerator(
            model_name=model_path,
            knowledge_base=self.knowledge_base,
            max_history=30  # 增加历史记录
        )
        
        # 输出目录
        self.output_dir = Path(f"outputs/novels/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置
        self.config = {
            "auto_save_interval": 3,  # 每3章保存
            "reflection_interval": 10,  # 每10章反思
            "quality_threshold": 0.7,  # 质量阈值
            "max_retries": 3,  # 重试次数
        }
        
        # 状态跟踪
        self.state = {
            "current_chapter": 0,
            "total_words": 0,
            "generation_history": [],
            "quality_scores": [],
        }
    
    def generate_novel(
        self, 
        title: str, 
        target_words: int = 500000,
        resume_from_checkpoint: bool = False,
        progress_callback=None
    ):
        """
        生成完整小说（主流程）
        """
        self.progress_callback = progress_callback
        logger.info(f"🚀 开始生成小说《{title}》，目标字数：{target_words}")
        
        # 检查是否从检查点恢复
        if resume_from_checkpoint:
            outline, chapters = self.load_checkpoint()
            logger.info(f"从第{self.state['current_chapter']}章恢复生成")
        else:
            # Step 1: 生成结构化大纲
            outline = self.generate_structured_outline(title, target_words)
            self.save_outline(outline)
            
            # Step 2: 将大纲加入RAG知识库
            if self.use_rag:
                self.index_outline_to_rag(outline)
            
            # Step 3: 生成动态章节计划
            chapters = self.plan_chapters_dynamically(outline)
            self.save_chapter_plan(chapters)
        
        # Step 4: 智能生成章节内容
        completed_chapters = self.generate_chapters_with_consistency(
            chapters, 
            outline,
            start_from=self.state['current_chapter']
        )
        
        # Step 5: 后处理与整合
        self.post_process_and_compile(title, completed_chapters, outline)
        
        logger.success(f"✅ 小说生成完成！保存在: {self.output_dir}")
    
    def generate_structured_outline(
        self, 
        title: str, 
        target_words: int
    ) -> NovelOutline:
        """
        生成结构化的JSON格式大纲
        """
        logger.info("📝 生成结构化大纲...")
        
        # 估算章节数（根据动态长度）
        avg_chapter_words = 5000
        estimated_chapters = target_words // avg_chapter_words
        
        # 构建要求模型返回JSON的提示
        prompt = f"""你是一位专业的{self.style}小说创作大师。
请为小说《{title}》创建一个详细的大纲。

目标字数：{target_words}字
预计章节：{estimated_chapters}章

请严格按照以下JSON格式返回（确保是合法的JSON）：
{{
    "title": "{title}",
    "genre": "{self.style}",
    "theme": "小说的核心主题",
    "world_setting": {{
        "name": "世界名称",
        "description": "详细的世界观描述",
        "rules": ["规则1", "规则2", "规则3"],
        "locations": {{"地点1": "描述", "地点2": "描述"}},
        "power_system": "力量体系说明"
    }},
    "characters": [
        {{
            "name": "角色名",
            "role": "主角/配角/反派",
            "personality": ["性格特点1", "性格特点2"],
            "background": "背景故事",
            "relationships": {{"其他角色": "关系描述"}},
            "development_arc": "角色成长弧线"
        }}
    ],
    "plot_points": [
        {{
            "chapter": 1,
            "event": "关键事件描述",
            "importance": "high/medium/low",
            "foreshadowing": ["伏笔1", "伏笔2"]
        }}
    ],
    "chapter_distribution": {{
        "opening": 0.1,
        "development": 0.3,
        "climax": 0.4,
        "resolution": 0.2
    }}
}}

要求：
1. 至少包含5个主要角色
2. 至少规划10个关键情节点
3. 伏笔要贯穿全文
4. 确保返回的是合法的JSON格式
"""
        
        # 多次尝试获取有效的JSON
        for attempt in range(self.config["max_retries"]):
            try:
                response = self.generator.generate(
                    prompt=prompt,
                    style=self.style,
                    max_new_tokens=3000,
                    temperature=0.7,
                    use_rag=False
                )
                
                # 提取JSON部分
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    outline_data = json.loads(json_str)
                    
                    # 转换为数据类
                    outline = self._parse_outline_json(outline_data, estimated_chapters, target_words)
                    logger.success("✅ 成功生成结构化大纲")
                    return outline
                    
            except json.JSONDecodeError as e:
                logger.warning(f"第{attempt+1}次解析失败: {e}")
                if attempt < self.config["max_retries"] - 1:
                    time.sleep(2)
                    continue
            except Exception as e:
                logger.error(f"生成大纲失败: {e}")
                
        # 如果JSON解析失败，使用备用方案
        logger.warning("JSON解析失败，使用备用方案生成大纲")
        return self._generate_fallback_outline(title, target_words, estimated_chapters)
    
    def _parse_outline_json(
        self, 
        data: Dict, 
        estimated_chapters: int,
        target_words: int
    ) -> NovelOutline:
        """解析JSON大纲数据"""
        
        # 解析世界观设定
        ws_data = data.get("world_setting", {})
        world_setting = WorldSetting(
            name=ws_data.get("name", "未知世界"),
            description=ws_data.get("description", ""),
            rules=ws_data.get("rules", []),
            locations=ws_data.get("locations", {}),
            power_system=ws_data.get("power_system", "")
        )
        
        # 解析角色
        characters = []
        for char_data in data.get("characters", []):
            character = Character(
                name=char_data.get("name", "未命名"),
                role=char_data.get("role", "配角"),
                personality=char_data.get("personality", []),
                background=char_data.get("background", ""),
                relationships=char_data.get("relationships", {}),
                development_arc=char_data.get("development_arc", "")
            )
            characters.append(character)
        
        # 解析情节点
        plot_points = []
        for pp_data in data.get("plot_points", []):
            plot_point = PlotPoint(
                chapter=pp_data.get("chapter", 1),
                event=pp_data.get("event", ""),
                importance=pp_data.get("importance", "medium"),
                foreshadowing=pp_data.get("foreshadowing", [])
            )
            plot_points.append(plot_point)
        
        return NovelOutline(
            title=data.get("title", "未命名"),
            genre=data.get("genre", self.style),
            theme=data.get("theme", ""),
            world_setting=world_setting,
            characters=characters,
            plot_points=plot_points,
            chapter_count=estimated_chapters,
            estimated_words=target_words,
            metadata=data.get("chapter_distribution", {})
        )
    
    def index_outline_to_rag(self, outline: NovelOutline):
        """
        将大纲内容索引到RAG知识库
        """
        if not self.knowledge_base:
            return
            
        logger.info("📚 将大纲索引到知识库...")
        
        documents = []
        
        # 1. 世界观设定
        world_doc = Document(
            id="world_setting",
            content=f"""世界观设定：
            名称：{outline.world_setting.name}
            描述：{outline.world_setting.description}
            规则：{', '.join(outline.world_setting.rules)}
            力量体系：{outline.world_setting.power_system}
            重要地点：{json.dumps(outline.world_setting.locations, ensure_ascii=False)}
            """,
            metadata={"type": "world_setting", "importance": "high"}
        )
        documents.append(world_doc)
        
        # 2. 角色设定
        for char in outline.characters:
            char_doc = Document(
                id=f"character_{char.name}",
                content=f"""角色：{char.name}
                身份：{char.role}
                性格：{', '.join(char.personality)}
                背景：{char.background}
                成长线：{char.development_arc}
                关系网：{json.dumps(char.relationships, ensure_ascii=False)}
                """,
                metadata={"type": "character", "name": char.name, "role": char.role}
            )
            documents.append(char_doc)
        
        # 3. 关键情节点
        for pp in outline.plot_points:
            plot_doc = Document(
                id=f"plot_{pp.chapter}_{pp.event[:20]}",
                content=f"""第{pp.chapter}章关键情节：
                事件：{pp.event}
                重要性：{pp.importance}
                伏笔：{', '.join(pp.foreshadowing)}
                """,
                metadata={"type": "plot", "chapter": pp.chapter, "importance": pp.importance}
            )
            documents.append(plot_doc)
        
        # 4. 主题和基调
        theme_doc = Document(
            id="theme",
            content=f"""小说主题：{outline.theme}
            风格：{outline.genre}
            基调：{self.style}
            """,
            metadata={"type": "theme"}
        )
        documents.append(theme_doc)
        
        # 生成嵌入并添加到知识库
        texts = [doc.content for doc in documents]
        embeddings = self.knowledge_base.embedding_model.encode_documents(
            texts, 
            batch_size=32,
            show_progress=False
        )
        
        self.knowledge_base.vector_store.add_documents(documents, embeddings)
        logger.success(f"✅ 成功索引{len(documents)}个文档到知识库")
    
    def plan_chapters_dynamically(self, outline: NovelOutline) -> List[Chapter]:
        """
        动态规划章节（考虑节奏变化）
        """
        logger.info("📊 动态规划章节...")
        
        chapters = []
        distribution = outline.metadata.get("chapter_distribution", {
            "opening": 0.1,
            "development": 0.3,
            "climax": 0.4,
            "resolution": 0.2
        })
        
        # 根据分布计算各阶段章节数
        total_chapters = outline.chapter_count
        stage_chapters = {
            "opening": int(total_chapters * distribution["opening"]),
            "development": int(total_chapters * distribution["development"]),
            "climax": int(total_chapters * distribution["climax"]),
            "resolution": int(total_chapters * distribution["resolution"])
        }
        
        chapter_num = 1
        
        for stage, count in stage_chapters.items():
            for i in range(count):
                # 动态决定章节长度和节奏
                length_type, target_words, pacing = self._determine_chapter_specs(
                    chapter_num, 
                    stage, 
                    outline.plot_points,
                    total_chapters
                )
                
                # 生成单章大纲
                chapter_outline = self._generate_chapter_outline(
                    chapter_num,
                    stage,
                    outline,
                    length_type,
                    pacing,
                    target_words 
                )
                
                chapters.append(chapter_outline)
                chapter_num += 1
                
                # 进度提示
                if chapter_num % 10 == 0:
                    logger.info(f"已规划{chapter_num}章")
        
        logger.success(f"✅ 完成{len(chapters)}章的动态规划")
        return chapters
    
    def _determine_chapter_specs(
        self,
        chapter_num: int,
        stage: str,
        plot_points: List[PlotPoint],
        total_chapters: int
    ) -> Tuple[ChapterLength, int, str]:
        """
        决定章节规格（长度、字数、节奏）
        """
        # 检查是否是关键章节
        is_key_chapter = any(
            pp.chapter == chapter_num and pp.importance == "high" 
            for pp in plot_points
        )
        
        # 根据阶段和重要性决定长度
        if is_key_chapter:
            length_type = ChapterLength.LONG
            target_words = np.random.randint(6000, 8000)
            pacing = "slow"  # 关键章节慢节奏，详细描写
        elif stage == "opening":
            length_type = ChapterLength.MEDIUM
            target_words = np.random.randint(4000, 5000)
            pacing = "medium"
        elif stage == "development":
            # 发展阶段有变化
            if chapter_num % 5 == 0:  # 每5章一个小高潮
                length_type = ChapterLength.LONG
                target_words = np.random.randint(5500, 7000)
                pacing = "medium"
            else:
                length_type = ChapterLength.MEDIUM
                target_words = np.random.randint(3500, 5000)
                pacing = "fast" if np.random.random() > 0.5 else "medium"
        elif stage == "climax":
            # 高潮部分
            if chapter_num > total_chapters * 0.7:  # 最后30%
                length_type = ChapterLength.EPIC
                target_words = np.random.randint(8000, 12000)
                pacing = "slow"
            else:
                length_type = ChapterLength.LONG
                target_words = np.random.randint(6000, 8000)
                pacing = "fast"
        else:  # resolution
            length_type = ChapterLength.MEDIUM
            target_words = np.random.randint(3000, 5000)
            pacing = "slow"
        
        return length_type, target_words, pacing
    
    def _generate_chapter_outline(
        self,
        chapter_num: int,
        stage: str,
        novel_outline: NovelOutline,
        length_type: ChapterLength,
        pacing: str
    ) -> Chapter:
        """生成单章大纲"""
        
        # 确定本章涉及的角色
        main_characters = [c.name for c in novel_outline.characters if c.role == "主角"]
        all_characters = [c.name for c in novel_outline.characters]
        
        # 根据length_type确定target_words
        target_words_map = {
            ChapterLength.SHORT: 3000,
            ChapterLength.MEDIUM: 5000,
            ChapterLength.LONG: 7000,
            ChapterLength.EPIC: 10000
        }
        target_words = target_words_map.get(length_type, 5000) 

        # 根据章节位置选择出场角色
        if chapter_num == 1:
            appearing = main_characters[:2]  # 开篇主角登场
        else:
            # 随机选择2-4个角色
            num_chars = np.random.randint(2, min(5, len(all_characters) + 1))
            appearing = np.random.choice(all_characters, num_chars, replace=False).tolist()
        
        # 查找本章关键事件
        chapter_events = [
            pp.event for pp in novel_outline.plot_points 
            if pp.chapter == chapter_num
        ]
        
        if not chapter_events:
            # 生成普通事件
            prompt = f"""为第{chapter_num}章生成3-5个事件。
            阶段：{stage}
            节奏：{pacing}
            出场角色：{', '.join(appearing)}
            
            返回JSON格式：
            {{"events": ["事件1", "事件2", "事件3"]}}
            """
            
            response = self.generator.generate(
                prompt=prompt,
                style=self.style,
                max_new_tokens=200,
                temperature=0.8
            )
            
            try:
                events_data = json.loads(response)
                chapter_events = events_data.get("events", ["发展剧情"])
            except:
                chapter_events = ["推进故事发展"]
        
        # 决定情感基调
        emotional_tones = {
            "opening": ["好奇", "期待", "神秘"],
            "development": ["紧张", "冲突", "成长", "友情"],
            "climax": ["激烈", "悲壮", "震撼", "转折"],
            "resolution": ["释然", "圆满", "感动", "回味"]
        }
        emotional_tone = np.random.choice(emotional_tones.get(stage, ["平静"]))
        
        # 生成章节标题
        title_prompt = f"为第{chapter_num}章起一个吸引人的标题，主要事件：{chapter_events[0] if chapter_events else '故事发展'}"
        title = self.generator.generate(
            prompt=title_prompt,
            max_new_tokens=20,
            temperature=0.9
        ).strip()
        
        # 生成章节大纲
        outline_prompt = f"""第{chapter_num}章大纲：
        标题提示：{title}
        阶段：{stage}
        字数：{length_type.value}（约{target_words}字）
        节奏：{pacing}
        情感：{emotional_tone}
        角色：{', '.join(appearing)}
        事件：{', '.join(chapter_events)}
        
        请用100-200字概括本章内容。
        """
        
        chapter_outline = self.generator.generate(
            prompt=outline_prompt,
            max_new_tokens=300,
            temperature=0.7
        )
        
        return Chapter(
            number=chapter_num,
            title=f"第{chapter_num}章 {title}",
            outline=chapter_outline,
            length_type=length_type,
            target_words=target_words,
            key_events=chapter_events,
            appearing_characters=appearing,
            emotional_tone=emotional_tone,
            pacing=pacing
        )
    
    def generate_chapters_with_consistency(
        self,
        chapters: List[Chapter],
        outline: NovelOutline,
        start_from: int = 0
    ) -> List[Chapter]:
        """
        生成章节内容（保持长期一致性）
        """
        logger.info(f"📖 开始生成章节内容（从第{start_from+1}章开始）...")
        
        completed_chapters = []
        
        # 加载已完成的章节
        if start_from > 0:
            completed_chapters = self.load_completed_chapters(start_from)
        
        # 生成新章节
        for i, chapter in enumerate(tqdm(chapters[start_from:], desc="生成章节")):
            chapter_num = start_from + i + 1
            
            # 生成章节内容
            chapter_with_content = self._generate_chapter_with_rag(
                chapter,
                outline,
                completed_chapters,
                chapter_num
            )
            
            # 调用进度回调
            if self.progress_callback:
                self.progress_callback(
                    chapter_num,
                    len(chapters),
                    self.state['total_words'],
                    chapter_with_content.quality_score,
                    chapter_with_content.content[:500]  # 预览
                )

            # 质量检查
            quality_score = self._evaluate_chapter_quality(chapter_with_content, outline)
            chapter_with_content.quality_score = quality_score
            
            # 如果质量太低，重新生成
            if quality_score < self.config["quality_threshold"]:
                logger.warning(f"第{chapter_num}章质量分数过低({quality_score:.2f})，重新生成...")
                chapter_with_content = self._regenerate_chapter(
                    chapter, outline, completed_chapters, chapter_num
                )
            
            completed_chapters.append(chapter_with_content)
            
            # 更新知识库（添加新生成的内容）
            if self.use_rag:
                self._update_rag_with_chapter(chapter_with_content)
            
            # 自动保存
            if (chapter_num % self.config["auto_save_interval"]) == 0:
                self.save_progress(completed_chapters, outline)
                logger.info(f"💾 已保存进度：{chapter_num}/{len(chapters)}章")
            
            # 反思与调整
            if self.enable_reflection and (chapter_num % self.config["reflection_interval"]) == 0:
                self._reflect_and_adjust(completed_chapters, chapters[chapter_num:], outline)
            
            # 更新状态
            self.state['current_chapter'] = chapter_num
            self.state['total_words'] += chapter_with_content.actual_words
            
            # 休息避免过热
            if chapter_num % 5 == 0:
                time.sleep(10)
        
        return completed_chapters
    
    def _generate_chapter_with_rag(
        self,
        chapter: Chapter,
        outline: NovelOutline,
        previous_chapters: List[Chapter],
        chapter_num: int
    ) -> Chapter:
        """
        使用RAG生成章节内容
        """
        logger.debug(f"生成第{chapter_num}章：{chapter.title}")
        
        target_words = chapter.target_words

        # 准备上下文
        context_parts = []
        
        # 1. 最近的章节内容
        if previous_chapters:
            recent_chapter = previous_chapters[-1]
            context_parts.append(f"上一章结尾：\n{recent_chapter.content[-500:]}")
        
        # 2. 从RAG检索相关信息
        if self.use_rag and self.knowledge_base:
            # 检索相关角色信息
            for char_name in chapter.appearing_characters:
                results = self.knowledge_base.search(
                    f"角色 {char_name}",
                    top_k=2
                )
                if results:
                    context_parts.append(f"{char_name}设定：{results[0]['content'][:200]}")
            
            # 检索相关情节
            for event in chapter.key_events:
                results = self.knowledge_base.search(event, top_k=1)
                if results:
                    context_parts.append(f"相关情节：{results[0]['content'][:150]}")
            
            # 检索世界观
            if chapter_num % 10 == 1:  # 每10章提醒一次世界观
                results = self.knowledge_base.search("世界观设定", top_k=1)
                if results:
                    context_parts.append(f"世界观：{results[0]['content'][:200]}")
        
        # 3. 章节任务
        context_parts.append(f"""
本章任务：
- 标题：{chapter.title}
- 字数要求：{chapter.target_words}字
- 节奏：{chapter.pacing}
- 情感基调：{chapter.emotional_tone}
- 出场角色：{', '.join(chapter.appearing_characters)}
- 关键事件：{', '.join(chapter.key_events)}
- 大纲：{chapter.outline}
        """)
        
        full_context = "\n\n".join(context_parts)
        
        # 分段生成内容
        content_segments = []
        current_words = 0
        segment_count = 0
        
        while current_words < target_words * 0.9:
            segment_count += 1
            
            if segment_count == 1:
                # 开篇
                prompt = f"""{full_context}

现在开始写第{chapter_num}章的开篇。
要求：
1. 自然承接上文
2. 引入本章主题
3. 设置悬念
4. 约{min(1500, chapter.target_words // 3)}字
"""
            elif current_words > chapter.target_words * 0.7:
                # 结尾
                prompt = f"""继续写第{chapter_num}章的结尾部分。

前文：{content_segments[-1][-300:] if content_segments else ''}

要求：
1. 完成本章所有事件
2. 留下悬念或转折
3. 约{chapter.target_words - current_words}字
"""
            else:
                # 中间部分
                prompt = f"""继续写第{chapter_num}章。

前文梗概：{content_segments[-1][-200:] if content_segments else ''}
待完成事件：{', '.join(chapter.key_events)}
节奏：{chapter.pacing}

继续创作约1500字。
"""
            
            # 生成内容片段
            segment = self.generator.generate(
                prompt=prompt,
                style=self.style,
                max_new_tokens=2000,
                temperature=0.75,
                use_rag=self.use_rag,
                use_history=True
            )
            
            content_segments.append(segment)
            current_words += len(segment)
            
            # 防止无限循环
            if segment_count > 10:
                logger.warning(f"第{chapter_num}章生成段落过多，强制结束")
                break
        
        # 组合并格式化内容
        full_content = self._format_and_clean_content(content_segments)
        
        chapter.content = full_content
        chapter.actual_words = len(full_content)
        
        # 保存单章
        self._save_single_chapter(chapter)
        
        return chapter
    
    def _evaluate_chapter_quality(self, chapter: Chapter, outline: NovelOutline) -> float:
        """
        评估章节质量
        """
        scores = []
        
        # 1. 长度符合度
        length_ratio = chapter.actual_words / chapter.target_words
        if 0.8 <= length_ratio <= 1.2:
            scores.append(1.0)
        elif 0.6 <= length_ratio <= 1.4:
            scores.append(0.7)
        else:
            scores.append(0.4)
        
        # 2. 角色出现检查
        char_score = 0
        for char in chapter.appearing_characters:
            if char in chapter.content:
                char_score += 1
        scores.append(char_score / max(len(chapter.appearing_characters), 1))
        
        # 3. 事件完成度
        event_score = 0
        for event_keyword in chapter.key_events:
            # 简化的关键词匹配
            if any(word in chapter.content for word in event_keyword.split()[:3]):
                event_score += 1
        scores.append(event_score / max(len(chapter.key_events), 1))
        
        # 4. 连贯性检查（检查是否有明显的断裂）
        paragraphs = chapter.content.split('\n\n')
        if len(paragraphs) > 1:
            scores.append(0.8 if len(paragraphs) < 20 else 0.6)
        else:
            scores.append(0.5)
        
        # 5. 风格一致性（简单检查）
        style_keywords = {
            "仙侠": ["修炼", "灵气", "道", "仙", "法宝"],
            "武侠": ["江湖", "武功", "侠", "剑", "内力"],
            "玄幻": ["魔法", "异界", "等级", "血脉", "天赋"],
        }
        
        if self.style in style_keywords:
            keyword_count = sum(1 for kw in style_keywords[self.style] if kw in chapter.content)
            scores.append(min(keyword_count / 3, 1.0))
        else:
            scores.append(0.7)
        
        return np.mean(scores)
    
    def _regenerate_chapter(
        self,
        chapter: Chapter,
        outline: NovelOutline,
        previous_chapters: List[Chapter],
        chapter_num: int,
        attempt: int = 1
    ) -> Chapter:
        """
        重新生成质量不合格的章节
        """
        if attempt > self.config["max_retries"]:
            logger.warning(f"第{chapter_num}章重试{attempt}次后仍不合格，使用最后版本")
            return chapter
        
        logger.info(f"重新生成第{chapter_num}章（尝试{attempt}）...")
        
        # 调整生成参数
        new_chapter = self._generate_chapter_with_rag(
            chapter,
            outline,
            previous_chapters,
            chapter_num
        )
        
        # 重新评估
        new_score = self._evaluate_chapter_quality(new_chapter, outline)
        new_chapter.quality_score = new_score
        
        if new_score >= self.config["quality_threshold"]:
            logger.success(f"第{chapter_num}章重新生成成功，质量分数：{new_score:.2f}")
            return new_chapter
        else:
            return self._regenerate_chapter(
                chapter, outline, previous_chapters, chapter_num, attempt + 1
            )
    
    def _update_rag_with_chapter(self, chapter: Chapter):
        """
        将新生成的章节内容更新到RAG知识库
        """
        if not self.knowledge_base:
            return
        
        # 创建章节文档
        doc = Document(
            id=f"chapter_{chapter.number}",
            content=f"""第{chapter.number}章：{chapter.title}
            
            摘要：{chapter.outline}
            
            关键内容：{chapter.content[:500]}
            
            出场角色：{', '.join(chapter.appearing_characters)}
            关键事件：{', '.join(chapter.key_events)}
            """,
            metadata={
                "type": "chapter",
                "number": chapter.number,
                "emotional_tone": chapter.emotional_tone,
                "pacing": chapter.pacing
            }
        )
        
        # 生成嵌入并添加
        embedding = self.knowledge_base.embedding_model.encode_documents(
            [doc.content],
            show_progress=False
        )
        
        self.knowledge_base.vector_store.add_documents([doc], embedding)
    
    def _reflect_and_adjust(
        self,
        completed_chapters: List[Chapter],
        remaining_chapters: List[Chapter],
        outline: NovelOutline
    ):
        """
        反思已写内容并调整后续章节计划
        """
        if not self.enable_reflection:
            return
        
        logger.info("🤔 进行阶段性反思与调整...")
        
        # 总结已完成的内容
        summary_prompt = f"""请分析前{len(completed_chapters)}章的内容：

        已完成章节概要：
        {self._summarize_chapters(completed_chapters[-10:])}
        
        原定大纲要点：
        {self._summarize_plot_points(outline.plot_points)}
        
        请回答：
        1. 故事发展是否偏离了原定轨道？
        2. 哪些伏笔需要在后续章节中回收？
        3. 角色发展是否符合预期？
        4. 需要对后续章节做哪些调整？
        
        返回JSON格式：
        {{
            "deviation_level": "none/minor/major",
            "pending_foreshadowing": ["伏笔1", "伏笔2"],
            "character_adjustments": {{"角色名": "调整建议"}},
            "plot_adjustments": ["调整1", "调整2"]
        }}
        """
        
        reflection = self.generator.generate(
            prompt=summary_prompt,
            max_new_tokens=1000,
            temperature=0.6,
            use_rag=True
        )
        
        try:
            reflection_data = json.loads(reflection)
            
            # 根据反思结果调整
            if reflection_data.get("deviation_level") == "major":
                logger.warning("检测到重大偏离，调整后续章节...")
                self._adjust_remaining_chapters(
                    remaining_chapters,
                    reflection_data,
                    outline
                )
            elif reflection_data.get("deviation_level") == "minor":
                logger.info("检测到轻微偏离，微调后续章节...")
                self._minor_adjustments(remaining_chapters, reflection_data)
            
            # 更新待回收的伏笔
            if reflection_data.get("pending_foreshadowing"):
                self._update_foreshadowing(
                    remaining_chapters,
                    reflection_data["pending_foreshadowing"]
                )
            
            logger.success("✅ 反思与调整完成")
            
        except json.JSONDecodeError:
            logger.warning("反思结果解析失败，继续原计划")
    
    def _adjust_remaining_chapters(
        self,
        remaining_chapters: List[Chapter],
        reflection_data: Dict,
        outline: NovelOutline
    ):
        """
        调整剩余章节
        """
        adjustments = reflection_data.get("plot_adjustments", [])
        
        for i, chapter in enumerate(remaining_chapters[:20]):  # 只调整接下来的20章
            if i < len(adjustments):
                # 更新章节大纲
                adjustment_prompt = f"""调整第{chapter.number}章的大纲：
                
                原大纲：{chapter.outline}
                调整建议：{adjustments[i]}
                
                生成新的章节大纲（100-200字）：
                """
                
                new_outline = self.generator.generate(
                    prompt=adjustment_prompt,
                    max_new_tokens=300,
                    temperature=0.7
                )
                
                chapter.outline = new_outline
                logger.debug(f"已调整第{chapter.number}章大纲")
    
    def _format_and_clean_content(self, segments: List[str]) -> str:
        """
        格式化和清理内容
        """
        # 合并段落
        full_text = "\n\n".join(segments)
        
        # 清理重复的标点
        full_text = re.sub(r'。{2,}', '。', full_text)
        full_text = re.sub(r'，{2,}', '，', full_text)
        
        # 添加段落缩进
        paragraphs = full_text.split('\n')
        formatted = []
        
        for para in paragraphs:
            para = para.strip()
            if para and not para.startswith('　　'):
                # 添加首行缩进
                formatted.append('　　' + para)
            elif para:
                formatted.append(para)
        
        return '\n\n'.join(formatted)
    
    def _summarize_chapters(self, chapters: List[Chapter]) -> str:
        """总结章节内容"""
        summary = []
        for ch in chapters:
            summary.append(f"第{ch.number}章：{ch.title} - {ch.outline[:50]}...")
        return "\n".join(summary)
    
    def _summarize_plot_points(self, plot_points: List[PlotPoint]) -> str:
        """总结情节点"""
        summary = []
        for pp in plot_points[:10]:  # 只看前10个
            summary.append(f"第{pp.chapter}章：{pp.event[:50]}...")
        return "\n".join(summary)
    
    def _update_foreshadowing(self, chapters: List[Chapter], foreshadowing: List[str]):
        """更新伏笔回收计划"""
        for fs in foreshadowing:
            # 找一个合适的章节回收伏笔
            target_chapter = np.random.choice(chapters[:10]) if len(chapters) > 10 else chapters[0]
            target_chapter.key_events.append(f"回收伏笔：{fs}")
            logger.debug(f"计划在第{target_chapter.number}章回收伏笔：{fs}")
    
    def _minor_adjustments(self, chapters: List[Chapter], reflection_data: Dict):
        """小幅调整"""
        char_adjustments = reflection_data.get("character_adjustments", {})
        for char_name, adjustment in char_adjustments.items():
            # 找包含该角色的章节
            for ch in chapters[:5]:
                if char_name in ch.appearing_characters:
                    ch.outline += f"\n注意：{adjustment}"
                    break
    
    # ========================================
    # 保存和加载功能
    # ========================================
    
    def save_outline(self, outline: NovelOutline):
        """保存大纲"""
        outline_file = self.output_dir / "outline.json"
        
        # 转换为可序列化格式
        outline_dict = {
            "title": outline.title,
            "genre": outline.genre,
            "theme": outline.theme,
            "world_setting": asdict(outline.world_setting),
            "characters": [asdict(char) for char in outline.characters],
            "plot_points": [asdict(pp) for pp in outline.plot_points],
            "chapter_count": outline.chapter_count,
            "estimated_words": outline.estimated_words,
            "metadata": outline.metadata
        }
        
        with open(outline_file, 'w', encoding='utf-8') as f:
            json.dump(outline_dict, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 大纲已保存: {outline_file}")
    
    def save_chapter_plan(self, chapters: List[Chapter]):
        """保存章节计划"""
        plan_file = self.output_dir / "chapter_plan.json"
        
        chapters_data = []
        for ch in chapters:
            chapters_data.append({
                "number": ch.number,
                "title": ch.title,
                "outline": ch.outline,
                "length_type": ch.length_type.value,
                "target_words": ch.target_words,
                "key_events": ch.key_events,
                "appearing_characters": ch.appearing_characters,
                "emotional_tone": ch.emotional_tone,
                "pacing": ch.pacing
            })
        
        with open(plan_file, 'w', encoding='utf-8') as f:
            json.dump(chapters_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 章节计划已保存: {plan_file}")
    
    def _save_single_chapter(self, chapter: Chapter):
        """保存单个章节"""
        chapter_file = self.output_dir / f"chapters/chapter_{chapter.number:03d}.txt"
        chapter_file.parent.mkdir(exist_ok=True)
        
        with open(chapter_file, 'w', encoding='utf-8') as f:
            f.write(f"{chapter.title}\n")
            f.write(f"字数：{chapter.actual_words} | 质量评分：{chapter.quality_score:.2f}\n")
            f.write("="*50 + "\n\n")
            f.write(chapter.content)
        
        # 同时保存元数据
        meta_file = self.output_dir / f"chapters/chapter_{chapter.number:03d}.json"
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump({
                "number": chapter.number,
                "title": chapter.title,
                "outline": chapter.outline,
                "actual_words": chapter.actual_words,
                "quality_score": chapter.quality_score,
                "emotional_tone": chapter.emotional_tone,
                "pacing": chapter.pacing
            }, f, ensure_ascii=False, indent=2)
    
    def save_progress(self, chapters: List[Chapter], outline: NovelOutline):
        """保存进度"""
        checkpoint_file = self.output_dir / "checkpoint.json"
        
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "current_chapter": self.state['current_chapter'],
            "total_words": self.state['total_words'],
            "completed_chapters": len(chapters),
            "quality_scores": [ch.quality_score for ch in chapters]
        }
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"检查点已保存: {checkpoint_file}")
    
    def load_checkpoint(self) -> Tuple[NovelOutline, List[Chapter]]:
        """加载检查点"""
        # 加载大纲
        outline_file = self.output_dir / "outline.json"
        with open(outline_file, 'r', encoding='utf-8') as f:
            outline_data = json.load(f)
        
        # 重建大纲对象
        # 1. 重建世界观设定
        ws_data = outline_data.get("world_setting", {})
        world_setting = WorldSetting(
            name=ws_data.get("name", "未知世界"),
            description=ws_data.get("description", ""),
            rules=ws_data.get("rules", []),
            locations=ws_data.get("locations", {}),
            power_system=ws_data.get("power_system", "")
        )
        
        # 2. 重建角色列表
        characters = []
        for char_data in outline_data.get("characters", []):
            character = Character(
                name=char_data.get("name", "未命名"),
                role=char_data.get("role", "配角"),
                personality=char_data.get("personality", []),
                background=char_data.get("background", ""),
                relationships=char_data.get("relationships", {}),
                development_arc=char_data.get("development_arc", ""),
                first_appearance=char_data.get("first_appearance", 1)
            )
            characters.append(character)
        
        # 3. 重建情节点
        plot_points = []
        for pp_data in outline_data.get("plot_points", []):
            plot_point = PlotPoint(
                chapter=pp_data.get("chapter", 1),
                event=pp_data.get("event", ""),
                importance=pp_data.get("importance", "medium"),
                foreshadowing=pp_data.get("foreshadowing", []),
                callbacks=pp_data.get("callbacks", [])
            )
            plot_points.append(plot_point)
        
        # 4. 创建完整的大纲对象
        outline = NovelOutline(
            title=outline_data.get("title", "未命名"),
            genre=outline_data.get("genre", self.style),
            theme=outline_data.get("theme", ""),
            world_setting=world_setting,
            characters=characters,
            plot_points=plot_points,
            chapter_count=outline_data.get("chapter_count", 100),
            estimated_words=outline_data.get("estimated_words", 500000),
            metadata=outline_data.get("metadata", {})
        )
        
        # 加载章节计划
        plan_file = self.output_dir / "chapter_plan.json"
        with open(plan_file, 'r', encoding='utf-8') as f:
            chapters_data = json.load(f)
        
        chapters = []
        for ch_data in chapters_data:
            # 重建章节对象
            # 处理枚举类型
            length_type_str = ch_data.get("length_type", "medium")
            length_type_map = {
                "short": ChapterLength.SHORT,
                "medium": ChapterLength.MEDIUM,
                "long": ChapterLength.LONG,
                "epic": ChapterLength.EPIC
            }
            length_type = length_type_map.get(length_type_str, ChapterLength.MEDIUM)
            
            chapter = Chapter(
                number=ch_data.get("number", 1),
                title=ch_data.get("title", "未命名章节"),
                outline=ch_data.get("outline", ""),
                length_type=length_type,
                target_words=ch_data.get("target_words", 5000),
                key_events=ch_data.get("key_events", []),
                appearing_characters=ch_data.get("appearing_characters", []),
                emotional_tone=ch_data.get("emotional_tone", "平静"),
                pacing=ch_data.get("pacing", "medium"),
                content=ch_data.get("content", ""),  # 可能已有部分内容
                actual_words=ch_data.get("actual_words", 0),
                quality_score=ch_data.get("quality_score", 0.0)
            )
            chapters.append(chapter)
        
        # 加载状态
        checkpoint_file = self.output_dir / "checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                self.state.update(checkpoint)
                
            logger.info(f"✅ 从检查点恢复：已完成 {self.state.get('current_chapter', 0)} 章，"
                    f"总字数 {self.state.get('total_words', 0)}")
        
        # 如果启用了RAG，重新索引大纲到知识库
        if self.use_rag and self.knowledge_base:
            logger.info("重新索引大纲到知识库...")
            self.index_outline_to_rag(outline)
            
            # 重新索引已完成的章节
            completed_chapters = self.load_completed_chapters(self.state.get('current_chapter', 0))
            for ch in completed_chapters:
                self._update_rag_with_chapter(ch)
        
        return outline, chapters
    
    def save_checkpoint(self, outline: NovelOutline, chapters: List[Chapter]):
        """保存完整的检查点（包括大纲和章节计划）"""
        # 保存大纲
        self.save_outline(outline)
        
        # 保存章节计划
        self.save_chapter_plan(chapters)
        
        # 保存进度状态
        checkpoint_file = self.output_dir / "checkpoint.json"
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "current_chapter": self.state['current_chapter'],
            "total_words": self.state['total_words'],
            "completed_chapters": self.state.get('current_chapter', 0),
            "quality_scores": self.state.get('quality_scores', []),
            "generation_history": self.state.get('generation_history', [])
        }
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 检查点已保存")
    
    def load_completed_chapters(self, up_to: int) -> List[Chapter]:
        """加载已完成的章节"""
        chapters = []
        
        for i in range(1, up_to + 1):
            chapter_file = self.output_dir / f"chapters/chapter_{i:03d}.txt"
            meta_file = self.output_dir / f"chapters/chapter_{i:03d}.json"
            
            if chapter_file.exists() and meta_file.exists():
                # 加载内容
                with open(chapter_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # 跳过标题和元信息行
                    content = ''.join(lines[3:])
                
                # 加载元数据
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                chapter = Chapter(
                    number=meta['number'],
                    title=meta['title'],
                    outline=meta['outline'],
                    length_type=ChapterLength.MEDIUM,  # 默认值
                    target_words=5000,  # 默认值
                    key_events=[],
                    appearing_characters=[],
                    emotional_tone=meta.get('emotional_tone', ''),
                    pacing=meta.get('pacing', 'medium'),
                    content=content,
                    actual_words=meta['actual_words'],
                    quality_score=meta.get('quality_score', 0.0)
                )
                
                chapters.append(chapter)
        
        return chapters
    
    def post_process_and_compile(
        self,
        title: str,
        chapters: List[Chapter],
        outline: NovelOutline
    ):
        """
        后处理和整合小说
        """
        logger.info("📑 整合小说...")
        
        # 生成完整小说文件
        novel_file = self.output_dir / f"{title}.txt"
        
        with open(novel_file, 'w', encoding='utf-8') as f:
            # 封面信息
            f.write(f"《{title}》\n\n")
            f.write(f"作者：AI创作\n")
            f.write(f"风格：{outline.genre}\n")
            f.write(f"主题：{outline.theme}\n")
            f.write(f"总字数：{sum(ch.actual_words for ch in chapters)}字\n")
            f.write(f"章节数：{len(chapters)}章\n\n")
            f.write("="*60 + "\n\n")
            
            # 目录
            f.write("【目录】\n\n")
            for ch in chapters:
                f.write(f"{ch.title} ({ch.actual_words}字)\n")
            f.write("\n" + "="*60 + "\n\n")
            
            # 正文
            for ch in chapters:
                f.write(f"\n\n{ch.title}\n\n")
                f.write(ch.content)
                f.write("\n\n" + "-"*40 + "\n")
        
        # 生成统计报告
        self._generate_statistics_report(title, chapters, outline)
        
        logger.success(f"""
        ✨ 小说创作完成！
        
        📖 标题：《{title}》
        📝 总字数：{sum(ch.actual_words for ch in chapters):,}字
        📚 章节数：{len(chapters)}章
        ⭐ 平均质量分：{np.mean([ch.quality_score for ch in chapters]):.2f}
        💾 保存位置：{novel_file}
        """)
    
    def _generate_statistics_report(
        self,
        title: str,
        chapters: List[Chapter],
        outline: NovelOutline
    ):
        """生成统计报告"""
        report_file = self.output_dir / "statistics_report.json"
        
        stats = {
            "title": title,
            "genre": outline.genre,
            "theme": outline.theme,
            "total_chapters": len(chapters),
            "total_words": sum(ch.actual_words for ch in chapters),
            "average_chapter_words": np.mean([ch.actual_words for ch in chapters]),
            "quality_scores": {
                "mean": np.mean([ch.quality_score for ch in chapters]),
                "min": min(ch.quality_score for ch in chapters),
                "max": max(ch.quality_score for ch in chapters),
                "std": np.std([ch.quality_score for ch in chapters])
            },
            "pacing_distribution": {},
            "character_appearances": {},
            "generation_time": datetime.now().isoformat()
        }
        
        # 统计节奏分布
        for ch in chapters:
            stats["pacing_distribution"][ch.pacing] = stats["pacing_distribution"].get(ch.pacing, 0) + 1
        
        # 统计角色出场
        for ch in chapters:
            for char in ch.appearing_characters:
                stats["character_appearances"][char] = stats["character_appearances"].get(char, 0) + 1
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 统计报告已生成: {report_file}")
    
    def generate_novel_with_yield(
        self, 
        title: str, 
        target_words: int = 500000,
        progress_callback=None
    ):
        """支持yield的生成方法"""
        
        # 生成大纲
        if progress_callback:
            progress_callback("正在生成大纲...", 0, 0, 0.0, "")
        
        outline = self.generate_structured_outline(title, target_words)
        
        if progress_callback:
            progress_callback("正在规划章节...", 0, 0, 0.0, str(outline.theme))
        
        chapters = self.plan_chapters_dynamically(outline)
        
        # 逐章生成
        for i, chapter in enumerate(chapters):
            if progress_callback:
                progress_callback(
                    f"生成第{i+1}/{len(chapters)}章",
                    i + 1,
                    self.state['total_words'],
                    0.8,
                    chapter.title
                )
            
            # 生成章节内容
            chapter_with_content = self._generate_chapter_with_rag(
                chapter, outline, [], i+1
            )
            
            # 更新进度
            self.state['current_chapter'] = i + 1
            self.state['total_words'] += chapter_with_content.actual_words

    def _generate_fallback_outline(
        self,
        title: str,
        target_words: int,
        estimated_chapters: int
    ) -> NovelOutline:
        """备用大纲生成（当JSON解析失败时）"""
        logger.warning("使用备用方案生成大纲")
        
        # 创建默认世界观
        world_setting = WorldSetting(
            name=f"{title}世界",
            description=f"一个充满{self.style}色彩的世界",
            rules=["规则待定"],
            locations={"主城": "故事发生的主要地点"},
            power_system="待定"
        )
        
        # 创建默认角色
        characters = [
            Character(
                name="主角",
                role="主角",
                personality=["勇敢", "聪明"],
                background="身世成谜",
                relationships={},
                development_arc="从弱到强"
            )
        ]
        
        # 创建默认情节点
        plot_points = [
            PlotPoint(
                chapter=1,
                event="故事开始",
                importance="high",
                foreshadowing=[]
            ),
            PlotPoint(
                chapter=estimated_chapters // 2,
                event="转折点",
                importance="high",
                foreshadowing=[]
            ),
            PlotPoint(
                chapter=estimated_chapters,
                event="大结局",
                importance="high",
                foreshadowing=[]
            )
        ]
        
        return NovelOutline(
            title=title,
            genre=self.style,
            theme="成长与冒险",
            world_setting=world_setting,
            characters=characters,
            plot_points=plot_points,
            chapter_count=estimated_chapters,
            estimated_words=target_words
        )


# ========================================
# 主函数
# ========================================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="增强版长篇小说生成系统")
    parser.add_argument("--title", type=str, required=True, help="小说标题")
    parser.add_argument("--words", type=int, default=500000, help="目标字数")
    parser.add_argument("--style", type=str, default="仙侠", help="小说风格")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--use-rag", action="store_true", help="启用RAG增强")
    parser.add_argument("--enable-reflection", action="store_true", help="启用反思机制")
    parser.add_argument("--resume", action="store_true", help="从检查点恢复")
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = EnhancedNovelGenerator(
        model_path=args.model,
        style=args.style,
        use_rag=args.use_rag,
        enable_reflection=args.enable_reflection
    )
    
    # 生成小说
    generator.generate_novel(
        title=args.title,
        target_words=args.words,
        resume_from_checkpoint=args.resume
    )


if __name__ == "__main__":
    main()