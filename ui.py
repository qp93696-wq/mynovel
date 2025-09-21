
"""
ui/gradio_app.py - Gradio Web界面
"""

import gradio as gr
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime, time
from loguru import logger

from rag.knowledge_base import NovelKnowledgeBase
from generation.rag_generator import RAGNovelGenerator
from generation.novel_generator import EnhancedNovelGenerator
from train.post_training_pipeline import PostTrainingConfig, PostTrainingPipeline


class NovelRAGApp:
    """Novel-RAG Gradio应用"""
    
    def __init__(self):
        """初始化应用"""
        self.knowledge_base = None
        self.generator = None
        self.training_pipeline = None
        self.long_novel_generator = None
        self.generation_thread = None
        self.generation_running = False
        self.generation_paused = False
        self.generation_active = False
        self.model_path = None
        
        
        # 初始化组件
        self._init_components()
    
    def generate_novel_stream(self, title, style, target_words, use_rag, enable_reflection, auto_save_interval, quality_threshold):
        """改进版的流式生成"""
        try:
            from generation.novel_generator import EnhancedNovelGenerator
            
            # 用于存储进度信息
            progress_info = {
                'status': '',
                'chapter': 0,
                'words': 0,
                'quality': 0.0,
                'preview': ''
            }
            
            def progress_callback(status, chapter, words, quality, preview):
                """进度回调"""
                progress_info['status'] = status
                progress_info['chapter'] = chapter
                progress_info['words'] = words
                progress_info['quality'] = quality
                progress_info['preview'] = preview
            
            # 初始化生成器
            self.long_novel_generator = EnhancedNovelGenerator(
                model_path=self.model_path,
                style=style,
                use_rag=use_rag,
                enable_reflection=enable_reflection
            )
            
            # 在新线程中运行生成
            import threading
            
            def generate_thread():
                self.long_novel_generator.generate_novel_with_yield(
                    title=title,
                    target_words=target_words,
                    progress_callback=progress_callback
                )
            
            thread = threading.Thread(target=generate_thread)
            thread.start()
            
            # 持续yield进度
            while thread.is_alive() or progress_info['status']:
                yield (
                    progress_info['status'],
                    progress_info['chapter'],
                    progress_info['words'],
                    progress_info['quality'],
                    progress_info['preview']
                )
                time.sleep(1)  # 每秒更新一次
            
            yield "生成完成！", progress_info['chapter'], progress_info['words'], progress_info['quality'], "完成"
            
        except Exception as e:
            yield f"错误: {str(e)}", 0, 0, 0.0, str(e)

    def download_novel(self):
        """下载生成的小说"""
        if self.long_novel_generator and self.long_novel_generator.output_dir:
            novel_files = list(self.long_novel_generator.output_dir.glob("*.txt"))
            if novel_files:
                return str(novel_files[0])
        return None

    def download_outline(self):
        """下载大纲"""
        if self.long_novel_generator and self.long_novel_generator.output_dir:
            outline_file = self.long_novel_generator.output_dir / "outline.json"
            if outline_file.exists():
                return str(outline_file)
        return None
    
    def open_output_folder(self):
        """打开输出目录"""
        if self.long_novel_generator and self.long_novel_generator.output_dir:
            import os
            import platform
            
            path = str(self.long_novel_generator.output_dir)
            
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":  # macOS
                os.system(f"open {path}")
            else:  # Linux
                os.system(f"xdg-open {path}")
            
            return path
        return "未找到输出目录"

    def start_long_novel_generation(
        self,
        title: str,
        style: str,
        target_words: int,
        use_rag: bool,
        enable_reflection: bool,
        auto_save_interval: int,
        quality_threshold: float,
        progress=gr.Progress()
    ):
        """开始长篇小说生成"""
        try:
            from generation.novel_generator import EnhancedNovelGenerator
            import threading
            
            # 检查是否已有生成任务
            if self.generation_running:
                return "已有生成任务正在运行", "", "", 0, 0, 0.0
            
            # 初始化生成器
            self.long_novel_generator = EnhancedNovelGenerator(
                model_path=self.config.model.model_name_or_path,
                style=style,
                use_rag=use_rag,
                enable_reflection=enable_reflection
            )
            
            # 设置配置
            self.long_novel_generator.config['auto_save_interval'] = auto_save_interval
            self.long_novel_generator.config['quality_threshold'] = quality_threshold
            
            # 创建生成线程
            def generate_with_progress():
                try:
                    self.generation_running = True
                    
                    # 使用回调函数更新进度
                    def progress_callback(chapter_num, total_chapters, words, quality, preview):
                        if not self.generation_paused:
                            progress((chapter_num / total_chapters, f"第{chapter_num}/{total_chapters}章"))
                            # 这里需要通过某种方式更新UI
                            # 可以使用队列或者其他机制
                    
                    self.long_novel_generator.generate_novel(
                        title=title,
                        target_words=target_words,
                        progress_callback=progress_callback
                    )
                    
                except Exception as e:
                    logger.error(f"生成失败: {e}")
                finally:
                    self.generation_running = False
            
            # 启动线程
            self.generation_thread = threading.Thread(target=generate_with_progress)
            self.generation_thread.start()
            
            return (
                f"开始生成《{title}》...",
                f"输出目录：{self.long_novel_generator.output_dir}",
                "",  # 预览
                0,   # 当前章节
                0,   # 总字数
                0.0  # 质量分
            )
            
        except Exception as e:
            logger.error(f"启动生成失败: {e}")
            return f"错误: {str(e)}", "", "", 0, 0, 0.0
    
    def pause_generation(self):
        """暂停生成"""
        if self.generation_running:
            self.generation_paused = True
            return "生成已暂停"
        return "没有正在运行的生成任务"
    
    def resume_generation(self):
        """恢复生成"""
        if self.generation_paused:
            self.generation_paused = False
            return "生成已恢复"
        return "没有暂停的任务"
    
    def stop_generation(self):
        """停止生成"""
        if self.generation_running:
            self.generation_running = False
            if self.generation_thread:
                self.generation_thread.join(timeout=5)
            return "生成已停止"
        return "没有正在运行的生成任务"
    
    def get_generation_status(self):
        """获取生成状态（用于定时更新）"""
        import numpy as np
        if not self.long_novel_generator:
            return "", 0, 0, 0.0, ""
        
        state = self.long_novel_generator.state
        
        # 获取最新章节预览
        preview = ""
        if state['current_chapter'] > 0:
            try:
                chapter_file = (self.long_novel_generator.output_dir / 
                               f"chapters/chapter_{state['current_chapter']:03d}.txt")
                if chapter_file.exists():
                    with open(chapter_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        preview = ''.join(lines[:50])  # 前50行
            except:
                pass
        
        # 计算平均质量分
        avg_quality = np.mean(state.get('quality_scores', [0])) if state.get('quality_scores') else 0
        
        status = f"正在生成第{state['current_chapter']}章..."
        
        return (
            status,
            state['current_chapter'],
            state['total_words'],
            avg_quality,
            preview
        )

    def _init_components(self):
        """初始化组件"""
        # 知识库
        self.knowledge_base = NovelKnowledgeBase(
            embedding_model_name="BAAI/bge-small-zh-v1.5",
            vector_store_path="./data/vector_store/novels"
        )
    
    def get_generator(self, model_name: str) -> RAGNovelGenerator:
        """根据模型名称获取或创建生成器实例"""
        # 这里可以加入缓存机制，避免重复加载
        return RAGNovelGenerator(
            model_name=model_name,
            knowledge_base=self.knowledge_base
        )
    
    def get_long_novel_generator(self, model_name: str, style: str, use_rag: bool, enable_reflection: bool) -> EnhancedNovelGenerator:
        """获取或创建长篇小说生成器"""
        return EnhancedNovelGenerator(
            model_path=model_name,
            style=style,
            use_rag=use_rag,
            enable_reflection=enable_reflection
        )
    
    def build_interface(self) -> gr.Blocks:
        """构建Gradio界面"""
        
        with gr.Blocks(title="小说生成系统", theme=gr.themes.Soft()) as app:
            gr.Markdown("""
            # 🎭 智能小说生成系统
            
            支持多种风格、知识库管理和模型训练。
            """)
            
            with gr.Tabs():
                # Tab 1: 模型训练
                with gr.Tab("🎯 模型训练"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 训练配置")
                            
                            train_model = gr.Dropdown(
                                label="基础模型",
                                choices=[
                                    "Qwen/Qwen2-1.5B-Instruct",
                                    "Qwen/Qwen2.5-3B-Instruct",
                                    "Qwen/Qwen2.5-0.5B-Instruct"
                                ],
                                value="Qwen/Qwen2-1.5B-Instruct"
                            )
                            
                            train_styles = gr.CheckboxGroup(
                                label="训练风格",
                                choices=["仙侠", "武侠", "玄幻", "都市", "科幻"],
                                value=[]
                            )
                            
                            with gr.Row():
                                sft_epochs = gr.Number(label="SFT轮数", value=3)
                                dpo_epochs = gr.Number(label="DPO轮数", value=2)
                            
                            with gr.Row():
                                lora_r = gr.Number(label="LoRA秩", value=8)
                                lora_alpha = gr.Number(label="LoRA Alpha", value=16)
                            
                            do_sft = gr.Checkbox(label="执行SFT训练", value=True)
                            do_dpo = gr.Checkbox(label="执行DPO训练", value=True)
                            do_eval = gr.Checkbox(label="执行评估", value=True)
                            
                            start_training_btn = gr.Button("🚀 开始训练", variant="primary")
                        
                        with gr.Column():
                            gr.Markdown("### 训练进度")
                            training_status = gr.Textbox(label="状态", lines=10)
                            training_progress = gr.Progress()
                            
                            gr.Markdown("### 训练结果")
                            training_results = gr.JSON(label="评估结果")


                # Tab 2: 知识库管理
                with gr.Tab("📚 知识库管理"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 添加小说到知识库")
                            
                            upload_file = gr.File(
                                label="上传小说文件",
                                file_types=[".txt"],
                                file_count="multiple"
                            )
                            
                            kb_style = gr.Dropdown(
                                label="小说风格",
                                choices=["仙侠", "武侠", "玄幻", "都市", "科幻"],
                                value="仙侠"
                            )
                            
                            add_to_kb_btn = gr.Button("📥 添加到知识库", variant="primary")
                            
                            kb_status = gr.Textbox(label="状态", lines=3)
                        
                        with gr.Column():
                            gr.Markdown("### 知识库统计")
                            kb_stats = gr.JSON(label="统计信息")
                            
                            refresh_stats_btn = gr.Button("🔄 刷新统计")
                            
                            gr.Markdown("### 搜索测试")
                            search_query = gr.Textbox(
                                label="搜索查询",
                                placeholder="输入搜索内容"
                            )
                            search_btn = gr.Button("🔍 搜索")
                            search_results = gr.JSON(label="搜索结果")

                # Tab 3: 小说生成
                with gr.Tab("📝 小说生成"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            generate_model = gr.Dropdown(
                                label="生成模型",
                                choices=[
                                "Qwen/Qwen2-1.5B-Instruct",
                                "Qwen/Qwen2.5-3B-Instruct"
                            ],
                            value="Qwen/Qwen2-1.5B-Instruct"
                        )
                            generate_prompt = gr.Textbox(
                                label="创作提示",
                                placeholder="输入你的创作需求，如：写一段主角突破境界的场景",
                                lines=3
                            )
                            
                            generate_style = gr.Dropdown(
                                label="小说风格",
                                choices=["仙侠", "武侠", "玄幻", "都市", "科幻"],
                                value="仙侠"
                            )
                            
                            with gr.Accordion("高级设置", open=False):
                                use_rag = gr.Checkbox(label="使用RAG增强", value=True)
                                max_tokens = gr.Slider(
                                    label="最大长度",
                                    minimum=100,
                                    maximum=2000,
                                    value=512,
                                    step=50
                                )
                                temperature = gr.Slider(
                                    label="创造性(Temperature)",
                                    minimum=0.1,
                                    maximum=1.5,
                                    value=0.7,
                                    step=0.1
                                )
                                top_p = gr.Slider(
                                    label="Top-p",
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=0.9,
                                    step=0.1
                                )
                            
                            generate_btn = gr.Button("🎨 开始创作", variant="primary")
                        
                        with gr.Column(scale=2):
                            generated_text = gr.Textbox(
                                label="生成内容",
                                lines=20,
                                max_lines=30
                            )
                            
                            with gr.Row():
                                save_btn = gr.Button("💾 保存")
                                clear_btn = gr.Button("🗑️ 清空")
                
                
                
                # Tab 4: 批量生成
                with gr.Tab("📦 批量生成"):
                    with gr.Row():
                        with gr.Column():
                            batch_prompts = gr.Textbox(
                                label="批量提示（每行一个）",
                                lines=10,
                                placeholder="第一个提示\n第二个提示\n..."
                            )
                            
                            batch_style = gr.Dropdown(
                                label="统一风格",
                                choices=["仙侠", "武侠", "玄幻", "都市", "科幻"],
                                value="仙侠"
                            )
                            
                            batch_generate_btn = gr.Button("🎨 批量生成", variant="primary")
                        
                        with gr.Column():
                            batch_results = gr.Dataframe(
                                headers=["提示", "生成内容"],
                                label="生成结果"
                            )
                            
                            export_btn = gr.Button("📤 导出结果")

                 # Tab 5: 长篇小说生成
                with gr.Tab("📖 长篇生成"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 长篇小说设置")
                            
                            long_novel_model = gr.Dropdown(
                            label="生成模型",
                            choices=[
                                "Qwen/Qwen2-1.5B-Instruct",
                                "Qwen/Qwen2.5-3B-Instruct"
                            ],
                            value="Qwen/Qwen2-1.5B-Instruct"
                            )
                            long_novel_title = gr.Textbox(
                                label="小说标题",
                                placeholder="输入小说名称，如：凡人修仙传",
                                value=""
                            )
                            
                            long_novel_style = gr.Dropdown(
                                label="小说风格",
                                choices=["仙侠", "武侠", "玄幻", "都市", "科幻"],
                                value="仙侠"
                            )
                            
                            target_words = gr.Slider(
                                label="目标字数",
                                minimum=10000,
                                maximum=1000000,
                                value=500000,
                                step=10000
                            )
                            
                            chapters_count = gr.Number(
                                label="章节数（自动计算）",
                                value=100,
                                interactive=False
                            )
                            
                            with gr.Accordion("高级设置", open=False):
                                use_rag_long = gr.Checkbox(
                                    label="启用RAG增强（推荐）",
                                    value=True
                                )
                                
                                enable_reflection = gr.Checkbox(
                                    label="启用反思机制",
                                    value=True,
                                    info="每10章反思并调整后续内容"
                                )
                                
                                auto_save_interval = gr.Slider(
                                    label="自动保存间隔（章）",
                                    minimum=1,
                                    maximum=10,
                                    value=3
                                )
                                
                                quality_threshold = gr.Slider(
                                    label="质量阈值",
                                    minimum=0.5,
                                    maximum=0.95,
                                    value=0.7,
                                    step=0.05
                                )
                            
                    
                            start_generation_btn = gr.Button(
                                    "🚀 开始生成", 
                                    variant="primary"
                                )
                            
                            stop_generation_btn = gr.Button(
                                "⏹️ 停止生成", 
                                variant="stop"
                                )
            
                        
                        with gr.Column(scale=2):
                            gr.Markdown("### 生成进度")
                            
                            # 进度显示
                            generation_status = gr.Textbox(
                                label="当前状态",
                                lines=3,
                                value="等待开始..."
                            )
                            
                            progress_bar = gr.Progress()
                            
                            with gr.Row():
                                current_chapter_num = gr.Number(
                                    label="当前章节",
                                    value=0
                                )
                                
                                total_words_count = gr.Number(
                                    label="已生成字数",
                                    value=0
                                )
                                
                                avg_quality_score = gr.Number(
                                    label="平均质量分",
                                    value=0.0
                                )
                            
                            # 实时预览
                            chapter_preview = gr.Textbox(
                                label="章节预览",
                                lines=15,
                                max_lines=20,
                                value=""
                            )
                            
                            # 生成日志
                            generation_log = gr.Textbox(
                                label="生成日志",
                                lines=10,
                                value=""
                            )
                    
                    # 文件管理行
                    with gr.Row():
                        gr.Markdown("### 文件管理")
                        
                        download_novel_btn = gr.Button("📥 下载小说")
                        download_outline_btn = gr.Button("📋 下载大纲")
                        open_folder_btn = gr.Button("📁 打开输出目录")
                        
                        output_path_display = gr.Textbox(
                            label="输出路径",
                            interactive=False,
                            value=""
                        )
            # 绑定事件
            start_generation_btn.click(
                fn=self.generate_novel_stream,
                inputs=[
                    long_novel_title,
                    long_novel_style,
                    target_words,
                    use_rag_long,
                    enable_reflection,
                    auto_save_interval,
                    quality_threshold
                    ],
                outputs=[
                    generation_status,
                    current_chapter_num,
                    total_words_count,
                    avg_quality_score,
                    chapter_preview
                ]
            )
            
            # 下载小说
            download_novel_btn.click(
                fn=self.download_novel,
                outputs=gr.File(label="下载文件")
            )

            # 下载大纲
            download_outline_btn.click(
                fn=self.download_outline,
                outputs=gr.File(label="大纲文件")
            )

            # 打开文件夹
            open_folder_btn.click(
                fn=self.open_output_folder,
                outputs=output_path_display
            )

            stop_generation_btn.click(
                fn=self.stop_generation,
                outputs=[generation_status]
            )
            
            # 自动计算章节数
            target_words.change(
                fn=lambda words: words // 5000,
                inputs=target_words,
                outputs=chapters_count
            )

            self._bind_events(
                generate_btn, generate_prompt, generate_style, use_rag,
                max_tokens, temperature, top_p, generated_text,
                save_btn, clear_btn,
                upload_file, kb_style, add_to_kb_btn, kb_status,
                kb_stats, refresh_stats_btn,
                search_query, search_btn, search_results,
                train_model, train_styles, sft_epochs, dpo_epochs,
                lora_r, lora_alpha, do_sft, do_dpo, do_eval,
                start_training_btn, training_status, training_results,
                batch_prompts, batch_style, batch_generate_btn, batch_results,
                export_btn
            )
        
        return app
    
    def _bind_events(self, *components):
        """绑定事件处理"""
        (generate_btn, generate_prompt, generate_style, use_rag,
         max_tokens, temperature, top_p, generated_text,
         save_btn, clear_btn,
         upload_file, kb_style, add_to_kb_btn, kb_status,
         kb_stats, refresh_stats_btn,
         search_query, search_btn, search_results,
         train_model, train_styles, sft_epochs, dpo_epochs,
         lora_r, lora_alpha, do_sft, do_dpo, do_eval,
         start_training_btn, training_status, training_results,
         batch_prompts, batch_style, batch_generate_btn, batch_results,
         export_btn) = components
        
        # 短篇生成事件
        '''generate_btn.click(
            fn=self.generate_novel,
            inputs=[generate_model,
                    generate_prompt, 
                    generate_style, 
                    use_rag,
                    max_tokens,
                    temperature, 
                    top_p
                ],
            outputs=generated_text
        )'''


    

        # 清空按钮
        clear_btn.click(
            fn=lambda: "",
            outputs=generated_text
        )
        
        # 保存按钮
        save_btn.click(
            fn=self.save_generated,
            inputs=[generated_text, generate_style],
            outputs=kb_status
        )
        
        # 知识库管理
        add_to_kb_btn.click(
            fn=self.add_to_knowledge_base,
            inputs=[upload_file, kb_style],
            outputs=[kb_status, kb_stats]
        )
        
        refresh_stats_btn.click(
            fn=self.get_kb_statistics,
            outputs=kb_stats
        )
        
        search_btn.click(
            fn=self.search_knowledge_base,
            inputs=[search_query, kb_style],
            outputs=search_results
        )
        
        # 训练事件
        start_training_btn.click(
            fn=self.start_training,
            inputs=[train_model, train_styles, sft_epochs, dpo_epochs,
                   lora_r, lora_alpha, do_sft, do_dpo, do_eval],
            outputs=[training_status, training_results]
        )
        
        # 批量生成
        batch_generate_btn.click(
            fn=self.batch_generate,
            inputs=[batch_prompts, batch_style],
            outputs=batch_results
        )
        
        export_btn.click(
            fn=self.export_results,
            inputs=batch_results,
            outputs=kb_status
        )
    
    def generate_novel(
        self,
        model_name: str,
        prompt: str,
        style: str,
        use_rag: bool,
        max_tokens: int,
        temperature: float,
        top_p: float
    ) -> str:
        """生成小说"""
        try:
            generator = self.get_generator(model_name)
            generated = generator.generate(
                prompt=prompt,
                style=style,
                use_rag=use_rag,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            return generated
        except Exception as e:
            logger.error(f"生成失败: {e}")
            return f"生成失败: {str(e)}"
    
    '''def start_long_novel_generation(
        self,
        model_name: str, # <--- 接收模型名称
        title: str,
        style: str,
        target_words: int,
        use_rag: bool,
        enable_reflection: bool,
        auto_save_interval: int,
        quality_threshold: float,
        progress=gr.Progress()
    ):
        """开始长篇小说生成"""
        try:
            import threading

            if self.generation_running:
                return "已有生成任务正在运行", "", "", 0, 0, 0.0

            # 动态创建长篇小说生成器
            self.long_novel_generator = self.get_long_novel_generator(
                model_name=model_name,
                style=style,
                use_rag=use_rag,
                enable_reflection=enable_reflection
            )
            
            # ... (后续代码与原 start_long_novel_generation 类似)
            # 设置配置
            self.long_novel_generator.config['auto_save_interval'] = auto_save_interval
            self.long_novel_generator.config['quality_threshold'] = quality_threshold

            # 创建生成线程
            def generate_with_progress():
                # ... (线程内代码不变)
            
            self.generation_thread = threading.Thread(target=generate_with_progress)
            self.generation_thread.start()

            return (
                f"开始生成《{title}》...",
                f"输出目录：{self.long_novel_generator.output_dir}",
                "", 0, 0, 0.0
            )

        except Exception as e:
            logger.error(f"启动生成失败: {e}")
            return f"错误: {str(e)}", "", "", 0, 0, 0.0'''
        
    def save_generated(self, text: str, style: str) -> str:
        """保存生成的内容"""
        try:
            save_dir = Path("outputs/generated")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = save_dir / f"{style}_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(text)
            
            return f"已保存到: {filename}"
        except Exception as e:
            return f"保存失败: {str(e)}"
    
    def add_to_knowledge_base(
        self,
        files: List[Any],
        style: str
    ) -> Tuple[str, Dict]:
        """添加到知识库"""
        try:
            if not files:
                return "请上传文件", self.get_kb_statistics()
            
            total_added = 0
            for file in files:
                added = self.knowledge_base.add_novel(
                    file.name,
                    style
                )
                total_added += added
            
            self.knowledge_base.save()
            
            return f"成功添加 {total_added} 个文档", self.get_kb_statistics()
        except Exception as e:
            return f"添加失败: {str(e)}", self.get_kb_statistics()
    
    def get_kb_statistics(self) -> Dict:
        """获取知识库统计"""
        return self.knowledge_base.get_statistics()
    
    def search_knowledge_base(self, query: str, style: str) -> List[Dict]:
        """搜索知识库"""
        try:
            results = self.knowledge_base.search(query, top_k=5, style=style)
            return results
        except Exception as e:
            return [{"error": str(e)}]
    
    def start_training(
        self,
        model: str,
        styles: List[str],
        sft_epochs: int,
        dpo_epochs: int,
        sft_lora_r: int,
        sft_lora_alpha: int,
        sft_enabled: bool,
        dpo_enabled: bool,
        eval_enabled: bool
    ) -> Tuple[str, Dict]:
        """开始训练"""
        try:
            config = PostTrainingConfig(
                base_model=model,
                styles=styles,
                sft_epochs=int(sft_epochs),
                dpo_epochs=int(dpo_epochs),
                sft_lora_r=int(sft_lora_r),
                sft_lora_alpha=int(sft_lora_alpha),
                sft_enabled=sft_enabled,
                dpo_enabled=dpo_enabled,
                eval_enabled=eval_enabled
            )
            
            pipeline = PostTrainingPipeline(config)
            pipeline.run()
            
            return "训练完成！", {"status": "success"}
        except Exception as e:
            return f"训练失败: {str(e)}", {"status": "failed", "error": str(e)}
    
    def batch_generate(
        self,
        prompts_text: str,
        style: str
    ) -> List[List[str]]:
        """批量生成"""
        prompts = [p.strip() for p in prompts_text.split('\n') if p.strip()]
        results = []
        
        for prompt in prompts:
            try:
                generated = self.generator.generate(
                    prompt=prompt,
                    style=style,
                    use_rag=True
                )
                results.append([prompt, generated])
            except Exception as e:
                results.append([prompt, f"生成失败: {str(e)}"])
        
        return results
    
    def export_results(self, results: List[List[str]]) -> str:
        """导出结果"""
        try:
            save_dir = Path("outputs/batch")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = save_dir / f"batch_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            return f"已导出到: {filename}"
        except Exception as e:
            return f"导出失败: {str(e)}"