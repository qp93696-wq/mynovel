
"""
ui/gradio_app.py - Gradio Web界面
"""

import gradio as gr
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
from loguru import logger

from rag.knowledge_base import NovelKnowledgeBase
from generation.rag_generator import RAGNovelGenerator
from train.post_training_pipeline import PostTrainingConfig, PostTrainingPipeline


class NovelRAGApp:
    """Novel-RAG Gradio应用"""
    
    def __init__(self):
        """初始化应用"""
        self.knowledge_base = None
        self.generator = None
        self.training_pipeline = None
        
        # 初始化组件
        self._init_components()
    
    def _init_components(self):
        """初始化组件"""
        # 知识库
        self.knowledge_base = NovelKnowledgeBase(
            embedding_model_name="BAAI/bge-small-zh-v1.5",
            vector_store_path="./data/vector_store/novels"
        )
        
        # 生成器
        self.generator = RAGNovelGenerator(
            model_name="Qwen/Qwen2-1.5B-Instruct",
            knowledge_base=self.knowledge_base
        )
    
    def build_interface(self) -> gr.Blocks:
        """构建Gradio界面"""
        
        with gr.Blocks(title="Novel-RAG 小说生成系统", theme=gr.themes.Soft()) as app:
            gr.Markdown("""
            # 🎭 Novel-RAG 智能小说生成系统
            
            基于RAG技术的小说生成系统，支持多种风格、知识库管理和模型训练。
            """)
            
            with gr.Tabs():
                # Tab 1: 小说生成
                with gr.Tab("📝 小说生成"):
                    with gr.Row():
                        with gr.Column(scale=1):
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
                
                # Tab 3: 模型训练
                with gr.Tab("🎯 模型训练"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 训练配置")
                            
                            train_model = gr.Dropdown(
                                label="基础模型",
                                choices=[
                                    "Qwen/Qwen2-1.5B-Instruct",
                                    "Qwen/Qwen2-7B-Instruct",
                                    "THUDM/chatglm3-6b"
                                ],
                                value="Qwen/Qwen2-1.5B-Instruct"
                            )
                            
                            train_styles = gr.CheckboxGroup(
                                label="训练风格",
                                choices=["仙侠", "武侠", "玄幻", "都市", "科幻"],
                                value=["仙侠", "武侠", "玄幻"]
                            )
                            
                            with gr.Row():
                                sft_epochs = gr.Number(label="SFT轮数", value=3)
                                dpo_epochs = gr.Number(label="DPO轮数", value=2)
                            
                            with gr.Row():
                                lora_r = gr.Number(label="LoRA秩", value=16)
                                lora_alpha = gr.Number(label="LoRA Alpha", value=32)
                            
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
            
            # 绑定事件
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
        
        # 生成事件
        generate_btn.click(
            fn=self.generate_novel,
            inputs=[generate_prompt, generate_style, use_rag,
                   max_tokens, temperature, top_p],
            outputs=generated_text
        )
        
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
        prompt: str,
        style: str,
        use_rag: bool,
        max_tokens: int,
        temperature: float,
        top_p: float
    ) -> str:
        """生成小说"""
        try:
            generated = self.generator.generate(
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
        lora_r: int,
        lora_alpha: int,
        do_sft: bool,
        do_dpo: bool,
        do_eval: bool
    ) -> Tuple[str, Dict]:
        """开始训练"""
        try:
            config = PostTrainingConfig(
                base_model=model,
                styles=styles,
                sft_epochs=int(sft_epochs),
                dpo_epochs=int(dpo_epochs),
                lora_r=int(lora_r),
                lora_alpha=int(lora_alpha),
                do_sft=do_sft,
                do_dpo=do_dpo,
                do_evaluation=do_eval
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