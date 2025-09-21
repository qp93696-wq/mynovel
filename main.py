"""
main.py - Novel-RAG 项目主入口
提供命令行界面和多种运行模式
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import yaml
from loguru import logger
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 配置日志
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """配置日志系统"""
    logger.remove()  # 移除默认处理器
    
    # 控制台输出
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # 文件输出
    if log_file:
        logger.add(
            log_file,
            level=log_level,
            rotation="10 MB",
            retention="7 days",
            encoding="utf-8"
        )


class NovelRAGCLI:
    """Novel-RAG 命令行界面"""
    
    def __init__(self):
        self.config = None
        self.load_system_config()
    
    def load_system_config(self):
        """加载系统配置"""
        try:
            from src.config.config import SystemConfig
            self.config = SystemConfig()
            logger.info("系统配置加载成功")
        except ImportError as e:
            logger.error(f"无法加载系统配置: {e}")
            self.config = None
    
    # ========================================
    # 生成相关命令
    # ========================================
    
    def generate(self, args):
        """生成小说内容"""
        logger.info("启动小说生成...")
        
        try:
            from generation.rag_generator import RAGNovelGenerator
            from rag.knowledge_base import NovelKnowledgeBase
            
            # 初始化知识库
            kb = None
            if args.use_rag:
                kb = NovelKnowledgeBase(
                    embedding_model_name=args.embedding_model or "BAAI/bge-small-zh-v1.5",
                    vector_store_path=args.vector_store or "./data/vector_store/novels"
                )
                logger.info(f"知识库加载完成，包含 {kb.vector_store.index.ntotal} 个文档")
            
            # 初始化生成器
            generator = RAGNovelGenerator(
                model_name=args.model or self.config.model.model_name_or_path,
                knowledge_base=kb,
                device=args.device
            )
            
            # 生成内容
            if args.stream:
                # 流式生成
                logger.info(f"开始流式生成，提示: {args.prompt[:50]}...")
                for chunk in generator.generate_stream(
                    prompt=args.prompt,
                    style=args.style,
                    use_rag=args.use_rag,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p
                ):
                    print(chunk, end='', flush=True)
                print()  # 换行
            else:
                # 普通生成
                logger.info(f"开始生成，提示: {args.prompt[:50]}...")
                result = generator.generate(
                    prompt=args.prompt,
                    style=args.style,
                    use_rag=args.use_rag,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p
                )
                print("\n" + "="*50)
                print("生成结果：")
                print("="*50)
                print(result)
            
            # 保存结果
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(result if not args.stream else "流式输出未保存")
                logger.success(f"结果已保存到: {args.output}")
            
            # 清理历史（如果需要）
            if args.clear_history:
                generator.clear_history()
                logger.info("历史记录已清空")
                
        except Exception as e:
            logger.error(f"生成失败: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1
        
        return 0
    
    # ========================================
    # 训练相关命令
    # ========================================
    
    def train(self, args):
        """执行训练流程"""
        logger.info("启动训练流程...")
        
        try:
            from train.post_training_pipeline import PostTrainingConfig, PostTrainingPipeline
            
            # 创建或加载配置
            if args.config:
                config = PostTrainingConfig.load(args.config)
                logger.info(f"从配置文件加载: {args.config}")
            else:
                config = PostTrainingConfig(
                    project_name=args.project or "novel_rag_training",
                    base_model=r"D:\Project\novel\models\transformers_cache\models_Qwen_Qwen2.5-0.5B-Instruct",
                    novel_data_dir=args.data_dir or "./data/novels",
                    styles=args.styles or ["仙侠", "武侠", "玄幻"],
                    max_samples_per_style=args.max_samples or 1000,
                    sft_enabled=not args.skip_sft,
                    sft_epochs=args.sft_epochs or 3,
                    dpo_enabled=not args.skip_dpo,
                    dpo_epochs=args.dpo_epochs or 2,
                    eval_enabled=not args.skip_eval,
                    use_wandb=args.wandb
                )
            
            # 创建训练管道
            pipeline = PostTrainingPipeline(config)
            
            # 执行训练
            if args.resume:
                pipeline.run_with_checkpoint()
            elif args.from_step:
                pipeline.resume(from_step=args.from_step)
            else:
                pipeline.run()
            
            logger.success("训练完成！")
            
        except Exception as e:
            logger.error(f"训练失败: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1
        
        return 0
    
    # ========================================
    # 知识库相关命令
    # ========================================
    
    def knowledge_base(self, args):
        """管理知识库"""
        logger.info(f"执行知识库操作: {args.kb_action}")
        
        try:
            from rag.knowledge_base import NovelKnowledgeBase
            
            # 初始化知识库
            kb = NovelKnowledgeBase(
                embedding_model_name=args.embedding_model or "BAAI/bge-small-zh-v1.5",
                vector_store_path=args.vector_store or "./data/vector_store/novels",
                chunk_size=args.chunk_size or 500,
                chunk_overlap=args.chunk_overlap or 100
            )
            
            if args.kb_action == "add":
                # 添加小说到知识库
                if args.novel_path:
                    count = kb.add_novel(
                        args.novel_path,
                        args.style or "未分类",
                        metadata={"source": "manual_add"}
                    )
                    logger.success(f"添加了 {count} 个文档块")
                elif args.novel_dir:
                    count = kb.add_novels_batch(
                        args.novel_dir,
                        style_mapping=None  # 自动推断
                    )
                    logger.success(f"批量添加了 {count} 个文档块")
                else:
                    logger.error("请指定 --novel-path 或 --novel-dir")
                    return 1
                
                kb.save()
                
            elif args.kb_action == "search":
                # 搜索知识库
                if not args.query:
                    logger.error("请指定 --query 参数")
                    return 1
                
                results = kb.search(
                    args.query,
                    top_k=args.top_k or 5,
                    style=args.style
                )
                
                print("\n搜索结果：")
                print("="*50)
                for i, result in enumerate(results, 1):
                    print(f"\n[{i}] 相似度: {result['score']:.3f}")
                    print(f"来源: {result['metadata'].get('source', 'unknown')}")
                    print(f"风格: {result['metadata'].get('style', 'unknown')}")
                    print(f"内容: {result['content'][:200]}...")
                    print("-"*30)
            
            elif args.kb_action == "stats":
                # 显示统计信息
                stats = kb.get_statistics()
                print("\n知识库统计：")
                print("="*50)
                print(json.dumps(stats, ensure_ascii=False, indent=2))
            
            elif args.kb_action == "clear":
                # 清空知识库
                if input("确定要清空知识库吗？(y/n): ").lower() == 'y':
                    kb.vector_store.index.reset()
                    kb.vector_store.documents.clear()
                    kb.vector_store.doc_map.clear()
                    kb.save()
                    logger.success("知识库已清空")
                else:
                    logger.info("操作已取消")
            
        except Exception as e:
            logger.error(f"知识库操作失败: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1
        
        return 0
    
    # ========================================
    # 评估相关命令
    # ========================================
    
    def evaluate(self, args):
        """评估模型"""
        logger.info("开始模型评估...")
        
        try:
            from train.evaluator import NovelEvaluator
            from train.data_processor import TrainingExample
            from models.model_loader import ModelLoader
            
            # 加载模型
            loader = ModelLoader(
                model_name_or_path=args.model or self.config.model.model_name_or_path,
                device=args.device,
                dtype=torch.float16 if args.fp16 else torch.float32
            )
            loader.load_model()
            
            # 创建评估器
            evaluator = NovelEvaluator(self.config)
            
            # 准备测试数据
            test_examples = []
            if args.test_file:
                with open(args.test_file, 'r', encoding='utf-8') as f:
                    test_data = json.load(f)
                for item in test_data[:args.num_samples or 100]:
                    test_examples.append(TrainingExample(**item))
            else:
                # 使用默认测试集
                logger.warning("未指定测试文件，使用示例数据")
                for i in range(min(args.num_samples or 10, 10)):
                    test_examples.append(TrainingExample(
                        instruction=f"创作一段{args.style or '仙侠'}风格的小说",
                        input="",
                        output="这是一段示例输出...",
                        style=args.style or "仙侠"
                    ))
            
            # 执行评估
            results = evaluator.evaluate_comprehensive(
                loader.model,
                loader.tokenizer,
                test_examples,
                save_report=args.save_report
            )
            
            # 显示结果
            print("\n评估结果：")
            print("="*50)
            for metric, value in results.items():
                print(f"{metric:20s}: {value:.4f}")
            
            # 保存结果
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.success(f"评估结果已保存到: {args.output}")
            
        except Exception as e:
            logger.error(f"评估失败: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1
        
        return 0
    
    # ========================================
    # Web UI 相关命令
    # ========================================
    
    def webui(self, args):
        """启动Web界面"""
        logger.info("启动Web界面...")
        
        try:
            from ui import NovelRAGApp
            
            # 创建应用
            app = NovelRAGApp()
            
            # 构建界面
            interface = app.build_interface()
            
            # 启动服务
            interface.launch(
                server_name=args.host or "127.0.0.1",
                server_port=args.port or 7860,
                share=args.share,
                inbrowser=not args.no_browser,
                debug=args.debug
            )
            
        except ImportError:
            logger.error("Gradio未安装，请运行: pip install gradio")
            return 1
        except Exception as e:
            logger.error(f"Web界面启动失败: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1
        
        return 0
    
    # ========================================
    # 交互式模式
    # ========================================
    
    def interactive(self, args):
        """交互式对话模式"""
        logger.info("进入交互式模式...")
        
        try:
            from generation.rag_generator import RAGNovelGenerator
            from rag.knowledge_base import NovelKnowledgeBase
            
            # 初始化
            kb = None
            if args.use_rag:
                kb = NovelKnowledgeBase(
                    embedding_model_name=args.embedding_model or "BAAI/bge-small-zh-v1.5",
                    vector_store_path=args.vector_store or "./data/vector_store/novels"
                )
            
            generator = RAGNovelGenerator(
                model_name=args.model or self.config.model.model_name_or_path,
                knowledge_base=kb,
                device=args.device
            )
            
            print("\n" + "="*50)
            print("Novel-RAG 交互式模式")
            print("输入 'quit' 或 'exit' 退出")
            print("输入 'clear' 清空历史")
            print("输入 'style <风格>' 切换风格")
            print("="*50 + "\n")
            
            current_style = args.style or "仙侠"
            
            while True:
                try:
                    # 获取用户输入
                    prompt = input(f"\n[{current_style}] > ").strip()
                    
                    if prompt.lower() in ['quit', 'exit']:
                        break
                    elif prompt.lower() == 'clear':
                        generator.clear_history()
                        print("历史已清空")
                        continue
                    elif prompt.lower().startswith('style '):
                        current_style = prompt[6:].strip()
                        print(f"风格已切换为: {current_style}")
                        continue
                    elif not prompt:
                        continue
                    
                    # 生成回复
                    print("\n生成中...", end='', flush=True)
                    
                    if args.stream:
                        print("\r", end='')  # 清除"生成中..."
                        for chunk in generator.generate_stream(
                            prompt=prompt,
                            style=current_style,
                            use_rag=args.use_rag,
                            use_history=True,
                            max_new_tokens=args.max_tokens or 512,
                            temperature=args.temperature or 0.7
                        ):
                            print(chunk, end='', flush=True)
                        print()  # 换行
                    else:
                        response = generator.generate(
                            prompt=prompt,
                            style=current_style,
                            use_rag=args.use_rag,
                            use_history=True,
                            max_new_tokens=args.max_tokens or 512,
                            temperature=args.temperature or 0.7
                        )
                        print("\r" + response)  # 覆盖"生成中..."
                    
                except KeyboardInterrupt:
                    print("\n\n使用 'quit' 或 'exit' 退出")
                    continue
                except Exception as e:
                    logger.error(f"生成错误: {e}")
                    continue
            
            print("\n再见！")
            
        except Exception as e:
            logger.error(f"交互模式失败: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1
        
        return 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Novel-RAG: 基于RAG的智能小说生成系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成小说
  python main.py generate --prompt "写一段主角突破境界的场景" --style 仙侠
  
  # 训练模型
  python main.py train --model Qwen/Qwen2.5-3B-Instruct --styles 仙侠 武侠
  
  # 管理知识库
  python main.py kb add --novel-dir ./data/novels/仙侠 --style 仙侠
  
  # 启动Web界面
  python main.py webui --share
  
  # 交互式模式
  python main.py interactive --use-rag --stream
        """
    )
    
    # 全局参数
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="日志级别")
    parser.add_argument("--log-file", type=str, help="日志文件路径")
    parser.add_argument("--device", type=str, help="设备 (cuda/cpu/mps)")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # ========================================
    # generate 命令
    # ========================================
    parser_gen = subparsers.add_parser("generate", help="生成小说内容")
    parser_gen.add_argument("--prompt", "-p", type=str, required=True, help="生成提示")
    parser_gen.add_argument("--style", "-s", type=str, default="仙侠", help="小说风格")
    parser_gen.add_argument("--model", "-m", type=str, help="模型名称或路径")
    parser_gen.add_argument("--use-rag", action="store_true", help="使用RAG增强")
    parser_gen.add_argument("--stream", action="store_true", help="流式输出")
    parser_gen.add_argument("--max-tokens", type=int, default=512, help="最大生成长度")
    parser_gen.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser_gen.add_argument("--top-p", type=float, default=0.9, help="Top-p采样")
    parser_gen.add_argument("--output", "-o", type=str, help="输出文件路径")
    parser_gen.add_argument("--clear-history", action="store_true", help="清空历史记录")
    parser_gen.add_argument("--embedding-model", type=str, help="嵌入模型")
    parser_gen.add_argument("--vector-store", type=str, help="向量库路径")
    
    # ========================================
    # train 命令
    # ========================================
    parser_train = subparsers.add_parser("train", help="训练模型")
    parser_train.add_argument("--config", "-c", type=str, help="配置文件路径")
    parser_train.add_argument("--model", "-m", type=str, help="基础模型")
    parser_train.add_argument("--project", type=str, help="项目名称")
    parser_train.add_argument("--data-dir", type=str, help="小说数据目录")
    parser_train.add_argument("--styles", nargs="+", help="训练风格列表")
    parser_train.add_argument("--max-samples", type=int, help="每种风格最大样本数")
    parser_train.add_argument("--sft-epochs", type=int, help="SFT训练轮数")
    parser_train.add_argument("--dpo-epochs", type=int, help="DPO训练轮数")
    parser_train.add_argument("--skip-sft", action="store_true", help="跳过SFT")
    parser_train.add_argument("--skip-dpo", action="store_true", help="跳过DPO")
    parser_train.add_argument("--skip-eval", action="store_true", help="跳过评估")
    parser_train.add_argument("--resume", action="store_true", help="从检查点恢复")
    parser_train.add_argument("--from-step", choices=["data", "sft", "dpo", "eval"], help="从指定步骤开始")
    parser_train.add_argument("--wandb", action="store_true", help="使用Weights & Biases")
    
    # ========================================
    # kb (knowledge base) 命令
    # ========================================
    parser_kb = subparsers.add_parser("kb", help="管理知识库")
    parser_kb.add_argument("kb_action", choices=["add", "search", "stats", "clear"], help="知识库操作")
    parser_kb.add_argument("--novel-path", type=str, help="单个小说文件路径")
    parser_kb.add_argument("--novel-dir", type=str, help="小说目录路径")
    parser_kb.add_argument("--style", type=str, help="小说风格")
    parser_kb.add_argument("--query", "-q", type=str, help="搜索查询")
    parser_kb.add_argument("--top-k", type=int, default=5, help="返回结果数量")
    parser_kb.add_argument("--embedding-model", type=str, help="嵌入模型")
    parser_kb.add_argument("--vector-store", type=str, help="向量库路径")
    parser_kb.add_argument("--chunk-size", type=int, help="文本块大小")
    parser_kb.add_argument("--chunk-overlap", type=int, help="文本块重叠")
    
    # ========================================
    # evaluate 命令
    # ========================================
    parser_eval = subparsers.add_parser("evaluate", help="评估模型")
    parser_eval.add_argument("--model", "-m", type=str, help="模型路径")
    parser_eval.add_argument("--test-file", type=str, help="测试数据文件")
    parser_eval.add_argument("--num-samples", type=int, help="评估样本数")
    parser_eval.add_argument("--style", type=str, help="评估风格")
    parser_eval.add_argument("--output", "-o", type=str, help="输出文件")
    parser_eval.add_argument("--save-report", action="store_true", help="保存评估报告")
    parser_eval.add_argument("--fp16", action="store_true", help="使用FP16")
    
    # ========================================
    # webui 命令
    # ========================================
    parser_webui = subparsers.add_parser("webui", help="启动Web界面")
    parser_webui.add_argument("--host", type=str, default="127.0.0.1", help="服务器地址")
    parser_webui.add_argument("--port", type=int, default=7860, help="服务器端口")
    parser_webui.add_argument("--share", action="store_true", help="创建公共链接")
    parser_webui.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    
    # ========================================
    # interactive 命令
    # ========================================
    parser_inter = subparsers.add_parser("interactive", help="交互式对话模式")
    parser_inter.add_argument("--model", "-m", type=str, help="模型名称或路径")
    parser_inter.add_argument("--style", "-s", type=str, default="仙侠", help="初始风格")
    parser_inter.add_argument("--use-rag", action="store_true", help="使用RAG增强")
    parser_inter.add_argument("--stream", action="store_true", help="流式输出")
    parser_inter.add_argument("--max-tokens", type=int, help="最大生成长度")
    parser_inter.add_argument("--temperature", type=float, help="生成温度")
    parser_inter.add_argument("--embedding-model", type=str, help="嵌入模型")
    parser_inter.add_argument("--vector-store", type=str, help="向量库路径")
    
    # 解析参数
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level, args.log_file)
    
    # 显示欢迎信息
    if not args.command:
        print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     Novel-RAG: 基于RAG的智能小说生成系统                      ║
║                                                              ║
║     版本: 2.0.0                                              ║
║     作者: Novel-RAG Team                                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """)
        parser.print_help()
        return 0
    
    # 创建CLI实例
    cli = NovelRAGCLI()
    
    # 执行命令
    if args.command == "generate":
        return cli.generate(args)
    elif args.command == "train":
        return cli.train(args)
    elif args.command == "kb":
        return cli.knowledge_base(args)
    elif args.command == "evaluate":
        return cli.evaluate(args)
    elif args.command == "webui":
        return cli.webui(args)
    elif args.command == "interactive":
        return cli.interactive(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())