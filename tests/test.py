from huggingface_hub import hf_hub_download

# 下载单个GGUF文件
file_path = hf_hub_download(
    repo_id="marcelone/Qwen3-4B-Instruct-2507-gguf",  # 替换为目标模型仓库
    filename="Qwen3-4B-Instruct-2507-gguf-IQ4_NL_GXL.gguf",  # 替换为具体的GGUF文件名
    local_dir="./models"  # 本地保存目录
)