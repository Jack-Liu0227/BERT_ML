"""
下载BERT模型脚本
用于下载steelbert, scibert, matscibert模型

使用说明:
1. SteelBERT是gated repo，需要先登录HuggingFace并获得访问权限
   运行: huggingface-cli login
   然后在 https://huggingface.co/MGE-LLMs/SteelBERT 申请访问权限

2. 如果遇到torch版本问题，脚本会自动使用safetensors格式
"""

import os
from transformers import AutoTokenizer, AutoModel

# 模型配置
MODELS = {
    'steelbert': 'MGE-LLMs/SteelBERT',
    'scibert': 'allenai/scibert_scivocab_uncased',
    'matscibert': 'm3rg-iitd/matscibert'
}

# 基础目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def download_model(model_name, model_id, use_auth_token=None):
    """下载单个模型"""
    save_path = os.path.join(BASE_DIR, model_name)
    
    print(f"开始下载 {model_name} from {model_id}")
    print(f"保存路径: {save_path}")
    
    try:
        # 下载tokenizer
        print(f"下载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=use_auth_token,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(save_path)
        
        # 下载模型 - 优先使用safetensors
        print(f"下载 model...")
        try:
            # 首选safetensors格式（避免torch版本问题）
            model = AutoModel.from_pretrained(
                model_id,
                token=use_auth_token,
                trust_remote_code=True,
                use_safetensors=True
            )
        except Exception as e:
            print(f"  尝试使用safetensors失败，回退到pytorch格式...")
            model = AutoModel.from_pretrained(
                model_id,
                token=use_auth_token,
                trust_remote_code=True
            )
        
        model.save_pretrained(save_path, safe_serialization=True)
        
        print(f"✓ {model_name} 下载完成\n")
        return True
    except Exception as e:
        print(f"✗ {model_name} 下载失败: {str(e)}\n")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("BERT模型下载脚本")
    print("=" * 60)
    print()
    
    # 检查是否需要登录
    print("提示: 如果需要访问gated repos (如SteelBERT)，请先运行:")
    print("  huggingface-cli login")
    print("  然后访问模型页面申请权限\n")
    
    # 尝试获取token（如果已登录）
    token = None
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print("✓ 检测到HuggingFace token\n")
    except:
        print("⚠ 未检测到HuggingFace token（部分模型可能需要）\n")
    
    results = {}
    for model_name, model_id in MODELS.items():
        results[model_name] = download_model(model_name, model_id, use_auth_token=token)
    
    # 显示下载结果
    print("=" * 60)
    print("下载结果汇总:")
    print("=" * 60)
    for model_name, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{model_name}: {status}")
    
    # 失败提示
    if not all(results.values()):
        print("\n" + "=" * 60)
        print("失败原因可能:")
        print("=" * 60)
        print("1. SteelBERT: 需要登录并申请访问权限")
        print("   - 运行: huggingface-cli login")
        print("   - 访问: https://huggingface.co/MGE-LLMs/SteelBERT")
        print("2. 网络连接问题")
        print("3. 磁盘空间不足")

if __name__ == "__main__":
    main()
