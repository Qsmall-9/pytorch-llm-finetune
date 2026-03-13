import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("=" * 60)
print("🤖 Qwen2-0.5B 微调效果对比")
print("=" * 60)

model_name = "Qwen/Qwen2-0.5B-Instruct"
lora_path = "./qwen2_lora_model"

# 检查是否有训练好的模型
import os
has_finetuned = os.path.exists(lora_path)
print(f"\n📂 检查模型: {'找到微调模型 ✓' if has_finetuned else '未找到微调模型 ✗'}")

# ========== 加载基础模型（微调前）==========
print(f"\n🔧 加载基础模型: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# ========== 加载微调模型（微调后）==========
if has_finetuned:
    print(f"🔧 加载微调模型: {lora_path}")
    finetuned_model = PeftModel.from_pretrained(base_model, lora_path)
    finetuned_model = finetuned_model.merge_and_unload()  # 合并权重，加速推理
else:
    finetuned_model = None
    print("⚠️  未找到微调模型，只测试基础模型")

# ========== 对话函数 ==========
def chat(model, user_input, model_name="模型"):
    messages = [
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": user_input}
    ]
    
    # 生成 prompt
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,    # 最多生成200个token
            temperature=0.7,       # 随机性
            top_p=0.9,            # 核采样
            do_sample=True,       # 采样而非贪婪
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取 assistant 的回答
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    return response

# ========== 交互式对话 ==========
print("\n" + "=" * 60)
print("💬 开始对话 (输入 'quit' 退出, 输入 'compare' 对比模式)")
print("=" * 60)

# 预设测试问题（训练数据里的 + 新的）
test_questions = [
    "什么是机器学习？",
    "Python是什么？",
    "如何学习编程？",
    "什么是深度学习？",  # 训练数据里没有的
]

while True:
    print(f"\n{'='*60}")
    print("📝 输入选项：")
    print("   1. 直接输入你的问题")
    print("   2. 输入数字 1-4 选择预设问题")
    print("   3. 输入 'compare' 对比基础模型 vs 微调模型")
    print("   4. 输入 'quit' 退出")
    print(f"{'='*60}")
    
    user_input = input("\n👤 你: ").strip()
    
    if user_input.lower() == 'quit':
        print("👋 再见！")
        break
    
    # 选择预设问题
    if user_input in ['1', '2', '3', '4']:
        idx = int(user_input) - 1
        user_input = test_questions[idx]
        print(f"   (选择问题: {user_input})")
    
    # 对比模式
    if user_input.lower() == 'compare':
        print(f"\n🔍 对比模式：基础模型 vs 微调模型")
        print("-" * 60)
        
        question = input("👤 输入要对比的问题: ")
        
        print(f"\n{'='*60}")
        print(f"📌 问题: {question}")
        print(f"{'='*60}")
        
        # 基础模型回答
        print(f"\n🤖 【基础模型】回答:")
        print("-" * 60)
        base_response = chat(base_model, question, "基础模型")
        print(base_response)
        
        # 微调模型回答
        if finetuned_model:
            print(f"\n🤖 【微调模型】回答:")
            print("-" * 60)
            fine_response = chat(finetuned_model, question, "微调模型")
            print(fine_response)
            
            print(f"\n{'='*60}")
            print("📊 对比总结:")
            print(f"   基础模型长度: {len(base_response)} 字")
            print(f"   微调模型长度: {len(fine_response)} 字")
            if len(fine_response) > len(base_response):
                print("   ✓ 微调模型回答更详细")
        else:
            print("\n⚠️  没有微调模型，无法对比")
        
        continue
    
    # 单模型对话模式（默认用微调模型，没有就用基础模型）
    active_model = finetuned_model if finetuned_model else base_model
    model_label = "微调模型" if finetuned_model else "基础模型"
    
    print(f"\n🤖 【{model_label}】回答:")
    print("-" * 60)
    response = chat(active_model, user_input, model_label)
    print(response)

print(f"\n{'='*60}")
print("✅ 演示结束")
print(f"{'='*60}")