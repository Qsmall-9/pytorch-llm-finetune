import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq  # 用于对话模型
)
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model

print("=" * 50)
print("🚀 Qwen2-0.5B-Instruct 微调 (稳定版)")
print("=" * 50)

# ========== 1. 检查GPU ==========
print(f"\n📟 环境检查:")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ========== 2. 准备数据 ==========
print(f"\n📚 准备数据集.")

train_data = [
    {"instruction": "什么是机器学习？", "output": "机器学习是人工智能的一个分支，它让计算机通过数据学习规律，而无需明确编程。"},
    {"instruction": "解释深度学习", "output": "深度学习是机器学习的子集，使用多层神经网络模拟人脑的工作方式。"},
    {"instruction": "什么是神经网络？", "output": "神经网络是一种受生物神经元启发的计算模型，由多层神经元组成。"},
    {"instruction": "Python是什么？", "output": "Python是一种高级编程语言，以简洁易读的语法著称，广泛应用于数据分析、人工智能等领域。"},
    {"instruction": "如何学习编程？", "output": "学习编程的建议：1)选择入门语言如Python；2)通过实际项目练习；3)阅读优秀代码；4)保持持续学习。"},
]

# 格式化为 Qwen2 的对话格式，并 tokenize
def format_and_tokenize(example):
    messages = [
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]}
    ]
    
    # 使用 chat template 生成文本
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False  # 训练时不需要生成提示
    )
    
    # Tokenize
    result = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors=None
    )
    
    # 对于因果LM，labels = input_ids（预测下一个token）
    result["labels"] = result["input_ids"].copy()
    return result

# ========== 3. 加载模型和分词器 ==========
model_name = "Qwen/Qwen2-0.5B-Instruct"
print(f"\n🤖 加载模型: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 先处理数据（需要 tokenizer）
print(f"\n📝 处理数据...")
dataset = Dataset.from_list(train_data)
tokenized_dataset = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)
print(f"   处理完成，共 {len(tokenized_dataset)} 条")

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print(f"   模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# ========== 4. 配置LoRA ==========
print(f"\n🔧 配置LoRA...")

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ========== 5. 训练配置 ==========
print(f"\n⚙️  配置训练...")

training_args = TrainingArguments(
    output_dir="./qwen2_lora_results",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=5e-4,
    warmup_steps=5,
    logging_steps=5,
    save_steps=20,
    save_total_limit=2,
    fp16=True, 
    optim="adamw_torch",
    report_to="none",
    remove_unused_columns=False,  # 重要：保留我们的字段
)

# 数据整理器
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    return_tensors="pt"
)

# ========== 6. 开始训练（用基础 Trainer）==========
print(f"\n🏃 开始训练...")
print("=" * 50)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,  # 处理 batch 数据
)

trainer.train()

# ========== 7. 保存模型 ==========
print(f"\n💾 保存模型...")
trainer.model.save_pretrained("./qwen2_lora_model")
tokenizer.save_pretrained("./qwen2_lora_model")
print(f"   已保存到: ./qwen2_lora_model")

print(f"\n✅ 训练完成！")