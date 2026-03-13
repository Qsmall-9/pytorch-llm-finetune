# 🤖 Qwen2 LoRA 微调学习项目

基于 Qwen2-0.5B-Instruct 的 LoRA 微调实践，适合个人 GPU 环境学习大模型微调技术。

## 🎯 项目特点

- **💡 轻量级**: 使用 LoRA 技术，只需 4-8GB 显存
- **📚 易理解**: 详细注释，适合学习 Transformer 和微调原理  
- **🔄 可复现**: 完整的训练和推理流程
- **💬 实用性**: 真实的对话数据微调案例

## 📋 环境要求

### 硬件要求
- **GPU**: 4GB+ 显存（推荐 8GB+）
- **内存**: 8GB+ RAM
- **存储**: 5GB+ 可用空间

### 软件要求
- Python 3.8+
- CUDA 11.8+ (如果使用 GPU)
- PyTorch 2.0+

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/your-username/qwen2-lora-finetuning.git
cd qwen2-lora-finetuning
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 开始训练
```bash
python Train.py
```

训练过程中你会看到：
- 🔧 环境检查和GPU状态
- 📚 数据加载和预处理
- 🤖 模型加载和LoRA配置
- 🏃 训练进度和损失变化
- 💾 模型保存

### 4. 测试效果
```bash
python Inference.py
```

支持三种模式：
- **直接对话**: 输入问题，获得回答
- **对比模式**: 输入 `compare` 对比基础模型 vs 微调模型
- **预设问题**: 输入数字 1-4 选择预设问题测试

## 📊 项目结构

```
qwen2-lora-finetuning/
├── Train.py                    # 训练脚本
├── Inference.py                # 推理测试脚本
├── requirements.txt            # 依赖包列表
├── README.md                   # 项目说明
├── .gitignore                  # Git忽略文件
│
├── data/                       # 数据目录
│   └── sample_data.json        # 示例训练数据
│
├── configs/                    # 配置文件
│   ├── lora_config.json        # LoRA配置
│   └── training_config.yaml    # 训练配置
│
└── qwen2_lora_model/          # 训练后的模型（训练完成后生成）
    ├── adapter_config.json     # 适配器配置
    ├── tokenizer_config.json   # 分词器配置
    └── chat_template.jinja     # 对话模板
```

## 🔧 核心技术

### LoRA (Low-Rank Adaptation)
- **原理**: 只训练少量参数（1-2%），大幅降低显存需求
- **优势**: 训练快速、显存友好、效果显著
- **参数**: r=8, alpha=32, dropout=0.05

### Transformer 架构
- **模型**: Qwen2-0.5B (5亿参数)
- **任务**: 因果语言建模（预测下一个词）
- **训练**: 自回归生成，使用因果掩码

## 📈 训练配置

### 默认参数
```yaml
训练轮数: 10 epochs
批大小: 4 (梯度累积 x2 = 有效批大小 8)
学习率: 5e-4
最大长度: 512 tokens
优化器: AdamW
精度: FP16 (半精度)
```

### LoRA 配置
```json
{
  "r": 8,                    // LoRA秩，控制参数量
  "lora_alpha": 32,          // 缩放因子
  "lora_dropout": 0.05,      // Dropout率
  "target_modules": [        // 微调的模块
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
  ]
}
```

## 🎓 学习要点

### 1. 数据处理
```python
# 对话格式化
messages = [
    {"role": "system", "content": "你是一个有帮助的AI助手。"},
    {"role": "user", "content": "什么是机器学习？"},
    {"role": "assistant", "content": "机器学习是..."}
]

# 使用chat template转换
text = tokenizer.apply_chat_template(messages, tokenize=False)
```

### 2. LoRA 微调
```python
# 只训练LoRA参数，冻结原模型
lora_config = LoraConfig(r=8, lora_alpha=32, ...)
model = get_peft_model(base_model, lora_config)

# 查看可训练参数
model.print_trainable_parameters()
# 输出: trainable params: 1,048,576 || all params: 494,033,920 || trainable%: 0.21%
```

### 3. 生成策略
```python
# 控制生成质量的关键参数
outputs = model.generate(
    max_new_tokens=200,    # 最大生成长度
    temperature=0.7,       # 随机性 (0=确定, 1=随机)
    top_p=0.9,            # 核采样阈值
    do_sample=True        # 启用采样
)
```

## 🔬 实验结果

### 训练效果
- **训练时间**: ~10分钟 (RTX 3080)
- **显存占用**: ~6GB
- **参数效率**: 只训练 0.21% 的参数
- **最终Loss**: ~0.9

### 对话效果示例
| 问题 | 基础模型 | 微调模型 |
|------|----------|----------|
| 什么是机器学习？ | 简短通用回答 | 详细专业解释 |
| Python是什么？ | 基础介绍 | 针对AI领域的回答 |

## 🛠️ 自定义训练

### 1. 准备数据
编辑 `data/sample_data.json`:
```json
[
  {
    "instruction": "你的问题",
    "output": "期望的回答"
  }
]
```

### 2. 调整配置
修改 `configs/training_config.yaml` 中的参数：
- 增加 `num_train_epochs` 提升效果
- 调整 `learning_rate` 控制学习速度
- 修改 `per_device_train_batch_size` 适配显存

### 3. 运行训练
```bash
python Train.py
```

## 🤔 常见问题

### Q: 显存不够怎么办？
**A**: 
- 减小 `per_device_train_batch_size` 到 2 或 1
- 增加 `gradient_accumulation_steps` 保持有效批大小
- 使用更小的模型如 Qwen2-0.5B

### Q: 训练效果不好？
**A**: 
- 增加训练数据量（推荐 100+ 条）
- 调整学习率（尝试 1e-4 或 1e-3）
- 增加训练轮数到 20-50
- 检查数据质量和格式

### Q: 如何评估模型效果？
**A**: 
- 使用 `Inference.py` 的对比模式
- 准备测试集进行定量评估
- 观察训练过程中的 loss 变化曲线

### Q: 可以用其他模型吗？
**A**: 
- 支持所有 HuggingFace 的对话模型
- 修改 `Train.py` 中的 `model_name`
- 注意调整 `target_modules` 适配不同架构

## 📚 学习资源

### 理论基础
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原论文
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) - LoRA技术论文
- [Qwen2 技术报告](https://arxiv.org/abs/2407.10671) - Qwen2模型详解

### 实践教程
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) - 官方文档
- [PEFT 库使用指南](https://huggingface.co/docs/peft) - 参数高效微调
- [PyTorch 深度学习](https://pytorch.org/tutorials/) - PyTorch教程

### 视频推荐
- [3Blue1Brown - Transformer可视化](https://www.youtube.com/watch?v=wjZofJX0v4M)
- [Andrej Karpathy - GPT从零实现](https://www.youtube.com/watch?v=kCc8FmEb1nY)

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 如何贡献
1. Fork 本项目
2. 创建特性分支: `git checkout -b feature/amazing-feature`
3. 提交更改: `git commit -m 'Add amazing feature'`
4. 推送分支: `git push origin feature/amazing-feature`
5. 提交 Pull Request

### 贡献方向
- 添加更多示例数据集
- 支持更多模型架构
- 改进训练脚本和配置
- 完善文档和教程

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- **Qwen Team**: 提供优秀的 Qwen2 基础模型
- **HuggingFace**: 提供便捷的 Transformers 和 PEFT 库
- **Microsoft**: 提出 LoRA 参数高效微调技术
- **PyTorch Team**: 提供强大的深度学习框架

---

⭐ **如果这个项目对你有帮助，请给个 Star！**

🔥 **欢迎 Fork 并分享你的微调实验结果！**