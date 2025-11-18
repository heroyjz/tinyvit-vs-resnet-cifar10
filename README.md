## ViT vs ResNet on CIFAR-10

这个仓库复现了一个面向教学/实验的图像分类对比：在 CIFAR-10 上分别微调与从零训练 TinyViT-5M 与 ResNet-18，观察迁移学习对收敛速度与最终精度的影响。

### 实验内容

1. **CNN 基线（ResNet-18）**
   - 目标：以经典卷积网络作为参照，衡量在小型数据集上“迁移学习 vs. 从零训练”的效果差异。
   - 方法：分别加载 ImageNet-1k 预训练权重和随机初始化权重，在 CIFAR-10 上以相同的数据增强、优化器和训练计划进行 30 轮训练。预训练版本作为迁移学习基线，随机初始化版本用于衡量纯 CIFAR 训练的上限。
   - 参数选择：输入分辨率统一为 224×224，以便与 TinyViT 保持一致；优化器选用 AdamW + Cosine LR（初始学习率 3e-4、weight decay 0.05），结合 RandAugment 与 RandomResizeCrop，使 CNN 与 Transformer 在相同的正则化条件下比较；batch size 128 足以填满 RTX 4070 显存并确保统计稳定。
   - 指标：关注训练/验证 loss、accuracy 曲线和最终测试精度，用于评估卷积架构在 CIFAR-10 上的收敛速度与泛化能力。

2. **Transformer 对照（TinyViT-5M）**
   - 目标：在与 ResNet 完全一致的训练设置下，比较轻量级 Vision Transformer 在预训练与从零训练两种模式下的表现，检验“预训练知识迁移”对 Transformer 的影响。
   - 方法：选择 TinyViT-5M（先在 ImageNet-22k 上训练，再在 1k 上微调的权重），保持输入尺寸、batch size、优化器、数据增强等参数不变，分别进行微调与随机初始化训练。由于 Transformer 对正则化更敏感，可选地加入 Mixup/CutMix 观察其对收敛的影响。
   - 参数选择：保持 30 个 epoch 的训练预算，记录 TinyViT 微调在前 6–10 个 epoch 的快速收敛，同时允许在需要时提前终止；学习率调度与 ResNet 相同，确保比较指标一致。
   - 指标：和 ResNet 一样关注 train/val loss、accuracy，并额外分析 TinyViT 在相同训练预算下的吞吐量与算力开销，展示 Transformer 在小数据集上的效率差异与精度优势。

3. **可视化与案例分析**
   - 为每次运行输出 `metrics.csv`、`config.json`、`best.pt`、`last.pt`，确保实验可复现。
   - 通过 `compare_runs.py`、`plot_four_panels.py` 生成 Train/Val Loss 与 Accuracy 的四象限对比图，突出四种设置（ResNet/TinyViT × 预训练/从零）的差异，并在曲线末尾标注最终数值。
   - 利用 `show_resnet_example.py` 随机抽取 CIFAR-10 图像展示实际预测结果，配合图表形成完整的定量+定性分析。

通过以上设置，可以在相同的硬件预算与训练超参数下比较 Transformer 与 CNN 的迁移学习效果，清晰说明预训练对 TinyViT 的收益，并用数据/图表支持“在小型数据集上是否值得使用 ViT”这一问题。

### 环境要求

硬件：

- CPU：Intel(R) Core(TM) i7-14700HX（WSL2 上可见 8 线程）
- GPU：NVIDIA GeForce RTX 4070 Laptop，驱动 581.42，CUDA 13.0
- 内存：≥16 GB（本地实验在 32 GB RAM 上完成）

系统：

- Windows 11 + WSL2，内核 `Linux 6.6.87.2-microsoft-standard-WSL2`
- glibc 2.39，`bash` 终端
- Conda 虚拟环境 `vit`

软件：

- Python 3.11.7
- PyTorch 2.9.0+cu130（CUDA runtime 13.0，cuDNN 91300），TorchVision 0.24.0+cu130
- `timm`、`matplotlib`、`pandas`、`tensorboard`（可选，用于日志/可视化）

> 参见 `env_setup.md` 获取完整的 Conda 创建与依赖安装脚本，推荐 `conda create -n vit python=3.10 && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` 等命令。

### 代码结构

```
.
├── train_resnet_cifar.py      # ResNet-18 训练脚本（支持预训练/从零）
├── train_tinyvit_cifar.py     # TinyViT-5M 训练脚本（支持预训练/从零、
├── train_show_resnet.py       # 单次实验的 loss/acc 可视化
├── compare_runs.py            # 多个 metrics.csv 的对比绘图
├── show_resnet_example.py     # 载入 checkpoint，展示预测示例
├── env_setup.md               # 环境搭建与依赖安装指南
├── design.md                  # 原始实验设计与分阶段计划
└── runs/                      # 默认日志与模型输出目录
```

### 使用方法

1. **准备环境**

   ```bash
   conda activate vit          # 进入已安装依赖的环境
   python check.py             # 确认 torch/cuda/cudnn 状态
   ```

2. **运行四组合实验**

   ```bash
   # ResNet-18 预训练微调
   python train_resnet_cifar.py --pretrained --output-dir runs/resnet_ft

   # ResNet-18 从零训练
   python train_resnet_cifar.py --no-pretrained --output-dir runs/resnet_scratch

   # TinyViT-5M 预训练微调
   python train_tinyvit_cifar.py --pretrained --output-dir runs/tinyvit_ft

   # TinyViT-5M 从零训练
   python train_tinyvit_cifar.py --no-pretrained --output-dir runs/tinyvit_scratch
   ```

   可通过 `--epochs`、`--batch-size`、`--optimizer`、`--mixup/--cutmix` 等参数自定义训练策略；所有运行都会在 `runs/...` 下生成 `config.json`、`metrics.csv`、`best.pt`、`last.pt`。

3. **可视化训练曲线**

   - 单次运行：
     ```bash
     python train_show_resnet.py --log-path runs/resnet_ft/metrics.csv --safe
     ```
     脚本会保存 `resnet_ft_curve.png` 并可选弹窗显示图像。

   - 对比多个实验：
     ```bash
     python compare_runs.py runs/resnet_ft/metrics.csv runs/tinyvit_ft/metrics.csv \
         --labels ResNet TinyViT --output runs/resnet_vs_tinyvit.png --safe
     ```

4. **展示预测样例**

   ```bash
   python show_resnet_example.py \
       --config runs/tinyvit_ft/config.json \
       --checkpoint runs/tinyvit_ft/best.pt \
       --index -1 --seed 42 --topk 3
   ```

   该脚本会加载指定模型，从 CIFAR-10 抽取一张图片并展示 Ground Truth 与 Top-K 预测，图像文件会保存到对应 `runs/...` 目录。

