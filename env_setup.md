## 安装与环境准备指南

以下步骤带你在 WSL/Ubuntu 中创建一个独立的 Conda 环境，并安装 ViT/ResNet 实验所需的所有依赖。这样可以避免污染 base 环境，同时保证 PyTorch 与 CUDA 版本可控。

### 1. 创建 Conda 环境

```bash
conda create -n vit python=3.10
conda activate vit
```

说明：

- `vit` 是自定义的环境名称，可以按需更改。
- 新环境不会继承 base 里安装的 PyTorch，需要在激活后重新安装依赖。
- 如果你已经在 base 里装好了所有依赖，也可以直接克隆：

  ```bash
  conda deactivate
  conda create -n vit --clone base
  conda activate vit
  ```

  这样会复制 base 的全部包（包括 torch / timm 等），不用重新下载；如需额外升级，再在 vit 环境内安装即可。

### 2. 安装基础依赖

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm matplotlib tensorboard
```

说明：

- 这里示例使用 PyTorch 官方的 CUDA 12.1 轮子（根据你的 CUDA 版本调整 `--index-url`）。
- `timm` 用于加载 TinyViT/ResNet 的模型定义。
- `matplotlib`、`tensorboard` 用于日志可视化。

### 3. 验证 PyTorch/CUDA

在项目根目录运行自带脚本：

```bash
python check.py
```

确保输出中 `cuda available: True`，表示该 env 可以调用 GPU。

### 4. 常用命令速查

```bash
conda deactivate            # 退出当前环境（删除前务必执行）
conda env list              # 查看所有环境
conda remove -n vit --all   # 删除 vit 环境
conda create -n vit python=3.10  # 重新创建环境
conda activate vit          # 进入新环境
```

后续所有训练脚本（例如 `train_resnet_cifar.py`、TinyViT 系列）都在激活后的 `vit` 环境中运行即可。需要新增依赖时，也请在该 env 里安装，保持版本一致性。***
