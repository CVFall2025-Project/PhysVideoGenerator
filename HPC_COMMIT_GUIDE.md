# HPC Commit and Push Guide

由于无法直接通过SSH连接到HPC，请按照以下步骤操作：

## 步骤1: 同步文件到HPC

在本地机器上运行：

```bash
cd /Users/hongyuchen/Documents/CV/Project
./sync_to_hpc.sh
```

这将同步以下文件到HPC：
- `evaluation/` 目录及其所有文件
- `evaluation_utils.py`
- `requirements_comparison.txt`
- `setup_comparison_experiments.sh`
- `commit_comparison_experiments.sh`
- `COMPARISON_EXPERIMENTS_SETUP.md`
- `COMMIT_INSTRUCTIONS.md`
- `commit_and_push_on_hpc.sh` (新脚本)

## 步骤2: 连接到HPC

使用以下命令连接到HPC：

```bash
cd /Users/hongyuchen/Documents/CV/Project
./connect_hpc.sh burst
```

如果连接失败，可能需要：
1. 使用VPN连接到NYU网络
2. 或者使用其他网络连接方式

## 步骤3: 在HPC上运行commit脚本

连接到HPC后，运行以下命令：

```bash
# 导航到仓库目录
cd /scratch/hc4569/repos/PhysVideoGenerator

# 运行commit和push脚本
bash /scratch/hc4569/commit_and_push_on_hpc.sh
```

或者，如果脚本已经同步到仓库目录：

```bash
cd /scratch/hc4569/repos/PhysVideoGenerator
bash commit_and_push_on_hpc.sh
```

## 步骤4: 验证

提交和推送完成后，验证更改：

```bash
# 查看最新commit
git log -1

# 查看远程分支
git ls-remote --heads origin comparison-experiments
```

或者在GitHub上查看：
https://github.com/CVFall2025-Project/PhysVideoGenerator/tree/comparison-experiments

## 手动操作（如果脚本失败）

如果脚本失败，可以手动执行以下步骤：

```bash
cd /scratch/hc4569/repos/PhysVideoGenerator

# 切换到comparison-experiments分支
git checkout comparison-experiments 2>/dev/null || git checkout -b comparison-experiments

# 添加文件
git add evaluation/ evaluation_utils.py requirements_comparison.txt
git add setup_comparison_experiments.sh commit_comparison_experiments.sh
git add COMPARISON_EXPERIMENTS_SETUP.md COMMIT_INSTRUCTIONS.md

# 检查状态
git status

# Commit
git commit -m "Add evaluation scripts for comparison experiments (OpenSora, VideoCrafter2, HunyuanVideo)

- Add evaluation_utils.py with common metrics (PSNR, SSIM, FVD, CLIP score)
- Add opensora_eval.py for OpenSora model evaluation (using official API)
- Add videocrafter2_eval.py for VideoCrafter2 model evaluation (using official API patterns)
- Add hunyuanvideo_eval.py for HunyuanVideo model evaluation (using official API patterns)
- Add run_comparison.py for unified comparison runner
- Add requirements_comparison.txt with dependencies
- Add setup and commit scripts for HPC
- Add evaluation/README.md and API_UPDATE_NOTES.md with usage documentation
- Update all scripts to use official APIs instead of placeholders"

# Push
git push -u origin comparison-experiments
```

## 故障排除

### 问题1: 文件未同步
**解决方案**: 确保运行了 `./sync_to_hpc.sh`，检查文件是否在 `/scratch/hc4569/` 目录下

### 问题2: Push失败（SSH密钥问题）
**解决方案**: 
1. 在HPC上设置SSH密钥：`bash setup_github_ssh_on_hpc.sh`
2. 或使用HTTPS方式push（需要Personal Access Token）

### 问题3: 分支不存在
**解决方案**: 脚本会自动创建分支，如果失败，手动创建：
```bash
git checkout -b comparison-experiments
```

### 问题4: 无法连接到HPC
**解决方案**: 
1. 检查网络连接
2. 确保使用VPN（如果需要）
3. 检查SSH配置：`cat ~/.ssh/config | grep greene`


