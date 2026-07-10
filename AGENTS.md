# Codex 项目协作规则

本文件适用于整个仓库。除非用户在当前任务中明确提出不同要求，否则 Codex 必须遵守以下规则。

## 沟通

- 默认使用中文回答。
- 先给简短结论，再分点说明依据、改动、检查结果和未验证事项。
- 不得把没有实际执行的测试描述为“已通过”。

## 统一 Git 工作流程

- `master` 只保存已经在 5090 完整验证通过的稳定版本。
- `codex` 是 Mac 和 5090 上所有日常开发、文档修改和提交的唯一工作分支。
- 不论在哪台电脑，每一轮开发都遵循同一流程：从最新 `master` 同步 `codex`，在 `codex` 修改、验证和推送；只有 5090 验证成功后才将 `codex` 合入 `master`。
- Codex 不得直接提交、推送或合并到 `master`，除非用户明确授权；不得删除或强制更新任何分支。

### 每次开始开发前（Mac 和 5090 完全相同）

开始前必须先执行 `git status`，确认没有未提交修改。若工作区不干净，不得直接切换、拉取或合并分支，应先提交、暂存或向用户确认。

```bash
git fetch origin

git switch master
git pull --ff-only origin master

git switch codex
git pull --ff-only origin codex
git merge --ff-only master
```

该步骤的目的：获取其他电脑已发布的稳定版本，并让 `codex` 从最新 `master` 开始。刚同步完成且尚未开发时，可用下列命令检查两者是否对齐：

```bash
git rev-list --left-right --count master...codex
```

理想输出为 `0 0`。

### 开发、提交与推送（Mac 和 5090 完全相同）

所有修改都在 `codex` 进行；只暂存本次任务涉及的文件：

```bash
# 修改并完成可用的检查
git status
git add <具体文件>
git commit -m "<修改说明>"
git push origin codex
```

### 5090 完整验证后的稳定发布

只有在 5090 已完成真实视频、CUDA、YOLO/RTMPose 等完整验证并确认成功后，才由用户或经用户明确授权的操作将 `codex` 合入 `master`：

```bash
git switch master
git pull --ff-only origin master
git merge --no-ff codex -m "merge: verified codex changes"
git push origin master
```

发布后，为下一轮开发再次对齐 `codex`：

```bash
git switch codex
git merge --ff-only master
git push origin codex
```

### 异常处理

- 若 `git merge --ff-only master` 或 `git pull --ff-only origin codex` 报错，说明分支出现分叉或当前基线不完整；不得执行 `rebase`、强制推送、`git reset --hard` 或覆盖历史，应先检查 `git status` 和 `git log --oneline --graph --decorate --all -15`，再请求处理方向。
- 若已经在旧 `codex` 上产生未提交修改，先不要拉取或合并；应先提交、暂存或确认如何处理。
- 若已经在旧 `codex` 上提交，新旧分支的代码不会被普通 `merge` 自动删除；但快进同步失败时仍必须先检查历史，不能强行覆盖。

## 5090 运行环境

- 项目的完整运行与最终验证在用户的 NVIDIA RTX 5090 电脑上进行。
- 当前 Mac/Codex 环境可能缺少 CUDA、模型权重、YOLO/RTMPose 的部分配置、数据集和机器专属依赖。
- 不得因为当前环境缺少这些文件，就擅自删除相关引用、降低功能、替换模型或写入仅适用于 Mac 的路径。
- 除非用户明确要求，不创建假的权重、配置、输入数据或输出结果来模拟成功。
- 优先使用相对路径、配置项或环境变量，避免写死 Mac 或 5090 的绝对路径。
- 不提交模型权重、输入视频、运行输出、`.env`、密钥、令牌或机器专属文件。

## 检查与交付

- 在当前环境能安全执行的情况下，进行语法检查、静态检查和不依赖 GPU/模型权重的测试。
- 对依赖 5090、CUDA、YOLO、RTMPose、模型权重或真实视频的流程，明确标记为“未在当前环境运行，需在 5090 验证”。
- 用户负责在 5090 上拉取 `codex`、完成实际运行验证，并在成功后将 `codex` 合并到 `master`。
- 下一次任务开始时，Codex 应重新拉取最新 `master`，再同步并继续使用 `codex`。
