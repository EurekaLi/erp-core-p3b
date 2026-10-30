# GitHub展示材料、核验与提交命令

## 1. 本次整理范围

仓库根目录：`D:\AILearning\project_for_offer\erp-core-p3b`。
origin已配置为`https://github.com/EurekaLi/erp-core-p3b.git`，当前分支为main。
不用再次git init或添加同名origin。

本次整理了首页README、`reports/release`聚合结果包、3张分类核心图，
并按文件名调整`.gitignore`的图表允许列表。
原有详细方法文档、01–14入口、源码、配置与测试纳入待提交材料。
2026-10-30补充15发布核验、REPRODUCING、NOTICE和换行规则。
个人面试题库docs/17保留在本地并忽略，不随研究项目发布。
Git操作由使用者执行，本次没有git add、commit或push。
面试演示由使用者自行制作，本次没有创建演示文件。

## 2. 哪些内容会进入版本控制

|内容|处理方式|
|---|---|
|README、方法/结果文档|纳入|
|项目源码、入口、测试和配置|纳入|
|环境骨架及实测环境版本记录|纳入；版本记录不等于完整环境锁|
|既有经典ERP核心图/统计摘要|保留已有跟踪|
|新公开聚合表及3张分类图|纳入|
|EEG原始数据、FIF、缓存数组|忽略|
|作者独立仓库和预训练权重|忽略，保留来源commit/revision/hash|
|完整逐trial预测及39份本地报告|默认忽略，由14重建|

发布材料以`reports/release`为公开入口，避免直接开放整个results目录。
历史文档含本机操作记录和本地生成文件路径；不表示本地大文件已随Git发布。

## 3. 查看本地状态

```powershell
Set-Location 'D:\AILearning\project_for_offer\erp-core-p3b'
git status --short
git branch --show-current
git remote -v
git diff --stat
git diff --check
```

`git diff`只展示已跟踪文件的变化；新文件同时看`git status`。
在具有全部测试依赖的DL环境中核验源码：

```powershell
python -m ruff check src scripts tests
python -m pytest -q
python -X utf8 scripts/15_check_release.py
```

本轮补充只读发布检查及其测试，没有修改训练或统计程序。
已有训练/报告的产物hash保持不变。

## 4. 预览将暂存的文件

```powershell
git add --dry-run -- .gitignore .gitattributes README.md REPRODUCING.md NOTICE.md environment.yml pyproject.toml configs docs scripts src tests reports
```

此命令只列出将纳入的文件，忽略规则仍生效。
没有使用`git add -f`，不强行纳入data、results、third_party或models目录。

## 5. 使用者暂存并核验

```powershell
git add -- .gitignore .gitattributes README.md REPRODUCING.md NOTICE.md environment.yml pyproject.toml configs docs scripts src tests reports
git diff --cached --stat
git diff --cached --check
git diff --cached --name-only
```

核对列表中没有EEG/FIF、npy/npz、pth/pt或临时测试目录，并确认3张图和聚合表出现。
查看完整暂存差异：

```powershell
git diff --cached
```

若误暂存某文件，替换下面的占位路径撤回暂存，工作文件会保留：

```powershell
git restore --staged -- '需要撤回暂存的仓库相对文件路径'
```

## 6. 本地提交

确认暂存差异后执行：

```powershell
git commit -m "Complete P3b classification results and GitHub showcase"
git status --short
git log -1 --oneline
```

提交只是本地历史。本轮没有自动推送。使用者准备上传时可自行执行：

```powershell
git push -u origin main
```

若提示远程有新提交，先fetch并查看差异，按实际冲突处理，不使用force覆盖远程。
项目已有初始提交，不重置历史，也无需创建另一份仓库。

## 7. 页面阅读顺序

访客先读首页主要结果和方法，再点`reports/release/README.md`看配对差值与错误权衡；
需要细节时查docs/04、07、13–15和源码/测试。
首页明确完成范围，长窗TFR、多种子或外部验证作为扩展保留。
