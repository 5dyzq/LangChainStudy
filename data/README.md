# Data

本目录用于存放本地数据和数据说明。

当前数据说明来自 `数据描述.docx`，核心文件包括：

- `data/[subject_id]_acceleration.txt`
- `data/[subject_id]_labeled_sleep.txt`

由于原始 TXT 文件较大，默认不提交到 GitHub。协作时建议：

1. 保留 `data/README.md` 和少量脱敏小样例。
2. 将真实原始数据放在本地 `data/data/`。
3. 在文档中记录数据来源、版本和下载/同步方式。

第一版假设：

- 加速度采样率约为 50Hz。
- 睡眠标签每 30 秒一条。
- 睡眠标签 `0` 为 Wake，`1/2/3/5` 为 Sleep。
