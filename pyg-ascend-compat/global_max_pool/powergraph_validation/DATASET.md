# PowerGraph 数据集来源与获取说明

本文档说明本次验证**实际使用了哪一份上游数据**、如何获取、如何校验，以及上游对许可证
的表述。**PowerGraph 原始数据不随本仓库提交** —— 解压后约 2.75 GiB。

---

## 1. 数据来源

```text
PowerGraph-Graph 仓库 : https://github.com/PowerGraph-Datasets/PowerGraph-Graph
本次使用的 commit     : eb100a2fd836bb8b6bd2d0b799af9c615eac8cb6
```

该仓库在本验证中**只读使用**：我们从该 checkout 直接 import 未修改的
`code/dataset/powergrid.py`（`PowerGrid` loader），不向其写入任何内容。

数据集本身不在该 git 仓库中，上游 README 指向一个 figshare 对象。

## 2. 实际使用的数据文件

```text
对象   : PowerGraph（figshare article 22820534）
文件   : dataset_cascades.zip
file id: 46619158               <- 上游 README 链接的文件
URL    : https://figshare.com/ndownloader/files/46619158
大小   : 61,628,977 bytes
md5    : 70b677416d2f377ccfee9f51d8369867
解压后 : 2,958,249,040 bytes（2.75 GiB）
DOI    : 10.6084/m9.figshare.22820534
         （article 页面：https://figshare.com/articles/dataset/PowerGraph/22820534）
```

原始验证期间做过的交叉核对：该 article 的更新版本（v5）提供同样名为
`dataset_cascades.zip` 的文件，file id 为 `50083479`；该归档也已下载，其 md5 与
figshare 公布的数值一致（`d4d144b9e720a760e1e077a31f34802d`）。两个归档内的原始文件
名称与字节大小完全一致，v5 只是多了一层顶层目录。本报告中的全部数字来自上游 README
所链接的 v3 文件（`46619158`）。

## 3. 包含的数据集

| 数据集 | graph 数 | 每图节点数 | F | 每图边数（有向） |
|---|---|---|---|---|
| `ieee24` | 21,500 | 24 | 3 | 68–74 |
| `ieee39` | 28,000 | 39 | 3 | 86–90 |
| `ieee118` | 122,500 | 118 | 3 | 362–370 |
| `uk` | 64,000 | 29 | 3 | 190–196 |

每个数据集的每图节点数是固定的；每图边数会变化，因为每个 graph 有 1–5 条被切除的支路
会被 loader 删除，之后前向边再被复制为双向边。节点特征恒为 `float32 [N, 3]`
（net active power、net apparent power、voltage magnitude），因此所有数据集的
**F = 3**。机器可读的审计结果见 `results/phase_a_raw_audit.json`。

## 4. 期望的目录结构

上游 loader 解析 `raw_dir = <root>/<name>/<name>/raw`，其中 `<root>` 即
`POWERGRAPH_DATA_ROOT`。解压后：

```text
$POWERGRAPH_DATA_ROOT/
├── ieee24/ieee24/raw/{Bf.mat,blist.mat,Ef.mat,exp.mat,of_bi.mat,of_mc.mat,of_reg.mat}
├── ieee39/ieee39/raw/{...}
├── ieee118/ieee118/raw/{...}
└── uk/uk/raw/{...}
```

归档中的 `Ef_nc.mat` 会被**有意跳过**：它不在 loader 的 `raw_file_names` 中，没有任何
代码读取它，而它占了 2.96 GB 归档中约 1.1 GB。

首次使用时，loader 会生成
`$POWERGRAPH_DATA_ROOT/<name>/<name>/processed_b/data.pt`
（ieee24 约 65 MB、ieee39 约 105 MB、ieee118 约 1.81 GB、uk 约 0.5 GB）。
这些 processed 文件同样不随仓库提交。

## 5. 获取步骤

```bash
cd pyg-ascend-compat/global_max_pool/powergraph_validation

# 1) 上游 loader checkout（只读）
git clone https://github.com/PowerGraph-Datasets/PowerGraph-Graph.git \
    upstream/PowerGraph-Graph

# 2) 数据集归档（官方 figshare 地址，自动校验 md5）
bash scripts/fetch_powergraph_data.sh

# 3) 解压到 POWERGRAPH_DATA_ROOT
python3 scripts/extract_powergraph_data.py
```

`fetch_powergraph_data.sh` 支持的环境变量：

| 环境变量 | 含义 |
|---|---|
| `POWERGRAPH_DATA_URL` | 归档下载地址（默认：上面的官方 figshare ndownloader 地址） |
| `POWERGRAPH_DATA_ARCHIVE` | 输出 zip 路径（默认 `<package>/data_download/dataset_cascades.zip`） |
| `POWERGRAPH_DATA_MD5` | 期望的 md5（默认 `70b677416d2f377ccfee9f51d8369867`；设为 `skip` 可跳过校验） |
| `POWERGRAPH_FIGSHARE_PROXY` | 可选 HTTP proxy，仅在直连失败时使用 |
| `POWERGRAPH_FIGSHARE_FILE_ID` | proxy 回退路径使用的 figshare file id（默认 `46619158`） |

### 5.1 关于原始验证网络的说明

在**原始验证环境**中，`figshare.com` 对所有路径都返回 HTTP 403（article 页面、
`api.figshare.com`、`ndownloader.figshare.com` 均如此，IPv4 与 IPv6 相同）。当时的数据
获取方式是：通过 HTTP proxy 向 figshare 请求其 302 重定向，再从
`s3-eu-west-1.amazonaws.com/pfigshare-u-files/...` **直接下载 payload** —— payload
传输本身不经过 proxy。

打包后的脚本只把该方式作为回退：它**优先直连官方 URL**，只有在直连失败且设置了
`POWERGRAPH_FIGSHARE_PROXY` 时才走 proxy。脚本**不依赖任何一次性的 presigned URL**
（这类 URL 只有数秒有效期），每次都会重新从官方 figshare 重定向获取；两条路径都会校验
期望的 md5。

## 6. 许可证与署名

PowerGraph 是第三方数据集。本仓库**不重新分发该数据集**，只保存由它得到的测试结果
数据。上游项目没有单独的 `LICENSE` 文件，且不同位置的许可证表述**并不完全一致**，
因此这里如实并列引用，不做额外裁定：

| 位置 | 表述 |
|---|---|
| `PowerGraph-Graph/README.md`（`## License`） | *“This work is licensed under a CC BY 4.0 license.”* |
| `PowerGraph-Graph/code/dataset/powergrid.py`（文件头） | *“PowerGrid dataset is licensed under a CC BY-SA 4.0 license.”* |
| figshare article 22820534 记录（`api.figshare.com` 返回） | license 名称为 `CC BY 4.0`，<https://creativecommons.org/licenses/by/4.0/> |

> 上游不同位置的许可证表述存在差异，本仓库仅如实记录，不对其法律含义作额外判断。
> 正式使用和再分发数据时请以上游授权要求为准。

希望再分发数据集本身的读者，请直接查阅上游仓库与 figshare 记录：

* <https://github.com/PowerGraph-Datasets/PowerGraph-Graph>（见 `README.md` 与 `code/dataset/powergrid.py`）
* <https://figshare.com/articles/dataset/PowerGraph/22820534>

## 7. 哪些内容入库、哪些需要重新生成

| 内容 | 是否入库 |
|---|---|
| 原始 `.mat` 文件（解压后 2.75 GiB） | **否** —— 用 `fetch_powergraph_data.sh` 下载 |
| `dataset_cascades.zip` 归档 | **否** —— 已被 git 忽略 |
| PyG `processed_b/data.pt` | **否** —— 由 loader 重新生成 |
| 上游 `PowerGraph-Graph` checkout | **否** —— clone 到 `upstream/`（已被 git 忽略） |
| 性能测试结果 CSV / JSON | **是** —— 位于 `results/` |
| 解析后的 Profiler gate 记录与各 case msprof CSV | **是** —— 位于 `evidence/profiler/` |
| 原始 msprof 输出目录 | **否** —— 用 `run_profiles.sh` 重新生成 |
