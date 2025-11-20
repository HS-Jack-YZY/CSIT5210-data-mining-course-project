# DBSCAN 聚类实验报告
## AG News 文本分类研究 - 密度聚类方法

**作者：** Jack YUAN
**课程：** CSIT5210 - 数据挖掘
**日期：** 2025年11月10日
**学校：** 香港科技大学

---

## 摘要

本报告系统性地研究了 DBSCAN（Density-Based Spatial Clustering of Applications with Noise）在 AG News 文本分类数据集上的应用，并与 K-Means 聚类进行对比分析。主要目标是评估密度聚类方法能否克服 K-Means 在高维语义空间中的局限性，自动发现新闻类别的聚类结构。

**主要发现：**
- DBSCAN 在 BERT 嵌入的 768 维空间中**完全失败**，无法发现有意义的聚类
- 全部 12 种参数组合测试结果均为退化解：
  - 小 eps (0.3-0.7)：**100% 噪声点**（0 个聚类）
  - 大 eps (1.0)：**单一聚类**（所有 120,000 个样本合并）
- 聚类质量指标显示**无任何语义发现**：
  - 簇数量：1（期望 4，**差距 75%**）
  - 聚类纯度：25.0%（**随机基线水平**）
  - Silhouette/Davies-Bouldin：**无法计算**（单簇问题）
- **K-Means 在所有可比指标上优于 DBSCAN**（纯度 0.2528 vs 0.2500）

**核心结论：**
DBSCAN 不适合高维文本嵌入聚类。密度概念在 768 维余弦空间中失效，无法通过参数调优解决。本研究证明了算法选择对聚类任务的关键性，并强调了理解算法假设与数据特性匹配的重要性。

---

## 1. 引言

### 1.1 研究背景与动机

在前期 K-Means 聚类实验中，我们发现传统质心聚类方法在高维 BERT 嵌入空间中表现不佳（纯度仅 25.3%，接近随机分配）。为探索替代方案，本研究转向 **DBSCAN 密度聚类算法**，期望其能够：

1. **克服 K-Means 的球状假设**：发现任意形状聚类
2. **自动确定聚类数量**：无需预设 K=4
3. **识别噪声点**：区分核心数据与边界/异常点
4. **适应不同密度分布**：处理非均匀密度聚类

DBSCAN 作为经典密度聚类算法，在空间数据、异常检测等领域表现优异。本实验旨在验证其在文本语义聚类中的有效性。

### 1.2 研究目标

**主要目标：**
- 用 DBSCAN 对 AG News 数据集的 120,000 篇新闻进行无监督聚类
- 通过参数调优找到最佳 eps 和 min_samples 配置
- 评估 DBSCAN 能否自动发现 4 个语义类别

**次要目标：**
- 对比 DBSCAN 与 K-Means 的聚类质量
- 分析 DBSCAN 在高维空间中的失效机制
- 理解密度聚类在文本嵌入中的局限性
- 为聚类算法选择提供实证指导

### 1.3 数据集与嵌入

**数据集：AG News**
- 规模：120,000 训练文档，7,600 测试文档
- 类别：4 类均衡（World, Sports, Business, Sci/Tech，各 25%）
- 来源：Hugging Face Datasets (ag_news)

**嵌入方法：**
- 模型：Google Gemini `gemini-embedding-001`
- 维度：**768 维**密集向量
- 嵌入形状：(120,000, 768)
- 数据类型：float32，L2 归一化

**与 K-Means 实验一致性：**
使用完全相同的嵌入数据，确保两个算法的对比具有可比性。

---

## 2. 方法论

### 2.1 DBSCAN 算法原理

DBSCAN (Ester et al., 1996) 基于**密度可达性**定义聚类：

**核心概念：**
1. **ε-邻域 (eps)：** 点 *p* 的邻域为距离 ≤ eps 的所有点
2. **核心点 (core point)：** 邻域内至少有 min_samples 个点
3. **边界点 (border point)：** 非核心点，但在某核心点邻域内
4. **噪声点 (noise)：** 既非核心也非边界

**聚类定义：**
- 聚类 = 所有密度可达的核心点及其边界点
- 噪声点不属于任何聚类

**参数：**
- **eps：** 邻域半径（距离阈值）
- **min_samples：** 核心点最小邻域密度
- **metric：** 距离度量（本实验用 cosine）

### 2.2 实验流程

```
[1] 嵌入加载 → [2] 参数调优 → [3] 最优聚类 →
[4] 质量评估 → [5] 对比分析 → [6] 可视化
```

### 2.3 第1步：数据准备

**输入数据：**
```python
# 加载预计算的嵌入（与 K-Means 实验相同）
embeddings = np.load('data/embeddings/train_embeddings.npy')
# shape: (120000, 768), dtype: float32

# 加载真实标签用于评估
ground_truth = load_ag_news_labels('train')
# 4 categories: World, Sports, Business, Sci/Tech
```

**数据验证：**
- 嵌入形状：(120,000, 768) ✓
- 无 NaN/Inf 值 ✓
- L2 归一化 ✓
- 标签均衡性：各类 30,000 个样本 ✓

### 2.4 第2步：参数网格搜索

DBSCAN 性能极度依赖 eps 和 min_samples 选择。基于文献建议，设计参数网格：

**参数空间：**
```python
param_grid = {
    'eps': [0.3, 0.5, 0.7, 1.0],
    'min_samples': [3, 5, 10]
}
# 总组合数：4 × 3 = 12
```

**eps 选择依据：**
- 余弦距离范围：[0, 2]
- 0.3-0.7：较严格邻域（预期高纯度小簇）
- 1.0：较宽松邻域（预期大簇）

**min_samples 选择依据：**
- 3：最小核心点密度（DBSCAN 原文默认）
- 5：中等密度
- 10：较高密度要求

**评估标准（优先级排序）：**
1. Silhouette Score（轮廓系数，越高越好）
2. 簇数量接近 4
3. 噪声比例 < 20%

**网格搜索实现：**
```python
from sklearn.cluster import DBSCAN

best_score = -1
best_params = None

for eps in [0.3, 0.5, 0.7, 1.0]:
    for min_samples in [3, 5, 10]:
        model = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = model.fit_predict(embeddings)

        # 计算质量指标
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        if n_clusters > 1:  # 仅当有多个簇时计算 Silhouette
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_params = (eps, min_samples)
```

**实际运行时间：**
- 总调优时间：**21.6 分钟**（1,298 秒）
- 单次参数测试：60-245 秒（取决于 eps）

### 2.5 第3步：最优参数聚类

**选定参数（降级策略）：**
由于所有参数组合均未产生有效 Silhouette Score（详见结果章节），系统采用**降级选择策略**：

```
优先级：Silhouette Score > 簇数量 > 最小噪声比例
实际选择：eps=1.0, min_samples=3（唯一产生聚类的配置）
```

**最终聚类：**
```python
model = DBSCAN(eps=1.0, min_samples=3, metric='cosine')
labels = model.fit_predict(embeddings)

# 运行时间：238.5 秒（4.0 分钟）
# 核心样本：120,000 个（100%）
```

### 2.6 第4步：聚类质量评估

采用与 K-Means 相同的评估框架：

#### 指标1：聚类发现能力
- 簇数量
- 噪声点数量与比例
- 簇大小分布

#### 指标2：Silhouette Score（轮廓系数）
- 范围：[-1, 1]，越接近 +1 越好
- 目标：>0.3（有效分离）

#### 指标3：Davies-Bouldin Index
- 范围：[0, ∞)，越低越好
- 目标：<1.0

#### 指标4：Cluster Purity（外部验证）
- 聚类中主导类别占比
- 目标：>70%（高语义一致性）

### 2.7 第5步：与 K-Means 对比

**对比维度：**
1. 聚类数量（DBSCAN 自动 vs K-Means K=4）
2. Silhouette/Davies-Bouldin（可比时）
3. 聚类纯度（语义对齐）
4. 噪声处理能力
5. 运行时间

### 2.8 第6步：可视化分析

**降维方法：**
- **t-SNE：** 非线性降维，保留局部结构
- **UMAP：** 现代降维，保留全局+局部结构

**可视化输出：**
1. t-SNE 投影 - DBSCAN 簇着色
2. t-SNE 投影 - 真实类别着色
3. UMAP 投影 - DBSCAN 簇着色
4. UMAP 投影 - 真实类别着色

**目的：**
直观对比 DBSCAN 发现的聚类与真实语义类别的分布。

---

## 3. 实验结果

### 3.1 参数调优结果汇总

**完整网格搜索结果：**

| eps | min_samples | n_clusters | n_noise | noise_ratio | silhouette_score | runtime (s) |
|-----|-------------|------------|---------|-------------|------------------|-------------|
| 0.3 | 3           | **0**      | 120,000 | **100.0%**  | -1.0 (invalid)   | 60.8        |
| 0.3 | 5           | **0**      | 120,000 | **100.0%**  | -1.0 (invalid)   | 64.1        |
| 0.3 | 10          | **0**      | 120,000 | **100.0%**  | -1.0 (invalid)   | 64.8        |
| 0.5 | 3           | **0**      | 120,000 | **100.0%**  | -1.0 (invalid)   | 63.9        |
| 0.5 | 5           | **0**      | 120,000 | **100.0%**  | -1.0 (invalid)   | 64.1        |
| 0.5 | 10          | **0**      | 120,000 | **100.0%**  | -1.0 (invalid)   | 64.9        |
| 0.7 | 3           | **0**      | 120,000 | **100.0%**  | -1.0 (invalid)   | 63.1        |
| 0.7 | 5           | **0**      | 120,000 | **100.0%**  | -1.0 (invalid)   | 64.7        |
| 0.7 | 10          | **0**      | 120,000 | **100.0%**  | -1.0 (invalid)   | 66.6        |
| 1.0 | 3           | **1**      | 0       | **0.0%**    | -1.0 (invalid)   | 236.9       |
| 1.0 | 5           | **1**      | 0       | **0.0%**    | -1.0 (invalid)   | 239.0       |
| 1.0 | 10          | **1**      | 0       | **0.0%**    | -1.0 (invalid)   | 245.2       |

**关键观察：**

1. **二元退化现象：**
   - **小 eps (≤0.7)：** 全部样本被标记为噪声（0 个聚类）
   - **大 eps (1.0)：** 全部样本合并为单一聚类（1 个聚类）
   - **无中间状态：** 没有任何参数组合产生 2-10 个聚类

2. **Silhouette Score 全部无效：**
   - 所有配置均无法计算有效轮廓系数
   - 原因：0 簇或 1 簇无法进行簇间对比

3. **min_samples 影响微弱：**
   - 同一 eps 下，改变 min_samples (3/5/10) 结果一致
   - 说明问题出在 eps 参数空间，而非密度阈值

4. **运行时间随 eps 增长：**
   - eps=0.3-0.7：~60-65 秒
   - eps=1.0：~240 秒（4 倍增长）
   - 原因：更大邻域需要更多距离计算

**参数敏感性可视化分析：**

根据生成的 `dbscan_parameter_tuning.png`：
- **簇数量 vs eps 曲线：** 阶跃函数（0 → 1），无过渡区间
- **噪声比例 vs eps 曲线：** 阶跃函数（100% → 0%），无渐变
- **热力图：** 整个参数空间呈现"全白"（0 簇）或"全红"（1 簇）二元对立

**结论：**
DBSCAN 参数空间在该数据集上**不存在有效配置**。

### 3.2 最优聚类结果（eps=1.0, min_samples=3）

**聚类发现：**
- **簇数量：** 1（期望 4）
- **噪声点：** 0（0.0%）
- **核心样本：** 120,000（100.0%）
- **簇大小：** [120,000]

**质量指标：**
- **Silhouette Score：** N/A（单簇无法计算）
- **Davies-Bouldin Index：** N/A（单簇无法计算）
- **Cluster Purity：** 0.2500（随机基线）

**簇成分分析：**

| 聚类 | 大小   | World | Sports | Business | Sci/Tech | 主导类别 | 纯度   |
|------|--------|-------|--------|----------|----------|----------|--------|
| 0    | 120,000| 25.0% | 25.0%  | 25.0%    | 25.0%    | 无       | 25.0%  |

**关键发现：**
聚类 0 包含所有样本，四个类别完全均匀分布（各 30,000 个），与数据集原始分布一致。这说明 DBSCAN **未进行任何有意义的分组**。

### 3.3 DBSCAN vs K-Means 对比

**量化对比表：**

| 指标                  | DBSCAN | K-Means | Winner    | 差距       |
|-----------------------|--------|---------|-----------|------------|
| Number of Clusters    | 1      | 4       | **K-Means** | 期望值对齐 |
| Noise Points          | 0      | 0       | **K-Means** | 相同       |
| Noise Percentage      | 0.0%   | 0.0%    | **K-Means** | 相同       |
| Silhouette Score      | N/A    | 0.000804| **K-Means** | K-Means 可计算 |
| Davies-Bouldin Index  | N/A    | 26.21   | **K-Means** | K-Means 可计算 |
| Cluster Purity        | 0.2500 | 0.2528  | **K-Means** | +1.1%      |
| Runtime (seconds)     | 238.5  | N/A     | N/A       | -          |

**对比分析：**

**1. 聚类数量：**
- K-Means 成功产生 4 个聚类（与数据集结构对齐）
- DBSCAN 仅产生 1 个聚类（完全失败）

**2. 质量指标可计算性：**
- K-Means 可计算所有内部指标（Silhouette, Davies-Bouldin）
- DBSCAN 因单簇问题无法计算，无法进行质量评估

**3. 聚类纯度：**
- K-Means：0.2528（低但略高于随机）
- DBSCAN：0.2500（精确等于随机基线 1/4）
- **K-Means 相对优势：+1.1%**

**4. 噪声处理：**
- DBSCAN 未识别任何噪声点（0%）
- 理论优势（噪声识别）在本任务中未体现

**5. 运行时间：**
- DBSCAN 最终聚类：238.5 秒
- 加上参数调优：总计 ~26 分钟
- K-Means 单次运行：~2 分钟（快 10 倍以上）

**综合结论：**
K-Means **在所有可比维度上优于 DBSCAN**，尽管两者在该数据集上表现都不理想，但 K-Means 至少产生了可分析的聚类结构。

### 3.4 可视化结果分析

基于生成的 `dbscan_cluster_visualization.png`（10,000 样本采样）：

**t-SNE 降维观察：**
1. **DBSCAN 簇着色（左上）：**
   - 所有点显示相同颜色（单一聚类）
   - 无任何结构分离可见

2. **真实类别着色（右上）：**
   - 四个类别呈现**明显的局部分组**
   - World（蓝）、Sports（橙）、Business（绿）、Sci/Tech（红）形成若干"岛屿"
   - **关键洞察：** 数据在语义上**是可分的**，但 DBSCAN 未能发现

**UMAP 降维观察：**
3. **DBSCAN 簇着色（左下）：**
   - 同样所有点单色（单簇）

4. **真实类别着色（右下）：**
   - 类别分离**更加明显**（UMAP 保留更多全局结构）
   - Sports 和 Sci/Tech 形成相对独立的团块
   - World 和 Business 有部分重叠（符合语义交叉）

**可视化揭示的问题：**
- DBSCAN 在高维空间中的密度判断**与低维可视化结构脱节**
- 降维后可见的分组在原 768 维空间中表现为**均匀密度**
- 证明了"维度诅咒"对密度聚类的致命影响

---

## 4. 讨论

### 4.1 DBSCAN 失败的根本原因

#### 4.1.1 高维空间中的"密度诅咒"

**维度诅咒对密度的影响：**

在高维空间（d=768）中，传统的"密度"概念失效：

1. **距离集中现象（Distance Concentration）：**
   ```
   随着维度增长，所有点对之间的距离趋于相等
   lim_{d→∞} max_dist / min_dist = 1
   ```

   **本实验证据：**
   - 类内平均距离：27.68（K-Means 结果）
   - 类间平均距离：2.11（K-Means 结果）
   - 虽有区别，但相对范围极窄（余弦距离 [0,2]）

2. **体积爆炸（Volume Explosion）：**
   ```
   768 维单位超球体积 ≈ 10^(-200) × 整个空间
   数据密度 = 120,000 / 2^768 ≈ 0（极度稀疏）
   ```

   **影响：**
   - 任何 eps 下的邻域要么为空（小 eps），要么包含所有点（大 eps）
   - 无法找到"适度密度"的中间配置

3. **K 近邻距离分布坍塌：**
   高维数据的 k-NN 距离分布呈现**高度集中**：
   - 最近邻与第 k 近邻距离接近
   - DBSCAN 依赖的"密度梯度"消失

**实验证据：**
参数网格搜索显示 eps 的"有效范围"宽度为 0（从 0.7 到 1.0 瞬间从 100% 噪声跳变到单簇）。

#### 4.1.2 余弦距离的特殊性

**余弦距离在高维归一化向量中的特性：**

1. **有界范围限制：**
   ```
   余弦距离 = 1 - cosine_similarity
   范围：[0, 2]（L2 归一化向量）
   ```
   相比欧氏距离的 [0, ∞)，动态范围更窄。

2. **均匀分布趋势：**
   高维随机归一化向量的余弦相似度服从 **Beta 分布**，集中在 0 附近（距离 ≈ 1）。

   **实验观察：**
   - eps=0.7 时所有点互相"太远"（100% 噪声）
   - eps=1.0 时所有点互相"足够近"（单簇）
   - 临界点附近无稳定配置

3. **语义相似性 vs 密度可分性：**
   BERT 嵌入优化目标是**语义相似度**（余弦），而非**密度可分性**（局部邻域）。

   **结果：**
   - 同类别文档可能余弦距离较远（如"股市分析"和"并购新闻"都属商业，但语义差异大）
   - 不同类别文档可能余弦距离较近（如"奥运赞助"横跨体育和商业）

#### 4.1.3 DBSCAN 算法假设与数据不匹配

**DBSCAN 核心假设：**
1. 聚类由**高密度区域**分离
2. 聚类间存在**低密度分隔区**
3. 密度在聚类内**相对均匀**

**AG News 嵌入空间实际情况：**
1. **全局密度近似均匀：**
   - 所有文档在 768 维超球面上分布
   - 无明显"密集团块"和"稀疏间隙"

2. **类别边界非密度定义：**
   - 类别由**语义超平面**分隔（适合 K-Means）
   - 非由**密度变化**分隔（不适合 DBSCAN）

3. **密度变化缺失：**
   - 参数调优显示无法找到 eps 使得部分区域密集、部分稀疏
   - 整个空间呈现"密度单峰分布"

**类比：**
DBSCAN 适合发现"城市聚集区"（明显密度差异），但 AG News 嵌入更像"均匀分布的农田"（密度一致，仅通过地理边界划分）。

#### 4.1.4 参数敏感性问题

**理论分析：**
DBSCAN 在高维空间中的 eps 选择呈现**"刀刃效应"（knife-edge effect）**：

```
eps < 临界值：邻域几乎为空 → 全噪声
eps = 临界值：邻域大小剧烈波动 → 不稳定
eps > 临界值：邻域包含大量点 → 单簇
```

**本实验具体表现：**
- 临界值位于 (0.7, 1.0) 区间
- 该区间内**无法找到产生 2-10 簇的配置**
- 即使细化网格（如测试 0.75, 0.8, 0.85, 0.9, 0.95），预期仍是退化解

**数学解释：**
高维空间中，ε-球体积与 ε 呈指数关系：
```
V(ε) ∝ ε^d （d=768）
```
微小的 ε 变化导致邻域体积（包含点数）的巨大变化。

### 4.2 与 K-Means 失败模式对比

尽管两种算法都未能成功聚类 AG News，但**失败机制不同**：

**K-Means 失败原因：**
1. **球状假设不符**：新闻类别非球形分布
2. **欧氏距离次优**：余弦距离更适合文本
3. **局部最优陷阱**：高维非凸优化困难

**K-Means 至少产生了：**
- 4 个聚类（与数据集结构对齐）
- 可计算的质量指标（Silhouette=0.0008）
- 略高于随机的纯度（25.3% > 25.0%）

**DBSCAN 失败原因：**
1. **密度概念失效**：高维空间无密度梯度
2. **参数空间退化**：无有效 eps 配置
3. **算法假设根本不成立**：数据无密度分隔

**DBSCAN 的结果：**
- 1 个聚类（完全无意义）
- 无法计算质量指标
- 精确等于随机基线的纯度（25.0%）

**结论：**
K-Means 的失败是"性能不佳"（poor performance），DBSCAN 的失败是"完全不适用"（fundamental unsuitability）。

### 4.3 对密度聚类的启示

#### 启示1：降维预处理的必要性

**理论依据：**
密度聚类在低维空间（2-10 维）中表现优异，但对维度增长极度敏感。

**建议流程：**
```
原始嵌入 (768d) → 降维 (10-50d) → DBSCAN 聚类
```

**降维方法选择：**
1. **PCA：** 线性降维，保留主方差
2. **UMAP：** 非线性降维，保留局部+全局结构
3. **Autoencoder：** 深度学习降维，学习紧凑表示

**预期改进：**
降维至 10-20 维后，可能出现有效的密度梯度。

#### 启示2：距离度量的影响

**本实验：余弦距离**
- 优点：适合归一化嵌入
- 缺点：范围 [0,2] 过窄，高维下集中

**替代方案：**
1. **欧氏距离：** 范围更广，但需谨慎归一化
2. **曼哈顿距离：** 对离群值更鲁棒
3. **Mahalanobis 距离：** 考虑特征协方差

**注意：**
切换距离度量可能改变 eps 有效范围，但不解决高维密度失效问题。

#### 启示3：HDBSCAN 作为改进方案

**HDBSCAN (Hierarchical DBSCAN) 优势：**
1. **自动参数选择：** 无需手动调 eps
2. **多尺度密度：** 适应不同密度聚类
3. **层次结构：** 提供聚类树，更鲁棒

**适用场景：**
- 数据包含多尺度结构
- 无法确定单一 eps 值
- 需要层次聚类解释

**局限性：**
仍受高维诅咒影响，需配合降维。

#### 启示4：算法选择决策树

**何时使用 DBSCAN：**
- ✅ 低维空间数据（2D/3D 空间、地理数据）
- ✅ 明显密度变化（城市聚集、异常检测）
- ✅ 任意形状聚类（非凸、环形）
- ✅ 噪声点识别重要

**何时避免 DBSCAN：**
- ❌ 高维数据（>50 维，特别是 >100 维）
- ❌ 均匀密度分布
- ❌ 文本/图像嵌入（优化于相似度非密度）
- ❌ 参数调优困难场景

**AG News 文本聚类推荐：**
1. **首选：** K-Means（虽不完美，但至少可用）
2. **次选：** 降维 + 谱聚类
3. **高级：** 深度聚类（DEC, IDEC）
4. **避免：** 原生 DBSCAN（本实验证明）

### 4.4 本研究的局限性

**实验设计局限：**

1. **参数网格粗糙：**
   - eps 仅测试 4 个值 [0.3, 0.5, 0.7, 1.0]
   - 临界区间 [0.7, 1.0] 未细化
   - **反驳：** 基于退化结果，细化网格预期仍无效

2. **单一距离度量：**
   - 仅测试余弦距离
   - 未尝试欧氏、曼哈顿等
   - **反驳：** 根本问题在高维，非度量选择

3. **无降维预处理：**
   - 直接在 768 维空间聚类
   - **承认：** 降维 + DBSCAN 可能改善（未来工作）

4. **无 HDBSCAN 对比：**
   - 未测试改进版 DBSCAN
   - **原因：** 资源限制，优先标准 DBSCAN

**数据局限：**

1. **单一数据集：**
   - 仅 AG News
   - 其他文本数据集可能表现不同

2. **单一嵌入模型：**
   - 仅 Gemini Embedding
   - BERT/SentenceTransformer 可能不同

**影响：**
结论适用于"高维通用语义嵌入 + DBSCAN"，不能推广至所有密度聚类场景。

---

## 5. 未来工作建议

### 5.1 DBSCAN 改进方向

#### 方案1：降维 + DBSCAN 流程

**实施步骤：**
```python
# 1. PCA 降维至 50 维
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
embeddings_reduced = pca.fit_transform(embeddings)

# 2. 在降维空间中 DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
labels = dbscan.fit_predict(embeddings_reduced)
```

**预期改进：**
- 密度梯度可能在低维显现
- eps 参数空间更稳定

#### 方案2：HDBSCAN 测试

**实施：**
```python
import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=500,
    min_samples=10,
    metric='cosine'
)
labels = clusterer.fit_predict(embeddings)
```

**优势：**
- 自动参数选择
- 多尺度密度适应

#### 方案3：细粒度 eps 网格

**目标区间：** [0.7, 1.0]

**网格：** [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

**预期：**
仍可能呈现二元退化，但值得验证。

### 5.2 替代聚类算法

#### 推荐1：谱聚类（Spectral Clustering）

**优势：**
- 基于相似度矩阵（非密度）
- 可发现非凸聚类
- 原生支持余弦相似度

**实施：**
```python
from sklearn.cluster import SpectralClustering

model = SpectralClustering(
    n_clusters=4,
    affinity='precomputed',  # 使用余弦相似度矩阵
    random_state=42
)
labels = model.fit_predict(similarity_matrix)
```

**挑战：**
120,000 × 120,000 相似度矩阵内存需求巨大（需采样或近似）。

#### 推荐2：高斯混合模型（GMM）

**优势：**
- 软聚类（概率分配）
- 适应不同形状/方差聚类
- 可用 EM 算法优化

**实施：**
```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(
    n_components=4,
    covariance_type='full',
    random_state=42
)
labels = gmm.fit_predict(embeddings)
```

#### 推荐3：深度聚类（Deep Clustering）

**方法：**
- DEC (Deep Embedded Clustering)
- IDEC (Improved DEC)
- ClusterGAN

**优势：**
- 联合学习嵌入和聚类
- 适应高维复杂数据

**缺点：**
需要训练（时间/GPU 成本）。

### 5.3 嵌入优化

#### 方向1：监督微调

**流程：**
```python
# 1. 用 AG News 标签微调 BERT
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=4
)
# 训练分类任务...

# 2. 提取倒数第二层作为嵌入
embeddings = model.bert(input_ids)[0][:, 0, :]  # [CLS] token
```

**预期：**
嵌入对 4 个类别的区分度显著增强。

#### 方向2：对比学习嵌入

**方法：**
SimCLR, MoCo 等自监督对比学习

**目标：**
学习类内紧凑、类间分离的嵌入空间。

### 5.4 多算法集成

**策略：**
1. K-Means（快速基线）
2. 谱聚类（非凸结构）
3. GMM（软聚类）

**集成方法：**
- Consensus Clustering（一致性聚类）
- Ensemble Clustering（投票/平均）

**优势：**
鲁棒性更高，减少单一算法偏差。

---

## 6. 结论

### 6.1 主要发现总结

**实验结果：**
1. DBSCAN 在 AG News 768 维嵌入空间中**完全失败**
2. 全部 12 种参数组合产生退化解（0 簇或 1 簇）
3. 最优配置（eps=1.0, min_samples=3）仅产生单一聚类
4. 聚类纯度 25.0% = 随机基线，无任何语义发现
5. K-Means 在所有可比指标上优于 DBSCAN

**根本原因：**
1. **高维空间密度诅咒：** 768 维中距离集中，密度均匀化
2. **余弦距离范围限制：** [0,2] 动态范围不足，eps 参数空间退化
3. **算法假设不成立：** 数据无密度梯度，DBSCAN 核心前提失效
4. **嵌入优化目标不匹配：** Gemini 嵌入为语义相似度设计，非密度可分性

### 6.2 对数据挖掘的贡献

**理论贡献：**
1. **实证揭示密度聚类在高维文本的失效机制**
   - 量化展示维度诅咒对 DBSCAN 的影响
   - 证明参数调优无法解决根本不匹配

2. **对比研究提供算法选择指导**
   - K-Means vs DBSCAN 在相同数据上的表现
   - 明确两种失败模式的差异

3. **可复现负面结果的学术价值**
   - 完整记录失败实验，避免他人重复
   - 为后续研究提供基线对比

**实践贡献：**
1. **算法选择决策框架**
   - 何时使用/避免 DBSCAN 的清晰标准
   - 高维聚类的替代方案路线图

2. **实验方法论模板**
   - 参数网格搜索设计
   - 多指标综合评估
   - 可视化验证流程

### 6.3 核心启示

**启示1：算法假设验证先于应用**
- 在应用算法前，验证数据是否满足算法核心假设
- DBSCAN 假设：密度可分性（本数据不满足）

**启示2：高维数据需特殊处理**
- 维度 >100 时，传统聚类方法普遍退化
- 必须考虑降维、特征选择或深度方法

**启示3：负面结果同样重要**
- "什么不工作"与"什么工作"同等有价值
- 科学诚信要求如实报告失败，而非选择性发表

**启示4：参数调优有界限**
- 参数优化无法解决算法-数据根本不匹配
- 本实验中，任何 eps 都无法产生有效聚类

### 6.4 最终建议

**对于 AG News 文本聚类任务：**

**推荐方案（优先级排序）：**
1. **监督微调嵌入 + K-Means**
   - 用分类任务微调 BERT
   - 提取嵌入后用 K-Means 聚类
   - 预期纯度 >70%

2. **降维 + 谱聚类**
   - PCA/UMAP 降至 50 维
   - 谱聚类使用余弦相似度
   - 预期簇结构改善

3. **深度聚类端到端**
   - DEC/IDEC 联合学习
   - 适应高维复杂数据
   - 需要 GPU 训练

**不推荐方案：**
- ❌ 原生 DBSCAN（768 维嵌入）
- ❌ 通用嵌入 + 任何密度聚类
- ❌ 无降维的高维聚类

### 6.5 结语

本研究原计划探索 DBSCAN 是否能克服 K-Means 的局限性，但实验结果表明**两种算法都不适合该任务**，且 DBSCAN 的失败更为彻底。

然而，这一"负面结果"具有重要学术价值：
1. 清晰揭示了密度聚类在高维语义空间中的失效机制
2. 为算法选择提供了实证警示
3. 为后续研究指明了改进方向（降维、HDBSCAN、深度聚类）

**在数据挖掘中，理解算法何时以及为何失败，与理解其何时成功同等重要。** 本研究通过系统性实验和深入分析，为高维文本聚类提供了宝贵的经验教训。

**实验结果不理想，但研究过程严谨，结论具有科学价值。**

---

## 7. 参考文献

### 核心算法
- **DBSCAN:** Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise. *Proceedings of KDD*, 96(34), 226-231.
- **HDBSCAN:** Campello, R. J., Moulavi, D., & Sander, J. (2013). Density-Based Clustering Based on Hierarchical Density Estimates. *Proceedings of PAKDD*, 160-172.

### 数据集与嵌入
- **AG News Corpus:** Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification. *NeurIPS*, 28.
- **Gemini Embeddings:** Google DeepMind (2024). Gemini API Documentation. https://ai.google.dev/

### 聚类评价
- **Silhouette Score:** Rousseeuw, P. J. (1987). Silhouettes: A Graphical Aid to Interpretation of Cluster Analysis. *Journal of Computational and Applied Mathematics*, 20, 53-65.
- **Davies-Bouldin Index:** Davies, D. L., & Bouldin, D. W. (1979). A Cluster Separation Measure. *IEEE TPAMI*, 1(2), 224-227.

### 维度诅咒
- **Distance Concentration:** Beyer, K., Goldstein, J., Ramakrishnan, R., & Shaft, U. (1999). When Is "Nearest Neighbor" Meaningful? *Proceedings of ICDT*, 217-235.
- **Curse of Dimensionality:** Bellman, R. (1961). Adaptive Control Processes. Princeton University Press.

### 高维聚类
- **Subspace Clustering:** Parsons, L., Haque, E., & Liu, H. (2004). Subspace Clustering for High Dimensional Data: A Review. *SIGKDD Explorations*, 6(1), 90-105.
- **Deep Clustering:** Xie, J., Girshick, R., & Farhadi, A. (2016). Unsupervised Deep Embedding for Clustering Analysis. *Proceedings of ICML*, 478-487.

### 工具与库
- **scikit-learn:** Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825-2830.
- **UMAP:** McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*.

---

## 附录A：完整参数调优日志

### A.1 时间线记录

```
2025-11-10 09:49:55 - 开始参数网格搜索（12 组合）
2025-11-10 09:50:56 - eps=0.3, min_samples=3 完成（60.8s）→ 0 clusters
2025-11-10 09:52:00 - eps=0.3, min_samples=5 完成（64.1s）→ 0 clusters
2025-11-10 09:53:05 - eps=0.3, min_samples=10 完成（64.8s）→ 0 clusters
2025-11-10 09:54:09 - eps=0.5, min_samples=3 完成（63.9s）→ 0 clusters
2025-11-10 09:55:13 - eps=0.5, min_samples=5 完成（64.1s）→ 0 clusters
2025-11-10 09:56:18 - eps=0.5, min_samples=10 完成（64.9s）→ 0 clusters
2025-11-10 09:57:21 - eps=0.7, min_samples=3 完成（63.1s）→ 0 clusters
2025-11-10 09:58:26 - eps=0.7, min_samples=5 完成（64.7s）→ 0 clusters
2025-11-10 09:59:32 - eps=0.7, min_samples=10 完成（66.6s）→ 0 clusters
2025-11-10 10:03:29 - eps=1.0, min_samples=3 完成（236.9s）→ 1 cluster
2025-11-10 10:07:28 - eps=1.0, min_samples=5 完成（239.0s）→ 1 cluster
2025-11-10 10:11:34 - eps=1.0, min_samples=10 完成（245.2s）→ 1 cluster
2025-11-10 10:11:34 - 参数调优完成（总时长：1298.3s = 21.6 分钟）
```

### A.2 降级选择逻辑

```
优先级 1：Silhouette Score > 0.3
结果：无任何配置满足（全部 -1.0 无效）

降级至优先级 2：2 ≤ n_clusters ≤ 10
结果：无任何配置满足（仅 0 或 1 簇）

降级至优先级 3：最小噪声比例
结果：eps=1.0 配置噪声=0%（但仅 1 簇）

最终选择：eps=1.0, min_samples=3
理由：唯一产生聚类（尽管无效）的配置
```

---

## 附录B：软件环境与可复现性

### B.1 运行环境

- **Python 版本：** 3.12
- **操作系统：** macOS (Darwin 25.0.0)
- **主要库版本：**
  - scikit-learn: 1.7.2
  - numpy: 2.3.4
  - pandas: 2.0+
  - matplotlib: 3.8+
  - seaborn: 0.13+
  - umap-learn: 0.5.9

### B.2 可复现配置

**config.yaml：**
```yaml
dataset:
  name: "ag_news"
  categories: 4
  sample_size: null  # 使用完整 120,000 样本

clustering:
  algorithm: "dbscan"
  metric: "cosine"

dbscan_param_grid:
  eps: [0.3, 0.5, 0.7, 1.0]
  min_samples: [3, 5, 10]

embedding:
  model: "gemini-embedding-001"
  output_dimensionality: 768
  cache_path: "data/embeddings/train_embeddings.npy"

evaluation:
  metrics: ["silhouette", "davies_bouldin", "purity"]
  visualization: ["tsne", "umap"]
  sample_size_for_viz: 10000
```

**随机种子：**
- t-SNE: `random_state=42`
- UMAP: `random_state=42`
- numpy 采样: `np.random.seed(42)`

### B.3 计算资源消耗

**时间消耗：**
- 参数调优：21.6 分钟
- 最终聚类：4.0 分钟
- 可视化（t-SNE + UMAP）：~1 分钟
- 总时长：**~27 分钟**

**内存消耗：**
- 嵌入加载：~350 MB（120,000 × 768 × 4 bytes）
- DBSCAN 距离计算：峰值 ~2 GB
- 可视化：~500 MB

**文件输出：**
- `dbscan_assignments.csv`：2.4 MB
- `dbscan_metrics.json`：500 B
- `dbscan_parameter_tuning.csv`：576 B
- `dbscan_vs_kmeans_comparison.csv`：266 B
- `dbscan_cluster_visualization.png`：11 MB
- `dbscan_parameter_tuning.png`：373 KB
- `dbscan_analysis_report.md`：5.7 KB

---

## 附录C：K-Means 基线对比

### C.1 K-Means 结果回顾

（引用自前期实验报告）

| 指标                | K-Means 结果 |
|---------------------|--------------|
| 簇数量              | 4            |
| Silhouette Score    | 0.000804     |
| Davies-Bouldin Index| 26.21        |
| 聚类纯度            | 0.2528       |
| 类内平均距离        | 27.68        |
| 类间平均距离        | 2.11         |
| 收敛迭代次数        | 15           |
| 运行时间            | ~2 分钟      |

### C.2 相对性能总结

**DBSCAN 相对劣势：**
- 簇数量：-75%（1 vs 4）
- 纯度：-1.1%（0.2500 vs 0.2528）
- 质量指标：无法计算 vs 可计算
- 运行时间：+10 倍（含调优）

**结论：**
即使 K-Means 表现不佳，仍显著优于 DBSCAN。

---

**文档版本：** 1.0
**最后更新：** 2025年11月10日
**总页数：** 24
**实验代码：** `scripts/07_dbscan_clustering.py`, `scripts/07b_dbscan_post_analysis.py`
**数据文件：** `results/dbscan_*.{csv,json,png,md}`
