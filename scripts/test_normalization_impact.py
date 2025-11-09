#!/usr/bin/env python3
"""
Quick test: Does L2 normalization improve K-Means clustering?
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from collections import Counter

print('=' * 80)
print('🧪 实验: 归一化嵌入后 K-Means 性能')
print('=' * 80)
print()

# 加载数据
print('📊 加载数据...')
embeddings = np.load('data/embeddings/train_embeddings.npy')
from datasets import load_dataset
dataset = load_dataset('ag_news', split='train')
true_labels = np.array([item['label'] for item in dataset])
print(f'   加载了 {len(embeddings)} 个嵌入向量')
print()

# 原始嵌入（未归一化）性能
print('🔍 测试 1: 原始嵌入（未归一化）')
print('   运行 K-Means...')
kmeans_original = KMeans(n_clusters=4, random_state=42, init='k-means++', n_init=1, max_iter=100)
labels_original = kmeans_original.fit_predict(embeddings)

# 计算 Silhouette Score（采样加速）
sample_size = min(10000, len(embeddings))
np.random.seed(42)
sample_idx = np.random.choice(len(embeddings), sample_size, replace=False)
silhouette_original = silhouette_score(embeddings[sample_idx], labels_original[sample_idx])

# 计算纯度
purities_original = []
for cluster_id in range(4):
    mask = labels_original == cluster_id
    cluster_true_labels = true_labels[mask]
    if len(cluster_true_labels) > 0:
        most_common = Counter(cluster_true_labels).most_common(1)[0][1]
        purity = most_common / len(cluster_true_labels)
        purities_original.append(purity)
avg_purity_original = np.mean(purities_original)

print(f'   ✅ 完成')
print(f'   Silhouette Score: {silhouette_original:.6f}')
print(f'   聚类纯度: {avg_purity_original:.4f} ({avg_purity_original*100:.2f}%)')
print()

# 归一化嵌入后的性能
print('🔍 测试 2: 归一化嵌入（L2 normalization）')
print('   对嵌入进行 L2 归一化...')
embeddings_normalized = normalize(embeddings, norm='l2')
print(f'   归一化后的范数: {np.linalg.norm(embeddings_normalized[0]):.6f} (应该 ≈ 1.0)')
print('   运行 K-Means...')

kmeans_normalized = KMeans(n_clusters=4, random_state=42, init='k-means++', n_init=1, max_iter=100)
labels_normalized = kmeans_normalized.fit_predict(embeddings_normalized)

# 计算指标
silhouette_normalized = silhouette_score(embeddings_normalized[sample_idx], labels_normalized[sample_idx])

purities_normalized = []
for cluster_id in range(4):
    mask = labels_normalized == cluster_id
    cluster_true_labels = true_labels[mask]
    if len(cluster_true_labels) > 0:
        most_common = Counter(cluster_true_labels).most_common(1)[0][1]
        purity = most_common / len(cluster_true_labels)
        purities_normalized.append(purity)
avg_purity_normalized = np.mean(purities_normalized)

print(f'   ✅ 完成')
print(f'   Silhouette Score: {silhouette_normalized:.6f}')
print(f'   聚类纯度: {avg_purity_normalized:.4f} ({avg_purity_normalized*100:.2f}%)')
print()

# 对比结果
print('=' * 80)
print('📊 结果对比')
print('=' * 80)
print()
print(f'{"指标":<20} | {"原始嵌入":>12} | {"归一化嵌入":>12} | {"改进":>10}')
print('-' * 70)
print(f'{"Silhouette Score":<20} | {silhouette_original:12.6f} | {silhouette_normalized:12.6f} | {((silhouette_normalized - silhouette_original) / abs(silhouette_original + 1e-10) * 100):+9.1f}%')
print(f'{"聚类纯度":<20} | {avg_purity_original:12.4f} | {avg_purity_normalized:12.4f} | {((avg_purity_normalized - avg_purity_original) / avg_purity_original * 100):+9.1f}%')
print()

# 结论
print('=' * 80)
print('💡 结论')
print('=' * 80)
improvement_pct = (avg_purity_normalized - avg_purity_original) / avg_purity_original * 100

if improvement_pct > 5:
    print('✅ 归一化有显著改善！')
    print(f'   聚类纯度提升: {(avg_purity_normalized - avg_purity_original)*100:.2f} 个百分点')
    print(f'   相对改进: {improvement_pct:.1f}%')
elif improvement_pct > 0:
    print('⚠️ 归一化有轻微改善')
    print(f'   聚类纯度提升: {(avg_purity_normalized - avg_purity_original)*100:.2f} 个百分点')
    print(f'   相对改进: {improvement_pct:.1f}%')
else:
    print('❌ 归一化没有改善')
    print('   问题不在于归一化，而是更根本的维度诅咒')
print()
