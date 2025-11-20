"""
Generate Three-Algorithm Performance Comparison Charts (ENGLISH VERSION)
Visualize K-Means, GMM, DBSCAN performance comparison
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# Use default font (no Chinese font required)
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['axes.unicode_minus'] = False

# Set project paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def create_algorithm_comparison_chart():
    """Create three-algorithm performance comparison chart (English)"""

    # Experimental data (from reports)
    algorithms = ['K-Means', 'GMM', 'DBSCAN']

    # Data preparation
    silhouette_scores = [0.000804, 0.000743, np.nan]  # DBSCAN single cluster cannot compute
    davies_bouldin = [26.21, 26.29, np.nan]  # DBSCAN cannot compute
    purity_scores = [0.2528, 0.2534, 0.2500]
    n_clusters = [4, 4, 1]
    runtime_seconds = [120, 815, 238]

    # Create figure - 2x3 layout
    fig = plt.figure(figsize=(18, 10))

    # Color scheme
    colors = ['#3498db', '#2ecc71', '#e74c3c']  # Blue, Green, Red

    # ========== Subplot 1: Silhouette Score ==========
    ax1 = plt.subplot(2, 3, 1)
    bars1 = ax1.bar(['K-Means', 'GMM'], silhouette_scores[:2],
                    color=colors[:2], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=0.3, color='red', linestyle='--', linewidth=2, label='Target >0.3')
    ax1.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax1.set_title('Clustering Quality - Silhouette Score', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 0.35])
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.6f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add note
    ax1.text(0.5, 0.05, 'DBSCAN: N/A (single cluster)',
            transform=ax1.transAxes, fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ========== Subplot 2: Davies-Bouldin Index ==========
    ax2 = plt.subplot(2, 3, 2)
    bars2 = ax2.bar(['K-Means', 'GMM'], davies_bouldin[:2],
                    color=colors[:2], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Target <1.0')
    ax2.set_ylabel('Davies-Bouldin Index', fontsize=12, fontweight='bold')
    ax2.set_title('Clustering Quality - Davies-Bouldin', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 30])
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add note
    ax2.text(0.5, 0.05, 'DBSCAN: N/A (single cluster)',
            transform=ax2.transAxes, fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ========== Subplot 3: Cluster Purity ==========
    ax3 = plt.subplot(2, 3, 3)
    bars3 = ax3.bar(algorithms, purity_scores,
                    color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='Target >70%')
    ax3.axhline(y=0.25, color='orange', linestyle=':', linewidth=2, label='Random Baseline 25%')
    ax3.set_ylabel('Cluster Purity', fontsize=12, fontweight='bold')
    ax3.set_title('Cluster Purity Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 0.8])
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height*100:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add warning text
    ax3.text(0.5, 0.85, 'All algorithms near random level',
            transform=ax3.transAxes, fontsize=10, fontweight='bold',
            ha='center', color='red',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # ========== Subplot 4: Number of Clusters ==========
    ax4 = plt.subplot(2, 3, 4)
    bars4 = ax4.bar(algorithms, n_clusters,
                    color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.axhline(y=4, color='red', linestyle='--', linewidth=2, label='Expected K=4')
    ax4.set_ylabel('Number of Clusters', fontsize=12, fontweight='bold')
    ax4.set_title('Cluster Count Comparison', fontsize=14, fontweight='bold')
    ax4.set_ylim([0, 5])
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add note
    ax4.text(0.65, 0.35, 'DBSCAN Failed\nSingle cluster only',
            transform=ax4.transAxes, fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.7),
            ha='center')

    # ========== Subplot 5: Runtime (seconds) ==========
    ax5 = plt.subplot(2, 3, 5)
    bars5 = ax5.bar(algorithms, runtime_seconds,
                    color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax5.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
    ax5.set_ylim([0, 900])
    ax5.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar in bars5:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}s\n({height/60:.1f}min)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ========== Subplot 6: Overall Score (0-10) ==========
    ax6 = plt.subplot(2, 3, 6)

    # Overall score calculation (based on report scorecard)
    quality_scores = [2, 2, 0]  # Quality score /10
    speed_scores = [10, 4, 6]   # Speed score /10
    usability_scores = [10, 6, 2]  # Usability score /10

    # Total score (average)
    total_scores = [(q + s + u) / 3 for q, s, u in
                   zip(quality_scores, speed_scores, usability_scores)]

    bars6 = ax6.bar(algorithms, total_scores,
                    color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax6.set_ylabel('Overall Score', fontsize=12, fontweight='bold')
    ax6.set_title('Overall Performance Score (out of 10)', fontsize=14, fontweight='bold')
    ax6.set_ylim([0, 10])
    ax6.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for i, bar in enumerate(bars6):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}/10',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Add detailed scores
        detail = f'Q:{quality_scores[i]} S:{speed_scores[i]} U:{usability_scores[i]}'
        ax6.text(bar.get_x() + bar.get_width()/2., height/2,
                detail, ha='center', va='center', fontsize=7, rotation=0)

    # Main title
    fig.suptitle('Three-Algorithm Clustering Performance Comparison - AG News Text Clustering',
                fontsize=16, fontweight='bold', y=0.98)

    # Add bottom caption
    fig.text(0.5, 0.02,
            'Data Source: K-Means/GMM/DBSCAN Experimental Reports | Dataset: AG News (120,000 documents) | Embedding: Gemini-768D',
            ha='center', fontsize=10, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save chart
    output_path = RESULTS_DIR / "algorithm_comparison_comprehensive.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Algorithm comparison chart saved: {output_path}")
    print(f"   - Resolution: 300 DPI")
    print(f"   - Size: 18×10 inches")
    print(f"   - Contains 6 subplots: Silhouette, Davies-Bouldin, Purity, Cluster Count, Runtime, Overall Score")

    return output_path

if __name__ == "__main__":
    print("=" * 60)
    print("📊 Generating Three-Algorithm Comparison Charts (English)")
    print("=" * 60)

    # Generate comparison chart
    chart_path = create_algorithm_comparison_chart()

    print("\n" + "=" * 60)
    print("✅ Chart generation complete!")
    print("=" * 60)
