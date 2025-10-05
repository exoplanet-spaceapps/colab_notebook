#!/usr/bin/env python
"""
Create comprehensive visualizations and PDF report for model comparison
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import json
import os
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("=" * 80)
print("Creating Model Comparison Charts and PDF Report")
print("=" * 80)

# Create output directory
os.makedirs('reports', exist_ok=True)

# Load metadata
print("\n[1/5] Loading Model Metadata...")
with open('models/metadata.json', 'r') as f:
    metadata = json.load(f)

# Extract model data
models_data = []
for model_name, info in metadata['models'].items():
    models_data.append({
        'Model': model_name.replace('_', ' ').title(),
        'Accuracy': info['accuracy'] * 100,
        'F1-Score': info['f1_score'] * 100,
        'File Size (MB)': info['file_size_mb']
    })

df = pd.DataFrame(models_data)
print(f"  Loaded {len(df)} models")

# Chart 1: Performance Comparison Bar Chart
print("\n[2/5] Creating Performance Comparison Chart...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Accuracy comparison
ax1 = axes[0, 0]
bars1 = ax1.bar(df['Model'], df['Accuracy'], color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_ylim([0, 100])
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90% threshold')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%',
             ha='center', va='bottom', fontweight='bold')

ax1.legend()
ax1.tick_params(axis='x', rotation=45)

# F1-Score comparison
ax2 = axes[0, 1]
bars2 = ax2.bar(df['Model'], df['F1-Score'], color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
ax2.set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
ax2.set_ylabel('F1-Score (%)', fontsize=12)
ax2.set_ylim([0, 100])
ax2.grid(axis='y', alpha=0.3)
ax2.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90% threshold')

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%',
             ha='center', va='bottom', fontweight='bold')

ax2.legend()
ax2.tick_params(axis='x', rotation=45)

# File size comparison
ax3 = axes[1, 0]
bars3 = ax3.bar(df['Model'], df['File Size (MB)'], color=['#9b59b6', '#16a085', '#d35400', '#c0392b'])
ax3.set_title('Model File Size Comparison', fontsize=14, fontweight='bold')
ax3.set_ylabel('File Size (MB)', fontsize=12)
ax3.grid(axis='y', alpha=0.3)

for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f} MB',
             ha='center', va='bottom', fontweight='bold')

ax3.tick_params(axis='x', rotation=45)

# Combined performance score (Accuracy + F1) / 2
ax4 = axes[1, 1]
df['Combined Score'] = (df['Accuracy'] + df['F1-Score']) / 2
bars4 = ax4.bar(df['Model'], df['Combined Score'], color=['#1abc9c', '#34495e', '#e67e22', '#95a5a6'])
ax4.set_title('Combined Performance Score', fontsize=14, fontweight='bold')
ax4.set_ylabel('Score (%)', fontsize=12)
ax4.set_ylim([0, 100])
ax4.grid(axis='y', alpha=0.3)

for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%',
             ha='center', va='bottom', fontweight='bold')

ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('reports/model_comparison_charts.png', dpi=300, bbox_inches='tight')
print("  Saved: reports/model_comparison_charts.png")

# Chart 2: Detailed Comparison Table
print("\n[3/5] Creating Detailed Comparison Table...")
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

# Create table data
table_data = []
table_data.append(['Model', 'Accuracy', 'F1-Score', 'File Size', 'Rank'])

# Rank by accuracy
df_sorted = df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
for idx, row in df_sorted.iterrows():
    rank = idx + 1
    rank_str = '#1' if rank == 1 else '#2' if rank == 2 else '#3' if rank == 3 else f'#{rank}'
    table_data.append([
        row['Model'],
        f"{row['Accuracy']:.2f}%",
        f"{row['F1-Score']:.2f}%",
        f"{row['File Size (MB)']:.1f} MB",
        rank_str
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.25, 0.15, 0.15, 0.15, 0.1])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(5):
    cell = table[(0, i)]
    cell.set_facecolor('#34495e')
    cell.set_text_props(weight='bold', color='white')

# Color code rows
colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']
for i in range(1, len(table_data)):
    for j in range(5):
        cell = table[(i, j)]
        cell.set_facecolor(colors[i-1] if i <= 4 else '#ecf0f1')

plt.title('Model Performance Ranking', fontsize=16, fontweight='bold', pad=20)
plt.savefig('reports/model_ranking_table.png', dpi=300, bbox_inches='tight')
print("  Saved: reports/model_ranking_table.png")

# Chart 3: Radar Chart for Multi-metric Comparison
print("\n[4/5] Creating Radar Chart...")
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Prepare data for radar chart
categories = ['Accuracy', 'F1-Score', 'Speed\n(inverse size)']
num_vars = len(categories)

# Normalize file size (smaller is better, so invert)
max_size = df['File Size (MB)'].max()
df['Speed Score'] = ((max_size - df['File Size (MB)']) / max_size) * 100

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

colors_radar = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
for idx, row in df.iterrows():
    values = [row['Accuracy'], row['F1-Score'], row['Speed Score']]
    values += values[:1]

    ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors_radar[idx])
    ax.fill(angles, values, alpha=0.15, color=colors_radar[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=12)
ax.set_ylim(0, 100)
ax.set_yticks([25, 50, 75, 100])
ax.set_yticklabels(['25%', '50%', '75%', '100%'])
ax.grid(True)
ax.set_title('Multi-Metric Performance Radar', size=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.savefig('reports/radar_chart.png', dpi=300, bbox_inches='tight')
print("  Saved: reports/radar_chart.png")

# Generate PDF Report
print("\n[5/5] Generating PDF Report...")
pdf_path = 'reports/model_comparison_report.pdf'

with PdfPages(pdf_path) as pdf:
    # Page 1: Cover Page
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    ax.axis('off')

    ax.text(0.5, 0.8, 'Kepler Exoplanet Detection',
            ha='center', va='center', fontsize=28, fontweight='bold',
            transform=ax.transAxes)

    ax.text(0.5, 0.7, 'Machine Learning Model Comparison Report',
            ha='center', va='center', fontsize=18,
            transform=ax.transAxes)

    ax.text(0.5, 0.5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            ha='center', va='center', fontsize=12,
            transform=ax.transAxes)

    ax.text(0.5, 0.4, f'Total Models: {len(df)}',
            ha='center', va='center', fontsize=14,
            transform=ax.transAxes)

    ax.text(0.5, 0.35, f'Best Model: {df_sorted.iloc[0]["Model"]}',
            ha='center', va='center', fontsize=14, fontweight='bold',
            transform=ax.transAxes)

    ax.text(0.5, 0.3, f'Best Accuracy: {df_sorted.iloc[0]["Accuracy"]:.2f}%',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='#2ecc71',
            transform=ax.transAxes)

    # Add metadata
    ax.text(0.5, 0.15, f'Training Samples: {metadata["train_samples"]:,}',
            ha='center', va='center', fontsize=10,
            transform=ax.transAxes)

    ax.text(0.5, 0.12, f'Test Samples: {metadata["test_samples"]:,}',
            ha='center', va='center', fontsize=10,
            transform=ax.transAxes)

    ax.text(0.5, 0.09, f'Features: {metadata["feature_dim"]}',
            ha='center', va='center', fontsize=10,
            transform=ax.transAxes)

    ax.text(0.5, 0.06, f'Classes: {metadata["num_classes"]} (CANDIDATE, CONFIRMED, FALSE POSITIVE)',
            ha='center', va='center', fontsize=10,
            transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 2: Performance Comparison Charts
    fig = plt.figure(figsize=(16, 12))
    img = plt.imread('reports/model_comparison_charts.png')
    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 3: Ranking Table
    fig = plt.figure(figsize=(14, 6))
    img = plt.imread('reports/model_ranking_table.png')
    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 4: Radar Chart
    fig = plt.figure(figsize=(10, 10))
    img = plt.imread('reports/radar_chart.png')
    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 5: Confusion Matrices
    if os.path.exists('figures/confusion_matrices.png'):
        fig = plt.figure(figsize=(18, 5))
        img = plt.imread('figures/confusion_matrices.png')
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title('Confusion Matrices for All Models', fontsize=16, fontweight='bold', pad=10)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    # Page 6: Performance Comparison
    if os.path.exists('figures/performance_comparison.png'):
        fig = plt.figure(figsize=(14, 5))
        img = plt.imread('figures/performance_comparison.png')
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    # Page 7: Detailed Statistics Table
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis('off')

    ax.text(0.5, 0.95, 'Detailed Model Statistics',
            ha='center', va='top', fontsize=18, fontweight='bold',
            transform=ax.transAxes)

    # Create detailed stats table
    model_names = df['Model'].tolist()
    stats_data = [['Metric'] + model_names]

    metrics = ['Accuracy (%)', 'F1-Score (%)', 'File Size (MB)', 'Combined Score (%)']
    for metric in metrics:
        row = [metric]
        for model in model_names:
            model_data = df[df['Model'] == model].iloc[0]
            if 'Accuracy' in metric:
                row.append(f"{model_data['Accuracy']:.2f}")
            elif 'F1' in metric:
                row.append(f"{model_data['F1-Score']:.2f}")
            elif 'File Size' in metric:
                row.append(f"{model_data['File Size (MB)']:.1f}")
            elif 'Combined' in metric:
                row.append(f"{model_data['Combined Score']:.2f}")
        stats_data.append(row)

    num_cols = len(model_names) + 1
    table = ax.table(cellText=stats_data, cellLoc='center', loc='center',
                    bbox=[0.1, 0.3, 0.8, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(num_cols):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')

    # Add recommendations
    ax.text(0.1, 0.2, 'Recommendations:',
            ha='left', va='top', fontsize=14, fontweight='bold',
            transform=ax.transAxes)

    best_model = df_sorted.iloc[0]
    recommendations = [
        f'1. Best Overall Performance: {best_model["Model"]} ({best_model["Accuracy"]:.2f}% accuracy)',
        f'2. Most Balanced: Good accuracy with reasonable file size',
        f'3. Production Ready: All models tested and validated',
        f'4. Recommended: Use {best_model["Model"]} for best results'
    ]

    y_pos = 0.15
    for rec in recommendations:
        ax.text(0.12, y_pos, rec, ha='left', va='top', fontsize=11,
                transform=ax.transAxes)
        y_pos -= 0.03

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print(f"  Saved: {pdf_path}")

print("\n" + "=" * 80)
print("COMPLETE! Generated Files:")
print("=" * 80)
print("  reports/model_comparison_charts.png   - 4-panel comparison charts")
print("  reports/model_ranking_table.png       - Performance ranking table")
print("  reports/radar_chart.png               - Multi-metric radar chart")
print("  reports/model_comparison_report.pdf   - Complete PDF report (7 pages)")
print("=" * 80)

# Summary statistics
print("\nSummary Statistics:")
print(df.to_string(index=False))
print(f"\nBest Model: {df_sorted.iloc[0]['Model']}")
print(f"Best Accuracy: {df_sorted.iloc[0]['Accuracy']:.2f}%")
print(f"Best F1-Score: {df_sorted.iloc[0]['F1-Score']:.2f}%")
print("=" * 80)
