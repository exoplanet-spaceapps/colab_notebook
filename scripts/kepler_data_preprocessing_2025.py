# -*- coding: utf-8 -*-
"""
Kepler Exoplanet 資料前處理與切分 - Google Colab 2025
=======================================================

本腳本專為 Google Colab (2025年10月環境) 設計
- Python 3.11
- NumPy 2.0.2
- pandas (最新版本)
- scikit-learn (最新版本)

功能：
1. 載入 features 和 labels 兩個 DataFrame
2. 從 labels 提取 koi_disposition 並轉換為 one-hot 編碼（三個類別）
3. 合併 features 與 one-hot labels
4. 隨機打散數據
5. 以 3:1 比例切分訓練集與測試集
6. 輸出 X_train, y_train, X_test, y_test

作者：Claude AI
日期：2025-10-05
"""

# =============================================================================
# 步驟 1：環境設置與套件檢查
# =============================================================================

print("=" * 80)
print("🚀 Kepler Exoplanet 資料前處理腳本啟動")
print("=" * 80)
print("\n[1/8] 檢查 Python 與套件版本...")

import sys
import warnings
warnings.filterwarnings('ignore')

# 檢查 Python 版本
python_version = sys.version.split()[0]
print(f"  ✓ Python 版本: {python_version}")

# 導入核心套件並檢查版本
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

print(f"  ✓ NumPy 版本: {np.__version__}")
print(f"  ✓ pandas 版本: {pd.__version__}")
print(f"  ✓ scikit-learn 版本: {sklearn.__version__}")

# 設定隨機種子以確保可重現性
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print(f"  ✓ 隨機種子已設定: {RANDOM_STATE}")
print("\n✅ 環境檢查完成！\n")

# =============================================================================
# 步驟 2：數據載入
# =============================================================================

print("=" * 80)
print("[2/8] 載入數據檔案...")
print("=" * 80)

# 如果在 Colab 中，需要上傳檔案
# 偵測是否在 Colab 環境
try:
    import google.colab
    IN_COLAB = True
    print("\n🔍 偵測到 Google Colab 環境")
    print("請上傳以下兩個檔案：")
    print("  1. koi_lightcurve_features_no_label.csv (features)")
    print("  2. q1_q17_dr25_koi.csv (labels)")

    from google.colab import files

    print("\n📤 請上傳 features 檔案...")
    uploaded_features = files.upload()
    features_filename = list(uploaded_features.keys())[0]

    print("\n📤 請上傳 labels 檔案...")
    uploaded_labels = files.upload()
    labels_filename = list(uploaded_labels.keys())[0]

except ImportError:
    IN_COLAB = False
    print("\n🔍 偵測到本地環境")
    # 本地環境使用相對路徑
    features_filename = 'koi_lightcurve_features_no_label.csv'
    labels_filename = 'q1_q17_dr25_koi.csv'

# 載入數據
print(f"\n📥 載入 features: {features_filename}")
features = pd.read_csv(features_filename)

print(f"📥 載入 labels: {labels_filename}")
labels = pd.read_csv(labels_filename)

print(f"\n  ✓ Features 形狀: {features.shape} (rows, columns)")
print(f"  ✓ Labels 形狀: {labels.shape} (rows, columns)")

# 驗證數據行數是否一致
if len(features) != len(labels):
    print(f"\n⚠️ 警告：features ({len(features)}) 與 labels ({len(labels)}) 行數不一致！")
    print("正在對齊索引...")
    # 取交集
    common_indices = features.index.intersection(labels.index)
    features = features.loc[common_indices]
    labels = labels.loc[common_indices]
    print(f"  ✓ 對齊後數據形狀: {features.shape}")

print("\n✅ 數據載入完成！\n")

# =============================================================================
# 步驟 3：數據探索與驗證
# =============================================================================

print("=" * 80)
print("[3/8] 數據探索與驗證...")
print("=" * 80)

# 顯示 labels 的欄位
print("\n📋 Labels DataFrame 欄位：")
print(labels.columns.tolist())

# 檢查 koi_disposition 欄位
if 'koi_disposition' in labels.columns:
    print("\n✓ 找到 'koi_disposition' 欄位")
    disposition_col = 'koi_disposition'
else:
    # 嘗試找到相似的欄位名稱
    possible_cols = [col for col in labels.columns if 'disposition' in col.lower()]
    if possible_cols:
        disposition_col = possible_cols[0]
        print(f"\n⚠️ 未找到 'koi_disposition'，使用 '{disposition_col}' 替代")
    else:
        raise ValueError("❌ 錯誤：找不到 disposition 相關欄位！")

# 提取標籤
y = labels[disposition_col].copy()

# 顯示標籤分佈
print(f"\n📊 標籤 ('{disposition_col}') 分佈：")
print(y.value_counts())
print(f"\n各類別比例：")
print(y.value_counts(normalize=True) * 100)

# 檢查缺失值
missing_labels = y.isnull().sum()
if missing_labels > 0:
    print(f"\n⚠️ 警告：標籤中有 {missing_labels} 個缺失值")
    print("正在移除缺失值的樣本...")
    valid_indices = y.notna()
    y = y[valid_indices]
    features = features[valid_indices]
    print(f"  ✓ 移除後數據形狀: {features.shape}")

print("\n✅ 數據驗證完成！\n")

# =============================================================================
# 步驟 4：One-Hot 編碼
# =============================================================================

print("=" * 80)
print("[4/8] 執行 One-Hot 編碼...")
print("=" * 80)

# 標準化標籤值（轉小寫並移除空格）
y_normalized = y.str.strip().str.upper()

# 顯示唯一值
unique_values = y_normalized.unique()
print(f"\n唯一標籤值: {unique_values}")
print(f"標籤類別數: {len(unique_values)}")

# 將標籤轉換為 one-hot 編碼
# 使用 pandas get_dummies 函數
y_onehot = pd.get_dummies(y_normalized, prefix='label')

print(f"\n  ✓ One-hot 編碼後的形狀: {y_onehot.shape}")
print(f"  ✓ One-hot 欄位: {y_onehot.columns.tolist()}")

# 顯示前幾行
print("\n📋 One-hot 編碼範例（前5行）：")
print(y_onehot.head())

# 統計每個類別的數量
print("\n📊 One-hot 編碼後各類別數量：")
for col in y_onehot.columns:
    count = y_onehot[col].sum()
    print(f"  {col}: {count} ({count/len(y_onehot)*100:.2f}%)")

print("\n✅ One-hot 編碼完成！\n")

# =============================================================================
# 步驟 5：合併 Features 與 Labels
# =============================================================================

print("=" * 80)
print("[5/8] 合併 Features 與 One-hot Labels...")
print("=" * 80)

# 重置索引以確保對齊
features_reset = features.reset_index(drop=True)
y_onehot_reset = y_onehot.reset_index(drop=True)

# 合併數據
combined_data = pd.concat([features_reset, y_onehot_reset], axis=1)

print(f"\n  ✓ 合併後數據形狀: {combined_data.shape}")
print(f"  ✓ 總欄位數: {combined_data.shape[1]}")
print(f"    - Features: {features.shape[1]}")
print(f"    - Labels: {y_onehot.shape[1]}")

# 檢查缺失值
missing_count = combined_data.isnull().sum().sum()
if missing_count > 0:
    print(f"\n⚠️ 警告：合併後數據有 {missing_count} 個缺失值")
    print("正在處理缺失值...")
    # 可選：填充或移除缺失值
    # combined_data = combined_data.dropna()
    # 或者填充為 0
    combined_data = combined_data.fillna(0)
    print("  ✓ 缺失值已處理")

print("\n✅ 數據合併完成！\n")

# =============================================================================
# 步驟 6：隨機打散（Random Shuffle）
# =============================================================================

print("=" * 80)
print("[6/8] 隨機打散數據...")
print("=" * 80)

print(f"\n使用隨機種子: {RANDOM_STATE}")

# 打散數據
shuffled_data = combined_data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print(f"  ✓ 打散前形狀: {combined_data.shape}")
print(f"  ✓ 打散後形狀: {shuffled_data.shape}")

# 顯示打散後的前幾行標籤分佈
label_cols = y_onehot.columns.tolist()
print("\n📋 打散後前10行的標籤分佈：")
print(shuffled_data[label_cols].head(10))

print("\n✅ 數據打散完成！\n")

# =============================================================================
# 步驟 7：訓練/測試集切分 (3:1 比例)
# =============================================================================

print("=" * 80)
print("[7/8] 切分訓練集與測試集 (3:1 比例)...")
print("=" * 80)

# 分離 features 和 labels
X = shuffled_data.drop(columns=label_cols)
y = shuffled_data[label_cols]

print(f"\n  ✓ X (features) 形狀: {X.shape}")
print(f"  ✓ y (labels) 形狀: {y.shape}")

# 使用 train_test_split 切分數據
# test_size=0.25 表示測試集佔 25% (1/4)，訓練集佔 75% (3/4)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y.idxmax(axis=1)  # 保持類別比例
)

print(f"\n📊 訓練集大小:")
print(f"  ✓ X_train: {X_train.shape}")
print(f"  ✓ y_train: {y_train.shape}")

print(f"\n📊 測試集大小:")
print(f"  ✓ X_test: {X_test.shape}")
print(f"  ✓ y_test: {y_test.shape}")

# 驗證比例
total_samples = len(X)
train_samples = len(X_train)
test_samples = len(X_test)

print(f"\n📈 數據切分比例驗證:")
print(f"  總樣本數: {total_samples}")
print(f"  訓練集: {train_samples} ({train_samples/total_samples*100:.2f}%)")
print(f"  測試集: {test_samples} ({test_samples/total_samples*100:.2f}%)")
print(f"  比例: {train_samples/test_samples:.2f}:1")

# 顯示各類別在訓練集和測試集中的分佈
print("\n📊 訓練集標籤分佈:")
for col in label_cols:
    count = y_train[col].sum()
    print(f"  {col}: {count} ({count/len(y_train)*100:.2f}%)")

print("\n📊 測試集標籤分佈:")
for col in label_cols:
    count = y_test[col].sum()
    print(f"  {col}: {count} ({count/len(y_test)*100:.2f}%)")

print("\n✅ 訓練/測試集切分完成！\n")

# =============================================================================
# 步驟 8：視覺化與結果摘要
# =============================================================================

print("=" * 80)
print("[8/8] 生成視覺化與結果摘要...")
print("=" * 80)

# 設定視覺化風格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 創建視覺化圖表
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Kepler Exoplanet 資料前處理結果視覺化',
             fontsize=18, fontweight='bold', y=0.995)

# 圖1: 類別分佈（原始數據）
ax1 = axes[0, 0]
y_normalized.value_counts().plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
ax1.set_title('原始標籤分佈', fontsize=14, fontweight='bold')
ax1.set_xlabel('類別', fontsize=12)
ax1.set_ylabel('數量', fontsize=12)
ax1.tick_params(axis='x', rotation=45)
for i, v in enumerate(y_normalized.value_counts()):
    ax1.text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')

# 圖2: 訓練集 vs 測試集大小
ax2 = axes[0, 1]
sizes = [train_samples, test_samples]
labels_pie = [f'訓練集\n{train_samples}\n({train_samples/total_samples*100:.1f}%)',
              f'測試集\n{test_samples}\n({test_samples/total_samples*100:.1f}%)']
colors_pie = ['#66b3ff', '#ff9999']
ax2.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='',
        startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax2.set_title('訓練集/測試集比例', fontsize=14, fontweight='bold')

# 圖3: 訓練集標籤分佈
ax3 = axes[1, 0]
train_label_counts = y_train.sum()
train_label_counts.plot(kind='bar', ax=ax3, color='lightgreen', edgecolor='black')
ax3.set_title('訓練集標籤分佈', fontsize=14, fontweight='bold')
ax3.set_xlabel('類別', fontsize=12)
ax3.set_ylabel('數量', fontsize=12)
ax3.tick_params(axis='x', rotation=45)
for i, v in enumerate(train_label_counts):
    ax3.text(i, v + 20, str(int(v)), ha='center', va='bottom', fontweight='bold')

# 圖4: 測試集標籤分佈
ax4 = axes[1, 1]
test_label_counts = y_test.sum()
test_label_counts.plot(kind='bar', ax=ax4, color='lightcoral', edgecolor='black')
ax4.set_title('測試集標籤分佈', fontsize=14, fontweight='bold')
ax4.set_xlabel('類別', fontsize=12)
ax4.set_ylabel('數量', fontsize=12)
ax4.tick_params(axis='x', rotation=45)
for i, v in enumerate(test_label_counts):
    ax4.text(i, v + 20, str(int(v)), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('kepler_preprocessing_visualization.png', dpi=300, bbox_inches='tight')
print("\n  ✓ 視覺化圖表已保存: kepler_preprocessing_visualization.png")
plt.show()

# 生成摘要報告
print("\n" + "=" * 80)
print("📄 資料前處理摘要報告")
print("=" * 80)
print(f"""
✅ 處理完成時間: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 數據概覽:
  • 原始數據總數: {total_samples:,} 筆
  • Features 維度: {X.shape[1]:,} 個特徵
  • Labels 類別數: {len(label_cols)} 類

🎯 標籤類別:
""")
for col in label_cols:
    original_count = y_onehot[col].sum()
    print(f"  • {col}: {original_count} ({original_count/len(y_onehot)*100:.2f}%)")

print(f"""
✂️ 數據切分結果:
  • 訓練集: {train_samples:,} 筆 ({train_samples/total_samples*100:.1f}%)
  • 測試集: {test_samples:,} 筆 ({test_samples/total_samples*100:.1f}%)
  • 切分比例: {train_samples/test_samples:.2f}:1

🔢 變數形狀:
  • X_train: {X_train.shape}
  • y_train: {y_train.shape}
  • X_test: {X_test.shape}
  • y_test: {y_test.shape}

🌱 隨機種子: {RANDOM_STATE}
""")

print("=" * 80)
print("\n🎉 資料前處理流程全部完成！")
print("\n💡 下一步可以開始訓練模型：")
print("   • X_train, y_train 用於訓練")
print("   • X_test, y_test 用於評估")
print("=" * 80)

# =============================================================================
# 可選：保存處理後的數據
# =============================================================================

print("\n💾 是否保存處理後的數據？")

# 如果在 Colab 中，提供下載選項
if IN_COLAB:
    save_data = input("輸入 'y' 保存數據並下載，其他鍵跳過: ")
    if save_data.lower() == 'y':
        # 保存為 CSV
        X_train.to_csv('X_train.csv', index=False)
        y_train.to_csv('y_train.csv', index=False)
        X_test.to_csv('X_test.csv', index=False)
        y_test.to_csv('y_test.csv', index=False)

        print("\n  ✓ 數據已保存為 CSV 檔案")

        # 下載檔案
        files.download('X_train.csv')
        files.download('y_train.csv')
        files.download('X_test.csv')
        files.download('y_test.csv')
        files.download('kepler_preprocessing_visualization.png')

        print("  ✓ 檔案下載完成")
else:
    # 本地環境直接保存
    X_train.to_csv('scripts/X_train.csv', index=False)
    y_train.to_csv('scripts/y_train.csv', index=False)
    X_test.to_csv('scripts/X_test.csv', index=False)
    y_test.to_csv('scripts/y_test.csv', index=False)
    print("\n  ✓ 數據已保存到 scripts/ 目錄")

print("\n" + "=" * 80)
print("🚀 腳本執行完畢！感謝使用！")
print("=" * 80)
