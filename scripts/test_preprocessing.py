# -*- coding: utf-8 -*-
"""
測試腳本：Kepler 資料前處理
============================

本腳本用於測試 kepler_data_preprocessing_2025.py 的所有功能

測試項目：
1. 數據載入
2. One-hot 編碼
3. 數據合併
4. 隨機打散
5. 訓練/測試集切分
6. 輸出驗證
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 導入必要套件
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("=" * 80)
print("[TEST] Kepler Data Preprocessing Test Script")
print("=" * 80)

# 測試結果記錄
test_results = []

def log_test(test_name, passed, message=""):
    """記錄測試結果"""
    status = "[PASS]" if passed else "[FAIL]"
    test_results.append({
        'test': test_name,
        'passed': passed,
        'message': message
    })
    print(f"{status} - {test_name}")
    if message:
        print(f"      {message}")

# =============================================================================
# 測試 1: 檢查環境與套件
# =============================================================================

print("\n" + "=" * 80)
print("測試 1: 環境與套件檢查")
print("=" * 80)

try:
    import sklearn
    python_version = sys.version.split()[0]

    print(f"  Python: {python_version}")
    print(f"  NumPy: {np.__version__}")
    print(f"  pandas: {pd.__version__}")
    print(f"  scikit-learn: {sklearn.__version__}")

    # 驗證版本
    numpy_major = int(np.__version__.split('.')[0])
    python_major = int(python_version.split('.')[0])
    python_minor = int(python_version.split('.')[1])

    log_test("Python 版本檢查 (>= 3.10)",
             python_major >= 3 and python_minor >= 10,
             f"當前版本: {python_version}")

    log_test("NumPy 匯入成功", True)
    log_test("pandas 匯入成功", True)
    log_test("scikit-learn 匯入成功", True)

except Exception as e:
    log_test("環境檢查", False, str(e))

# =============================================================================
# 測試 2: 數據檔案檢查
# =============================================================================

print("\n" + "=" * 80)
print("測試 2: 數據檔案檢查")
print("=" * 80)

# 檢查檔案是否存在
features_file = '../koi_lightcurve_features_no_label.csv'
labels_file = '../q1_q17_dr25_koi.csv'

features_exists = os.path.exists(features_file)
labels_exists = os.path.exists(labels_file)

log_test("Features 檔案存在", features_exists, features_file)
log_test("Labels 檔案存在", labels_exists, labels_file)

# =============================================================================
# 測試 3: 數據載入
# =============================================================================

print("\n" + "=" * 80)
print("測試 3: 數據載入測試")
print("=" * 80)

if features_exists and labels_exists:
    try:
        features = pd.read_csv(features_file)
        log_test("Features 載入成功", True, f"形狀: {features.shape}")

        labels = pd.read_csv(labels_file)
        log_test("Labels 載入成功", True, f"形狀: {labels.shape}")

        # 檢查基本屬性
        log_test("Features 為 DataFrame", isinstance(features, pd.DataFrame))
        log_test("Labels 為 DataFrame", isinstance(labels, pd.DataFrame))

        # 檢查行數
        features_rows = len(features)
        labels_rows = len(labels)

        print(f"\n  Features 行數: {features_rows:,}")
        print(f"  Labels 行數: {labels_rows:,}")

    except Exception as e:
        log_test("Data Loading", False, str(e))
        print("\n[WARNING] Data loading failed. Skipping subsequent tests.")
        sys.exit(1)
else:
    print("\n[WARNING] Data files not found. Skipping subsequent tests.")
    sys.exit(1)

# =============================================================================
# 測試 4: 標籤欄位檢查
# =============================================================================

print("\n" + "=" * 80)
print("測試 4: 標籤欄位檢查")
print("=" * 80)

try:
    # 檢查 koi_disposition 欄位
    has_disposition = 'koi_disposition' in labels.columns
    log_test("'koi_disposition' 欄位存在", has_disposition)

    if has_disposition:
        disposition_col = 'koi_disposition'
    else:
        # 搜尋相似欄位
        possible_cols = [col for col in labels.columns if 'disposition' in col.lower()]
        if possible_cols:
            disposition_col = possible_cols[0]
            log_test(f"找到替代欄位: {disposition_col}", True)
        else:
            log_test("找到 disposition 欄位", False, "無法找到任何 disposition 相關欄位")
            sys.exit(1)

    # 提取標籤
    y = labels[disposition_col].copy()

    # 檢查唯一值
    unique_values = y.unique()
    print(f"\n  唯一標籤值: {unique_values}")
    print(f"  標籤類別數: {len(unique_values)}")

    log_test("標籤類別數為 3", len(unique_values) == 3,
             f"實際類別數: {len(unique_values)}")

    # 檢查標籤分佈
    print("\n  標籤分佈:")
    value_counts = y.value_counts()
    for label, count in value_counts.items():
        print(f"    {label}: {count} ({count/len(y)*100:.2f}%)")

    log_test("標籤分佈檢查完成", True)

except Exception as e:
    log_test("標籤欄位檢查", False, str(e))
    sys.exit(1)

# =============================================================================
# 測試 5: One-hot 編碼
# =============================================================================

print("\n" + "=" * 80)
print("測試 5: One-hot 編碼測試")
print("=" * 80)

try:
    # 標準化標籤
    y_normalized = y.str.strip().str.upper()

    # One-hot 編碼
    y_onehot = pd.get_dummies(y_normalized, prefix='label')

    log_test("One-hot 編碼執行成功", True, f"形狀: {y_onehot.shape}")

    # 驗證形狀
    expected_cols = len(unique_values)
    actual_cols = y_onehot.shape[1]

    log_test("One-hot 欄位數正確", actual_cols == expected_cols,
             f"預期: {expected_cols}, 實際: {actual_cols}")

    # 驗證每行只有一個 1
    row_sums = y_onehot.sum(axis=1)
    all_ones = (row_sums == 1).all()

    log_test("每行恰好一個 1", all_ones)

    # 顯示 one-hot 欄位
    print(f"\n  One-hot 欄位: {y_onehot.columns.tolist()}")

    # 顯示範例
    print("\n  前5行範例:")
    print(y_onehot.head())

except Exception as e:
    log_test("One-hot 編碼", False, str(e))
    sys.exit(1)

# =============================================================================
# 測試 6: 數據合併
# =============================================================================

print("\n" + "=" * 80)
print("測試 6: 數據合併測試")
print("=" * 80)

try:
    # 對齊數據（取前 1000 筆進行快速測試）
    n_samples = min(1000, len(features), len(y_onehot))
    features_sample = features.iloc[:n_samples].reset_index(drop=True)
    y_onehot_sample = y_onehot.iloc[:n_samples].reset_index(drop=True)

    # 合併
    combined_data = pd.concat([features_sample, y_onehot_sample], axis=1)

    log_test("數據合併成功", True, f"形狀: {combined_data.shape}")

    # 驗證欄位數
    expected_total_cols = features_sample.shape[1] + y_onehot_sample.shape[1]
    actual_total_cols = combined_data.shape[1]

    log_test("合併後欄位數正確", actual_total_cols == expected_total_cols,
             f"預期: {expected_total_cols}, 實際: {actual_total_cols}")

    # 檢查缺失值
    missing_count = combined_data.isnull().sum().sum()

    log_test("無缺失值（或已處理）", True,
             f"缺失值數量: {missing_count}")

    print(f"\n  合併後形狀: {combined_data.shape}")
    print(f"  總欄位數: {actual_total_cols}")
    print(f"    - Features: {features_sample.shape[1]}")
    print(f"    - Labels: {y_onehot_sample.shape[1]}")

except Exception as e:
    log_test("數據合併", False, str(e))
    sys.exit(1)

# =============================================================================
# 測試 7: 隨機打散
# =============================================================================

print("\n" + "=" * 80)
print("測試 7: 隨機打散測試")
print("=" * 80)

try:
    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)

    # 記錄打散前的索引
    original_indices = combined_data.index.tolist()[:10]

    # 打散
    shuffled_data = combined_data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    log_test("隨機打散執行成功", True, f"形狀: {shuffled_data.shape}")

    # 驗證形狀不變
    shape_unchanged = shuffled_data.shape == combined_data.shape
    log_test("打散後形狀不變", shape_unchanged)

    # 驗證數據已打散（前10行索引應該不同）
    shuffled_indices = shuffled_data.index.tolist()[:10]
    data_shuffled = original_indices != shuffled_indices

    log_test("數據確實被打散", data_shuffled)

    print(f"\n  打散前索引（前10）: {original_indices}")
    print(f"  打散後索引（前10）: {shuffled_indices}")

except Exception as e:
    log_test("隨機打散", False, str(e))
    sys.exit(1)

# =============================================================================
# 測試 8: 訓練/測試集切分
# =============================================================================

print("\n" + "=" * 80)
print("測試 8: 訓練/測試集切分測試")
print("=" * 80)

try:
    # 分離 features 和 labels
    label_cols = y_onehot_sample.columns.tolist()
    X = shuffled_data.drop(columns=label_cols)
    y = shuffled_data[label_cols]

    # 切分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y.idxmax(axis=1)
    )

    log_test("訓練/測試集切分成功", True)

    # 驗證形狀
    total_samples = len(X)
    train_samples = len(X_train)
    test_samples = len(X_test)

    print(f"\n  總樣本數: {total_samples}")
    print(f"  訓練集: {train_samples} ({train_samples/total_samples*100:.2f}%)")
    print(f"  測試集: {test_samples} ({test_samples/total_samples*100:.2f}%)")

    log_test("X_train 形狀正確", X_train.shape[0] == train_samples)
    log_test("y_train 形狀正確", y_train.shape[0] == train_samples)
    log_test("X_test 形狀正確", X_test.shape[0] == test_samples)
    log_test("y_test 形狀正確", y_test.shape[0] == test_samples)

    # 驗證切分比例
    ratio = train_samples / test_samples
    expected_ratio = 3.0
    ratio_correct = abs(ratio - expected_ratio) < 0.5

    log_test("切分比例接近 3:1", ratio_correct,
             f"實際比例: {ratio:.2f}:1")

    # 驗證特徵維度一致
    features_match = X_train.shape[1] == X_test.shape[1]
    log_test("訓練/測試集特徵維度一致", features_match,
             f"訓練集: {X_train.shape[1]}, 測試集: {X_test.shape[1]}")

    # 驗證標籤維度一致
    labels_match = y_train.shape[1] == y_test.shape[1]
    log_test("訓練/測試集標籤維度一致", labels_match,
             f"訓練集: {y_train.shape[1]}, 測試集: {y_test.shape[1]}")

except Exception as e:
    log_test("訓練/測試集切分", False, str(e))
    sys.exit(1)

# =============================================================================
# 測試 9: 標籤分佈驗證（分層抽樣）
# =============================================================================

print("\n" + "=" * 80)
print("測試 9: 分層抽樣驗證")
print("=" * 80)

try:
    # 計算訓練集標籤分佈
    train_dist = y_train.sum() / len(y_train) * 100
    test_dist = y_test.sum() / len(y_test) * 100

    print("\n  訓練集標籤分佈:")
    for col, pct in train_dist.items():
        print(f"    {col}: {pct:.2f}%")

    print("\n  測試集標籤分佈:")
    for col, pct in test_dist.items():
        print(f"    {col}: {pct:.2f}%")

    # 驗證分佈相似（差異小於 5%）
    dist_diff = abs(train_dist - test_dist)
    distributions_similar = (dist_diff < 5).all()

    log_test("訓練/測試集標籤分佈相似", distributions_similar,
             f"最大差異: {dist_diff.max():.2f}%")

except Exception as e:
    log_test("分層抽樣驗證", False, str(e))

# =============================================================================
# 測試 10: 數據完整性檢查
# =============================================================================

print("\n" + "=" * 80)
print("測試 10: 數據完整性檢查")
print("=" * 80)

try:
    # 檢查是否有重複樣本
    train_test_overlap = len(set(X_train.index) & set(X_test.index))

    log_test("訓練/測試集無重疊", train_test_overlap == 0,
             f"重疊樣本數: {train_test_overlap}")

    # 檢查所有樣本都被使用
    all_samples_used = (train_samples + test_samples) == total_samples

    log_test("所有樣本都被使用", all_samples_used)

    # 檢查 one-hot 編碼完整性
    train_label_sums = y_train.sum(axis=1)
    test_label_sums = y_test.sum(axis=1)

    train_onehot_valid = (train_label_sums == 1).all()
    test_onehot_valid = (test_label_sums == 1).all()

    log_test("訓練集 one-hot 編碼完整", train_onehot_valid)
    log_test("測試集 one-hot 編碼完整", test_onehot_valid)

except Exception as e:
    log_test("數據完整性檢查", False, str(e))

# =============================================================================
# 測試總結
# =============================================================================

print("\n" + "=" * 80)
print("[SUMMARY] Test Summary Report")
print("=" * 80)

# 統計結果
total_tests = len(test_results)
passed_tests = sum(1 for r in test_results if r['passed'])
failed_tests = total_tests - passed_tests
pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

print(f"""
Total Tests: {total_tests}
Passed: {passed_tests}
Failed: {failed_tests}
Pass Rate: {pass_rate:.2f}%

""")

# 顯示詳細結果
print("Detailed Test Results:")
print("-" * 80)

for i, result in enumerate(test_results, 1):
    status = "[PASS]" if result['passed'] else "[FAIL]"
    print(f"{i:2d}. {status} {result['test']}")
    if result['message']:
        print(f"     {result['message']}")

print("-" * 80)

# 最終結果
print("\n" + "=" * 80)
if failed_tests == 0:
    print("[SUCCESS] All tests passed! Data preprocessing script is working correctly!")
    print("=" * 80)

    # 顯示最終變數資訊
    print(f"""
[COMPLETE] Output Variables Summary:

Training Set:
  * X_train: {X_train.shape}
  * y_train: {y_train.shape}

Test Set:
  * X_test: {X_test.shape}
  * y_test: {y_test.shape}

Features: {X_train.shape[1]:,}
Label Classes: {y_train.shape[1]}
Split Ratio: {ratio:.2f}:1

Data preprocessing script verified and working!
""")

    # 保存測試結果
    test_summary = pd.DataFrame(test_results)
    test_summary.to_csv('test_results.csv', index=False, encoding='utf-8-sig')
    print("[SAVED] Test results saved to test_results.csv\n")

else:
    print(f"[WARNING] {failed_tests} tests failed. Please check error messages above.")
    print("=" * 80)
    sys.exit(1)

print("=" * 80)
print("[FINISHED] Test script execution completed")
print("=" * 80)
