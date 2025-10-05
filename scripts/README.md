# Kepler Exoplanet 資料前處理腳本

## 📋 專案概述

本腳本專為 **Google Colab 2025年10月環境** 設計，提供完整的 Kepler 系外行星數據前處理流程。

### 🎯 功能特點

✅ **完整的資料前處理流程**
- 自動載入 features 與 labels 數據
- One-hot 編碼（三個類別：CONFIRMED, CANDIDATE, FALSE POSITIVE）
- 數據合併、隨機打散與切分
- 3:1 訓練/測試集切分

✅ **環境相容性**
- Python 3.11
- NumPy 2.0.2
- pandas 最新版本
- scikit-learn 最新版本

✅ **自動化視覺化**
- 類別分佈圖表
- 訓練/測試集比例餅圖
- 標籤分佈對比

✅ **Colab 友好**
- 自動偵測 Colab 環境
- 檔案上傳介面
- 結果下載功能

---

## 📁 檔案說明

```
scripts/
├── kepler_data_preprocessing_2025.py    # 主要前處理腳本
└── README.md                            # 說明文件（本檔案）
```

**所需數據檔案：**
- `koi_lightcurve_features_no_label.csv` - 光變曲線特徵 (3,197+ 維度)
- `q1_q17_dr25_koi.csv` - KOI 標籤數據

---

## 🚀 使用方法

### 方法 1: Google Colab 執行（推薦）

1. **上傳腳本到 Colab**
   - 開啟 Google Colab: https://colab.research.google.com/
   - 新增程式碼區塊
   - 複製 `kepler_data_preprocessing_2025.py` 內容貼上

2. **執行腳本**
   ```python
   # 直接執行整個腳本
   %run kepler_data_preprocessing_2025.py
   ```

3. **上傳數據檔案**
   - 腳本會自動提示上傳
   - 先上傳 `koi_lightcurve_features_no_label.csv`
   - 再上傳 `q1_q17_dr25_koi.csv`

4. **下載結果**
   - 腳本執行完畢後會提示是否保存
   - 輸入 `y` 即可下載處理後的數據

### 方法 2: 本地執行

1. **準備環境**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```

2. **準備數據**
   - 將兩個 CSV 檔案放在專案根目錄

3. **執行腳本**
   ```bash
   cd scripts
   python kepler_data_preprocessing_2025.py
   ```

4. **查看結果**
   - 處理後的數據會保存在 `scripts/` 目錄

---

## 📊 輸出說明

### 1. 變數輸出

執行完成後，會在記憶體中產生以下變數：

| 變數名稱 | 形狀 | 說明 |
|---------|------|------|
| `X_train` | (n_train, 3197+) | 訓練集特徵 |
| `y_train` | (n_train, 3) | 訓練集標籤 (one-hot) |
| `X_test` | (n_test, 3197+) | 測試集特徵 |
| `y_test` | (n_test, 3) | 測試集標籤 (one-hot) |

### 2. 檔案輸出

- `X_train.csv` - 訓練集特徵
- `y_train.csv` - 訓練集標籤
- `X_test.csv` - 測試集特徵
- `y_test.csv` - 測試集標籤
- `kepler_preprocessing_visualization.png` - 視覺化圖表

### 3. One-hot 編碼欄位

標籤會轉換為三個二元欄位：

```python
label_CANDIDATE        # 候選行星
label_CONFIRMED        # 已確認行星
label_FALSE POSITIVE   # 偽陽性
```

---

## 🔍 腳本流程

```
[1/8] 檢查環境與套件版本
   ↓
[2/8] 載入數據檔案
   ↓
[3/8] 數據探索與驗證
   ↓
[4/8] One-hot 編碼轉換
   ↓
[5/8] 合併 Features 與 Labels
   ↓
[6/8] 隨機打散數據
   ↓
[7/8] 3:1 訓練/測試集切分
   ↓
[8/8] 視覺化與結果摘要
```

---

## ⚙️ 關鍵參數

### 隨機種子
```python
RANDOM_STATE = 42
```
確保結果可重現

### 切分比例
```python
test_size = 0.25  # 測試集佔 25% (1/4)
                   # 訓練集佔 75% (3/4)
                   # 比例為 3:1
```

### 分層抽樣
```python
stratify = y.idxmax(axis=1)  # 保持各類別比例一致
```

---

## 📈 範例輸出

```
==================================================================================
📄 資料前處理摘要報告
==================================================================================

✅ 處理完成時間: 2025-10-05 20:30:45

📊 數據概覽:
  • 原始數據總數: 1,865 筆
  • Features 維度: 3,197 個特徵
  • Labels 類別數: 3 類

🎯 標籤類別:
  • label_CANDIDATE: 842 (45.15%)
  • label_CONFIRMED: 378 (20.27%)
  • label_FALSE POSITIVE: 645 (34.58%)

✂️ 數據切分結果:
  • 訓練集: 1,398 筆 (75.0%)
  • 測試集: 467 筆 (25.0%)
  • 切分比例: 2.99:1

🔢 變數形狀:
  • X_train: (1398, 3197)
  • y_train: (1398, 3)
  • X_test: (467, 3197)
  • y_test: (467, 3)

🌱 隨機種子: 42
==================================================================================
```

---

## 🛠️ 故障排除

### 問題 1: 找不到 koi_disposition 欄位
**解決方案：** 腳本會自動搜尋包含 "disposition" 的欄位並使用

### 問題 2: Features 與 Labels 行數不一致
**解決方案：** 腳本會自動對齊索引，取交集

### 問題 3: 缺失值處理
**解決方案：** 腳本自動移除標籤缺失的樣本，並以 0 填充特徵缺失值

### 問題 4: NumPy 版本不相容
**解決方案：**
```python
# 在 Colab 中更新
!pip install --upgrade numpy
```

---

## 🎓 技術細節

### One-hot 編碼實現
```python
# 使用 pandas get_dummies
y_onehot = pd.get_dummies(y_normalized, prefix='label')
```

### 分層抽樣實現
```python
# 使用 stratify 參數保持類別比例
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y.idxmax(axis=1)  # 根據最大值所在欄位分層
)
```

### 隨機打散實現
```python
# 使用 pandas sample 方法
shuffled_data = combined_data.sample(
    frac=1,                    # 100% 抽樣
    random_state=RANDOM_STATE  # 固定種子
).reset_index(drop=True)
```

---

## 📚 後續步驟

完成前處理後，可以開始訓練模型：

```python
# 範例：使用 Random Forest
from sklearn.ensemble import RandomForestClassifier

# 轉換 one-hot 為單一標籤（如果需要）
y_train_labels = y_train.idxmax(axis=1)
y_test_labels = y_test.idxmax(axis=1)

# 訓練模型
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train_labels)

# 評估
score = model.score(X_test, y_test_labels)
print(f"準確率: {score:.4f}")
```

---

## 📝 版本資訊

- **版本：** 1.0.0
- **建立日期：** 2025-10-05
- **適用環境：** Google Colab (2025年10月)
- **Python 版本：** 3.11+
- **作者：** Claude AI

---

## 📧 支援

如有問題，請檢查：
1. Python 版本是否為 3.11+
2. 數據檔案格式是否正確
3. 套件版本是否相容

---

## 🎉 完成！

此腳本提供了完整、自動化的資料前處理流程，確保數據可以直接用於機器學習模型訓練。
