# 🪐 Kepler Exoplanet Detection Project

> 使用機器學習技術檢測 Kepler 太空望遠鏡觀測的系外行星

## 📋 專案概述

本專案使用 **NASA Kepler 太空望遠鏡** 的光變曲線數據，透過多種機器學習模型（CNN、XGBoost、Random Forest）來識別系外行星候選者。專案包含完整的資料前處理、模型訓練、評估與視覺化流程。

### ✨ 核心特點

- 🎯 **三分類任務**: CONFIRMED（已確認）/ CANDIDATE（候選）/ FALSE POSITIVE（偽陽性）
- 🧠 **多模型支援**: Genesis CNN、XGBoost、Random Forest
- 📊 **完整流程**: 資料前處理 → 訓練 → 評估 → 視覺化 → 報告生成
- 🚀 **Google Colab 友好**: 完全相容 Colab 環境
- 🔬 **SPARC 方法論**: 整合 Claude-Flow 的 SPARC 開發流程
- ⚡ **GPU 加速**: 支援 GPU 訓練（TensorFlow 與 XGBoost）

---

## 📁 專案結構

```
colab_notebook/
├── 📊 數據檔案
│   ├── koi_lightcurve_features_no_label.csv    # 光變曲線特徵 (17MB, 1,866行, 784欄位)
│   └── q1_q17_dr25_koi.csv                     # KOI 標籤數據 (290KB, 8,054行)
│
├── 🐍 主要程式碼
│   └── kepler_exoplanet_detection_complete_training_(2025).py
│
├── 📁 scripts/ - 資料處理腳本
│   ├── kepler_data_preprocessing_2025.py       # 資料前處理主腳本
│   ├── test_preprocessing.py                   # 測試腳本
│   ├── README.md                                # 腳本詳細說明
│   ├── COLAB_QUICK_START.md                    # Colab 快速指南
│   ├── PROJECT_SUMMARY.md                      # 專案總覽
│   └── TEST_REPORT.md                          # 測試報告
│
├── 🛠️ 配置與文檔
│   ├── CLAUDE.md                                # Claude Code 配置
│   ├── .mcp.json                                # MCP 伺服器配置
│   ├── .gitignore                               # Git 忽略檔案
│   └── README.md                                # 本檔案
│
└── 📂 系統目錄
    ├── .claude/                                 # Claude Code 代理配置
    ├── .claude-flow/                            # Claude Flow 協調系統
    ├── .hive-mind/                              # 群體智能配置
    ├── .swarm/                                  # Swarm 協調
    ├── memory/                                  # 記憶體系統
    └── coordination/                            # 任務編排
```

---

## 🚀 快速開始

### 選項 1: Google Colab 執行（推薦）

#### Step 1: 開啟 Google Colab
前往 [Google Colab](https://colab.research.google.com/)

#### Step 2: 上傳並執行主腳本
```python
# 1. 上傳腳本檔案到 Colab
# 2. 在程式碼區塊中執行

# 如果需要資料前處理，先執行：
%run scripts/kepler_data_preprocessing_2025.py

# 然後執行完整訓練：
%run kepler_exoplanet_detection_complete_training_(2025).py
```

#### Step 3: 上傳數據檔案
腳本會自動提示上傳以下檔案：
1. `koi_lightcurve_features_no_label.csv`
2. `q1_q17_dr25_koi.csv`

#### Step 4: 等待訓練完成
訓練流程包含：
- 🔄 資料載入與前處理
- 🧠 模型訓練（Genesis CNN, XGBoost, Random Forest）
- 📊 性能評估
- 📈 視覺化圖表生成
- 📄 PDF 報告生成

---

### 選項 2: 本地環境執行

#### Step 1: 環境準備

**系統需求**:
- Python 3.11+
- NumPy 2.0.2+
- 16GB+ RAM（建議）
- GPU（可選，加速訓練）

**安裝依賴**:
```bash
pip install numpy pandas scikit-learn tensorflow keras xgboost matplotlib seaborn reportlab imbalanced-learn
```

#### Step 2: 確認數據檔案
確保以下檔案在專案根目錄：
- `koi_lightcurve_features_no_label.csv`
- `q1_q17_dr25_koi.csv`

#### Step 3: 執行資料前處理（可選）
```bash
cd scripts
python kepler_data_preprocessing_2025.py
```

這會產生：
- `X_train.csv`, `y_train.csv` - 訓練集
- `X_test.csv`, `y_test.csv` - 測試集
- `kepler_preprocessing_visualization.png` - 視覺化圖表

#### Step 4: 執行完整訓練
```bash
# 返回專案根目錄
cd ..

# 執行訓練腳本
python kepler_exoplanet_detection_complete_training_(2025).py
```

---

## 📊 資料說明

### 數據來源
- **來源**: NASA Kepler Space Telescope
- **資料集**: Q1-Q17 DR25 KOI Catalog
- **特徵**: 784 個從光變曲線提取的時間序列特徵
- **標籤**: 三類（CONFIRMED, CANDIDATE, FALSE POSITIVE）

### 資料統計

| 項目 | 數值 |
|-----|------|
| 總樣本數 | 1,866 筆（有特徵的） |
| 特徵維度 | 784 個時間序列特徵 |
| 標籤類別 | 3 類 |
| 資料分佈 | CANDIDATE (~45%), FALSE POSITIVE (~35%), CONFIRMED (~20%) |
| 訓練/測試比例 | 3:1 (75% / 25%) |

### 特徵類型
- 統計特徵（均值、中位數、標準差等）
- 時間序列特徵（自相關、偏自相關等）
- 頻域特徵（FFT 係數、CWT 係數等）
- 非線性特徵（熵、對稱性等）

---

## 🧠 模型架構

### 1️⃣ Genesis CNN
**深度學習卷積神經網路**

```python
架構:
- Conv1D (64 filters, kernel=50) + ReLU + MaxPooling
- Conv1D (64 filters, kernel=50) + ReLU + MaxPooling
- Conv1D (64 filters, kernel=12) + ReLU + AveragePooling
- Dropout (0.25)
- Dense (256) + ReLU
- Dense (256) + ReLU
- Dense (2, Softmax)

優化器: Adam
損失函數: Categorical Crossentropy
GPU 加速: ✅
```

### 2️⃣ XGBoost
**梯度提升決策樹**

```python
參數:
- n_estimators: 100
- max_depth: 6
- learning_rate: 0.1
- tree_method: gpu_hist (GPU 加速)
- scale_pos_weight: 自動計算（處理類別不平衡）

GPU 加速: ✅
```

### 3️⃣ Random Forest
**隨機森林分類器**

```python
參數:
- n_estimators: 100
- max_depth: 10
- class_weight: balanced（處理類別不平衡）
- n_jobs: -1（使用所有 CPU 核心）

平行處理: ✅
```

---

## 📈 訓練流程

### 完整流程圖

```
1. 數據載入
   ↓
2. SMOTE 類別平衡 (處理類別不平衡問題)
   ↓
3. 模型訓練（平行訓練三個模型）
   ├─ Genesis CNN (GPU 加速)
   ├─ XGBoost (GPU 加速)
   └─ Random Forest (CPU 平行)
   ↓
4. 模型評估
   ├─ Accuracy, Precision, Recall
   ├─ F1-Score, ROC-AUC
   └─ Confusion Matrix
   ↓
5. 視覺化生成
   ├─ 性能比較圖
   ├─ ROC-AUC 曲線
   └─ 混淆矩陣熱圖
   ↓
6. PDF 報告生成
   └─ 完整的模型比較報告
```

### 處理技術

- ✅ **SMOTE**: 合成少數類樣本，平衡資料集
- ✅ **Early Stopping**: 防止 CNN 過擬合
- ✅ **Stratified Split**: 保持類別比例的切分
- ✅ **GPU 加速**: TensorFlow 與 XGBoost GPU 支援

---

## 📊 輸出結果

### 訓練完成後會生成

#### 1. JSON 結果檔
```
reports/kaggle_comparison_results.json
```
包含：
- 完整的模型指標
- 訓練時間
- 數據集資訊
- 系統環境資訊

#### 2. 視覺化圖表（3張）
```
figures/
├── performance_comparison.png      # 性能比較（4項指標）
├── roc_time_comparison.png         # ROC-AUC & 訓練時間
└── confusion_matrices.png          # 混淆矩陣（3個模型）
```

#### 3. PDF 報告
```
reports/KAGGLE_MODEL_COMPARISON_REPORT.pdf
```
完整的模型比較報告，包含：
- 模型性能表格
- 視覺化圖表
- 元數據資訊

#### 4. 下載壓縮檔（Colab）
```
kaggle_results_complete.zip
```
包含所有結果檔案

---

## 🎯 使用範例

### 範例 1: 資料前處理

```python
# 執行前處理腳本
%run scripts/kepler_data_preprocessing_2025.py

# 查看處理後的變數
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# 查看標籤分佈
print("\n訓練集標籤分佈:")
for col in y_train.columns:
    count = y_train[col].sum()
    print(f"  {col}: {count} ({count/len(y_train)*100:.2f}%)")
```

### 範例 2: 快速測試模型

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 轉換標籤為單一類別
y_train_labels = y_train.idxmax(axis=1)
y_test_labels = y_test.idxmax(axis=1)

# 訓練 Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train_labels)

# 預測與評估
y_pred = model.predict(X_test)
print(classification_report(y_test_labels, y_pred))
```

### 範例 3: 完整訓練流程

```python
# 直接執行完整訓練腳本
%run kepler_exoplanet_detection_complete_training_(2025).py

# 腳本會自動：
# 1. 載入數據
# 2. SMOTE 平衡
# 3. 訓練三個模型
# 4. 評估與比較
# 5. 生成視覺化
# 6. 產生 PDF 報告
# 7. 下載結果
```

---

## 🔧 進階使用

### Claude-Flow 整合

本專案整合了 Claude-Flow 的 SPARC 方法論和多代理協調系統。

#### 啟用 Claude-Flow

```bash
# 確認 MCP 伺服器已配置
cat .mcp.json

# 執行 SPARC 流程
npx claude-flow@alpha sparc run spec-pseudocode "分析 Kepler 數據模式"
npx claude-flow@alpha sparc run architect "設計新的模型架構"
npx claude-flow@alpha sparc tdd "實作特徵工程"
```

#### 使用 Swarm 協調

```bash
# 初始化 Swarm
npx claude-flow@alpha swarm init --topology mesh --agents 5

# 派發任務
npx claude-flow@alpha task orchestrate "優化模型超參數"

# 監控進度
npx claude-flow@alpha swarm monitor
```

### 自訂訓練參數

修改主腳本中的參數：

```python
# 調整 CNN 架構
def build_genesis_cnn():
    model = Sequential([
        Conv1D(128, 50, ...),  # 增加 filters
        # ... 自訂層數
    ])

# 調整 XGBoost 參數
xgb_model = xgb.XGBClassifier(
    n_estimators=200,      # 增加樹的數量
    max_depth=8,           # 增加深度
    learning_rate=0.05,    # 降低學習率
    ...
)
```

---

## 🛠️ 疑難排解

### 問題 1: 記憶體不足

**症狀**: `MemoryError` 或 OOM (Out of Memory)

**解決方案**:
```python
# 減少批次大小
history = genesis_model.fit(
    X_train_cnn, y_train_cat,
    batch_size=16,  # 從 32 降到 16
    ...
)

# 或減少特徵維度
X_train_reduced = X_train.iloc[:, :500]  # 只用前 500 個特徵
```

### 問題 2: GPU 不可用

**症狀**: TensorFlow 或 XGBoost 警告無 GPU

**解決方案**:
```python
# 檢查 GPU
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Colab 啟用 GPU
# Runtime → Change runtime type → Hardware accelerator → GPU
```

### 問題 3: 數據檔案找不到

**症狀**: `FileNotFoundError`

**解決方案**:
```python
# 確認檔案路徑
import os
print("Current directory:", os.getcwd())
print("Files:", os.listdir('.'))

# Colab 手動上傳
from google.colab import files
uploaded = files.upload()
```

### 問題 4: 套件版本不相容

**症狀**: `ImportError` 或版本衝突

**解決方案**:
```bash
# 更新套件到最新版本
pip install --upgrade numpy pandas scikit-learn tensorflow keras xgboost

# 或安裝特定版本
pip install numpy==2.0.2 pandas==2.3.1
```

---

## 📚 延伸閱讀

### 官方文檔
- [NASA Kepler Mission](https://www.nasa.gov/mission_pages/kepler/main/index.html)
- [Kepler Data Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

### 相關論文
- Shallue & Vanderburg (2018): "Identifying Exoplanets with Deep Learning"
- Armstrong et al. (2020): "K2 Exoplanet Detection via Neural Networks"

### 教學資源
- [Google Colab 教學](https://colab.research.google.com/notebooks/intro.ipynb)
- [SMOTE 技術詳解](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [CNN 用於時間序列](https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/)

---

## 🤝 貢獻

歡迎提交 Issue 或 Pull Request！

### 開發流程
1. Fork 本專案
2. 創建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交變更 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 開啟 Pull Request

---

## 📝 授權

本專案遵循 MIT License

---

## 📧 聯絡資訊

- **專案維護**: Claude AI
- **問題回報**: 請使用 GitHub Issues
- **文檔**: 查看 `scripts/` 目錄中的詳細文檔

---

## 🎉 致謝

- NASA Kepler Mission Team
- Kaggle Kepler Dataset Contributors
- TensorFlow & XGBoost Teams
- Open Source Community

---

**最後更新**: 2025-10-05
**版本**: 1.0.0
**狀態**: ✅ 已完成測試，可供使用
