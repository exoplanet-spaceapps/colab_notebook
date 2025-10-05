# 🚀 Google Colab 快速開始指南

## 最快速的使用方法（3步驟）

### Step 1: 開啟 Google Colab
前往 https://colab.research.google.com/

### Step 2: 複製腳本到 Colab
1. 點擊 "新增筆記本"
2. 在程式碼區塊中，點擊左上角 "檔案" → "上傳筆記本"
3. 或直接複製 `kepler_data_preprocessing_2025.py` 的全部內容到程式碼區塊

### Step 3: 執行！
點擊播放按鈕 ▶️ 或按 `Ctrl+Enter`

---

## 📋 在 Colab 中的完整操作流程

### 1️⃣ 創建新的 Colab Notebook

```python
# 在第一個程式碼區塊中，複製整個 kepler_data_preprocessing_2025.py 檔案內容
# 然後執行這個區塊
```

### 2️⃣ 上傳數據檔案

腳本執行時會自動提示：

```
📤 請上傳 features 檔案...
```

**操作步驟：**
1. 點擊 "Choose Files" 按鈕
2. 選擇 `koi_lightcurve_features_no_label.csv`
3. 等待上傳完成

然後會提示：

```
📤 請上傳 labels 檔案...
```

**操作步驟：**
1. 點擊 "Choose Files" 按鈕
2. 選擇 `q1_q17_dr25_koi.csv`
3. 等待上傳完成

### 3️⃣ 等待處理完成

腳本會依序執行：
- ✅ [1/8] 檢查環境
- ✅ [2/8] 載入數據
- ✅ [3/8] 數據驗證
- ✅ [4/8] One-hot 編碼
- ✅ [5/8] 合併數據
- ✅ [6/8] 隨機打散
- ✅ [7/8] 訓練/測試集切分
- ✅ [8/8] 視覺化

### 4️⃣ 下載結果（可選）

完成後會看到提示：

```
💾 是否保存處理後的數據？
輸入 'y' 保存數據並下載，其他鍵跳過:
```

輸入 `y` 並按 Enter，會自動下載：
- `X_train.csv`
- `y_train.csv`
- `X_test.csv`
- `y_test.csv`
- `kepler_preprocessing_visualization.png`

---

## 🎯 執行後的變數

處理完成後，以下變數可直接在 Colab 中使用：

```python
# 訓練集
X_train  # shape: (約1400, 3197)
y_train  # shape: (約1400, 3) - one-hot encoded

# 測試集
X_test   # shape: (約467, 3197)
y_test   # shape: (約467, 3) - one-hot encoded
```

### 查看變數資訊

```python
# 在新的程式碼區塊中執行

print("訓練集形狀:")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")

print("\n測試集形狀:")
print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")

print("\n標籤欄位:")
print(y_train.columns.tolist())
```

---

## 🔧 常見操作

### 查看前幾筆數據

```python
# 顯示前5筆訓練數據
print("X_train 前5行:")
print(X_train.head())

print("\ny_train 前5行:")
print(y_train.head())
```

### 轉換 one-hot 為單一標籤

```python
# 將 one-hot 編碼轉回單一標籤
y_train_labels = y_train.idxmax(axis=1)  # 回傳欄位名稱
y_test_labels = y_test.idxmax(axis=1)

print("轉換後的標籤範例:")
print(y_train_labels.head())
```

### 檢查類別分佈

```python
# 訓練集類別分佈
print("訓練集標籤分佈:")
for col in y_train.columns:
    count = y_train[col].sum()
    print(f"  {col}: {count} ({count/len(y_train)*100:.2f}%)")

# 測試集類別分佈
print("\n測試集標籤分佈:")
for col in y_test.columns:
    count = y_test[col].sum()
    print(f"  {col}: {count} ({count/len(y_test)*100:.2f}%)")
```

---

## 🧪 快速測試模型

完成數據處理後，可以立即測試模型：

### 範例 1: Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 轉換標籤
y_train_labels = y_train.idxmax(axis=1)
y_test_labels = y_test.idxmax(axis=1)

# 訓練模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train_labels)

# 預測與評估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test_labels, y_pred)

print(f"準確率: {accuracy:.4f}")
print("\n分類報告:")
print(classification_report(y_test_labels, y_pred))
```

### 範例 2: Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 標準化特徵
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 轉換標籤
y_train_labels = y_train.idxmax(axis=1)
y_test_labels = y_test.idxmax(axis=1)

# 訓練模型
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train_labels)

# 評估
accuracy = model.score(X_test_scaled, y_test_labels)
print(f"準確率: {accuracy:.4f}")
```

---

## 💡 進階技巧

### 1. 保存變數供後續使用

```python
# 保存處理後的數據到 Colab 的暫存空間
import pickle

with open('processed_data.pkl', 'wb') as f:
    pickle.dump({
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }, f)

print("✓ 數據已保存")
```

### 2. 重新載入已保存的數據

```python
import pickle

with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

print("✓ 數據已載入")
```

### 3. 連接 Google Drive（持久化存儲）

```python
from google.colab import drive
drive.mount('/content/drive')

# 保存到 Google Drive
X_train.to_csv('/content/drive/MyDrive/X_train.csv', index=False)
y_train.to_csv('/content/drive/MyDrive/y_train.csv', index=False)
X_test.to_csv('/content/drive/MyDrive/X_test.csv', index=False)
y_test.to_csv('/content/drive/MyDrive/y_test.csv', index=False)

print("✓ 數據已保存到 Google Drive")
```

---

## 🐛 疑難排解

### 問題：上傳檔案失敗

**解決方案：**
```python
# 手動上傳檔案的替代方法
from google.colab import files

print("上傳 features 檔案:")
uploaded = files.upload()
features_file = list(uploaded.keys())[0]

print("上傳 labels 檔案:")
uploaded = files.upload()
labels_file = list(uploaded.keys())[0]

# 然後手動載入
import pandas as pd
features = pd.read_csv(features_file)
labels = pd.read_csv(labels_file)
```

### 問題：記憶體不足

**解決方案：**
```python
# 啟用高記憶體模式（Colab Pro 專用）
# 或減少數據特徵

# 選取前1000個特徵
X_train_reduced = X_train.iloc[:, :1000]
X_test_reduced = X_test.iloc[:, :1000]
```

### 問題：執行時間過長

**解決方案：**
```python
# 使用 GPU 加速（如果適用）
# 或減少數據量進行快速測試

# 快速測試：只使用10%的數據
from sklearn.model_selection import train_test_split

X_sample, _, y_sample, _ = train_test_split(
    X_train, y_train,
    train_size=0.1,
    random_state=42
)
```

---

## ✅ 檢查清單

使用前請確認：

- [ ] 已開啟 Google Colab
- [ ] 已複製完整腳本
- [ ] 準備好兩個 CSV 檔案
- [ ] 檔案大小合理（features 約 17MB, labels 約 290KB）
- [ ] 網路連線穩定

---

## 🎓 學習資源

### 延伸閱讀
- [Google Colab 官方文件](https://colab.research.google.com/notebooks/intro.ipynb)
- [pandas 文件](https://pandas.pydata.org/docs/)
- [scikit-learn 文件](https://scikit-learn.org/stable/)

### 相關教學
- One-hot 編碼: https://scikit-learn.org/stable/modules/preprocessing.html
- 訓練/測試集切分: https://scikit-learn.org/stable/modules/cross_validation.html

---

## 🎉 開始使用！

現在您已經了解如何在 Google Colab 中使用這個腳本了。祝您的機器學習專案順利！

**記住：** 完成前處理後，就可以開始訓練各種機器學習模型了！

---

**快速連結：**
- 📄 [完整 README](./README.md)
- 🐍 [主腳本](./kepler_data_preprocessing_2025.py)
- 🌐 [Google Colab](https://colab.research.google.com/)
