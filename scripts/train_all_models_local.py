#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kepler 系外行星三分类检测 - 本地训练脚本
训练 Genesis CNN, XGBoost, RandomForest 三个模型 + 集成模型
"""

import os
import sys
import time
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE

import xgboost as xgb

warnings.filterwarnings('ignore')

# 设置随机种子
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 80)
print("Kepler Exoplanet 3-Class Detection - Local Training")
print("=" * 80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Python 版本: {sys.version.split()[0]}")
print("=" * 80)

# ============================================================================
# 1. 数据加载与预处理
# ============================================================================
print("\n[1/9] 数据加载与预处理...")

# 加载特征数据
print("  读取特征数据...")
features = pd.read_csv('koi_lightcurve_features_no_label.csv')
print(f"  特征数据: {features.shape}")

# 加载标签数据
print("  读取标签数据...")
labels_full = pd.read_csv('q1_q17_dr25_koi.csv')
print(f"  标签数据: {labels_full.shape}")

# 提取 disposition 列
if 'koi_disposition' in labels_full.columns:
    disposition_col = 'koi_disposition'
elif 'disposition' in labels_full.columns:
    disposition_col = 'disposition'
else:
    print("  错误：找不到 disposition 列！")
    sys.exit(1)

labels = labels_full[[disposition_col]].copy()
labels.columns = ['disposition']

# 检查特征数据中的 ID 列
feature_id_col = None
for col in ['kepoi_name', 'kepid', 'koi_id']:
    if col in features.columns:
        feature_id_col = col
        break

if feature_id_col is None:
    print("  警告：特征数据中没有找到 ID 列，使用索引对齐")
    # 假设顺序一致
    features_aligned = features.iloc[:len(labels)]
else:
    # 通过 ID 对齐
    label_id_col = None
    for col in ['kepoi_name', 'kepid', 'koi_id']:
        if col in labels_full.columns:
            label_id_col = col
            break

    if label_id_col:
        merged = pd.merge(
            features,
            labels_full[[label_id_col, disposition_col]],
            on=label_id_col if label_id_col == feature_id_col else None,
            how='inner'
        )
        features_aligned = merged.drop(columns=[label_id_col, disposition_col])
        labels = merged[[disposition_col]]
        labels.columns = ['disposition']
    else:
        features_aligned = features

print(f"  对齐后数据: {features_aligned.shape}")
print(f"  标签分布:\n{labels['disposition'].value_counts()}")

# 移除 ID 列
id_columns = ['kepoi_name', 'kepid', 'koi_id', 'rowid']
for col in id_columns:
    if col in features_aligned.columns:
        features_aligned = features_aligned.drop(columns=[col])

print(f"  最终特征形状: {features_aligned.shape}")

# One-Hot 编码标签
print("\n  应用 One-Hot 编码...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels['disposition'])

# 保存label映射
label_mapping = {i: name for i, name in enumerate(label_encoder.classes_)}
print(f"  Label 映射: {label_mapping}")

# 转换为 one-hot
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse_output=False)
y_onehot = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))

print(f"  One-hot 形状: {y_onehot.shape}")

# ============================================================================
# 2. 数据切分
# ============================================================================
print("\n[2/9] 数据切分（75% train / 25% test）...")

# 合并数据以便一起shuffle
combined = pd.concat([features_aligned.reset_index(drop=True),
                     pd.DataFrame(y_onehot, columns=[f'class_{i}' for i in range(y_onehot.shape[1])])],
                     axis=1)

# 随机打散
combined_shuffled = combined.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# 分离特征和标签
feature_cols = [col for col in combined_shuffled.columns if not col.startswith('class_')]
label_cols = [col for col in combined_shuffled.columns if col.startswith('class_')]

X = combined_shuffled[feature_cols].values
y = combined_shuffled[label_cols].values
y_labels = np.argmax(y, axis=1)  # 用于stratify

# 分层切分
X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y_labels
)

y_train_labels = np.argmax(y_train_raw, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

print(f"  训练集: X={X_train_raw.shape}, y={y_train_raw.shape}")
print(f"  测试集: X={X_test.shape}, y={y_test.shape}")
print(f"  训练集类别分布: {np.bincount(y_train_labels)}")
print(f"  测试集类别分布: {np.bincount(y_test_labels)}")

# ============================================================================
# 3. 特征标准化
# ============================================================================
print("\n[3/9] 特征标准化...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test)

print(f"  标准化后训练集: {X_train_scaled.shape}")
print(f"  标准化后测试集: {X_test_scaled.shape}")

# 保存 scaler
joblib.dump(scaler, 'models/feature_scaler.pkl')
print("  ✓ Scaler 已保存: models/feature_scaler.pkl")

# ============================================================================
# 4. SMOTE 类别平衡
# ============================================================================
print("\n[4/9] 应用 SMOTE 处理类别不平衡...")

smote = SMOTE(random_state=RANDOM_STATE)
X_train, y_train_labels_balanced = smote.fit_resample(X_train_scaled, y_train_labels)

print(f"  SMOTE 后训练集: {X_train.shape}")
print(f"  平衡后类别分布: {np.bincount(y_train_labels_balanced)}")

# 转换回 one-hot
y_train = np.zeros((len(y_train_labels_balanced), y_onehot.shape[1]))
for i, label in enumerate(y_train_labels_balanced):
    y_train[i, label] = 1

print(f"  One-hot 训练标签: {y_train.shape}")

# ============================================================================
# 5. 训练 XGBoost 模型
# ============================================================================
print("\n[5/9] 训练 XGBoost 模型...")

xgb_start = time.time()

# 计算类别权重
class_counts = np.bincount(y_train_labels_balanced)
sample_weights = np.array([1.0 / class_counts[label] for label in y_train_labels_balanced])
sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
    tree_method='hist',
    n_jobs=-1,
    eval_metric='mlogloss'
)

print("  开始训练...")
xgb_model.fit(X_train, y_train_labels_balanced, sample_weight=sample_weights, verbose=False)

xgb_time = time.time() - xgb_start

# 预测
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_pred_xgb_proba = xgb_model.predict_proba(X_test_scaled)

xgb_acc = accuracy_score(y_test_labels, y_pred_xgb)
xgb_f1 = f1_score(y_test_labels, y_pred_xgb, average='weighted')

print(f"  ✓ 训练完成！")
print(f"    训练时间: {xgb_time:.2f} 秒")
print(f"    测试准确率: {xgb_acc:.4f}")
print(f"    F1-Score: {xgb_f1:.4f}")

# 保存模型
xgb_model.save_model('models/xgboost_3class.json')
print("  ✓ 模型已保存: models/xgboost_3class.json")

# ============================================================================
# 6. 训练 Random Forest 模型
# ============================================================================
print("\n[6/9] 训练 Random Forest 模型...")

rf_start = time.time()

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=0
)

print("  开始训练...")
rf_model.fit(X_train, y_train_labels_balanced)

rf_time = time.time() - rf_start

# 预测
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_rf_proba = rf_model.predict_proba(X_test_scaled)

rf_acc = accuracy_score(y_test_labels, y_pred_rf)
rf_f1 = f1_score(y_test_labels, y_pred_rf, average='weighted')

print(f"  ✓ 训练完成！")
print(f"    训练时间: {rf_time:.2f} 秒")
print(f"    测试准确率: {rf_acc:.4f}")
print(f"    F1-Score: {rf_f1:.4f}")

# 保存模型
joblib.dump(rf_model, 'models/random_forest_3class.pkl')
print("  ✓ 模型已保存: models/random_forest_3class.pkl")

# ============================================================================
# 7. 训练 Genesis CNN 模型（TensorFlow）
# ============================================================================
print("\n[7/9] 训练 Genesis CNN 模型...")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    cnn_start = time.time()

    # 计算类别权重
    class_weights = {}
    for i in range(len(class_counts)):
        class_weights[i] = len(y_train_labels_balanced) / (len(class_counts) * class_counts[i])

    print(f"  类别权重: {class_weights}")

    # 构建 CNN 模型
    def build_genesis_cnn(input_dim, num_classes):
        model = models.Sequential([
            # Input reshape
            layers.Reshape((input_dim, 1), input_shape=(input_dim,)),

            # Block 1
            layers.Conv1D(64, kernel_size=50, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv1D(64, kernel_size=50, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=16),
            layers.Dropout(0.25),

            # Block 2
            layers.Conv1D(128, kernel_size=12, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv1D(128, kernel_size=12, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.AveragePooling1D(pool_size=8),
            layers.Dropout(0.3),

            # Classification head
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    cnn_model = build_genesis_cnn(X_train.shape[1], y_train.shape[1])

    print("\n  模型架构:")
    cnn_model.summary()

    # 回调函数
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    print("\n  开始训练...")
    history = cnn_model.fit(
        X_train, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=50,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    cnn_time = time.time() - cnn_start

    # 预测
    y_pred_cnn_proba = cnn_model.predict(X_test_scaled, verbose=0)
    y_pred_cnn = np.argmax(y_pred_cnn_proba, axis=1)

    cnn_acc = accuracy_score(y_test_labels, y_pred_cnn)
    cnn_f1 = f1_score(y_test_labels, y_pred_cnn, average='weighted')

    print(f"\n  ✓ 训练完成！")
    print(f"    训练时间: {cnn_time:.2f} 秒")
    print(f"    测试准确率: {cnn_acc:.4f}")
    print(f"    F1-Score: {cnn_f1:.4f}")
    print(f"    最终 Epoch: {len(history.history['loss'])}")

    # 保存模型
    cnn_model.save('models/genesis_cnn_3class.keras')
    print("  ✓ 模型已保存: models/genesis_cnn_3class.keras")

    cnn_trained = True

except ImportError as e:
    print(f"  ⚠ TensorFlow 未安装或导入失败: {e}")
    print("  跳过 CNN 训练")
    cnn_trained = False
    cnn_acc = 0.0
    cnn_f1 = 0.0
    cnn_time = 0.0

# ============================================================================
# 8. 创建集成模型（Voting Classifier）
# ============================================================================
print("\n[8/9] 创建集成模型（Soft Voting）...")

# 包装 XGBoost 和 RF
from sklearn.base import BaseEstimator, ClassifierMixin

class XGBWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
    def fit(self, X, y):
        return self
    def predict(self, X):
        return self.model.predict(X)
    def predict_proba(self, X):
        return self.model.predict_proba(X)

estimators = [
    ('xgb', XGBWrapper(xgb_model)),
    ('rf', rf_model)
]

ensemble = VotingClassifier(
    estimators=estimators,
    voting='soft',
    weights=[1, 1]  # 相等权重
)

print("  拟合集成模型...")
ensemble.fit(X_train, y_train_labels_balanced)

# 预测
y_pred_ensemble = ensemble.predict(X_test_scaled)
y_pred_ensemble_proba = ensemble.predict_proba(X_test_scaled)

ensemble_acc = accuracy_score(y_test_labels, y_pred_ensemble)
ensemble_f1 = f1_score(y_test_labels, y_pred_ensemble, average='weighted')

print(f"  ✓ 集成模型完成！")
print(f"    测试准确率: {ensemble_acc:.4f}")
print(f"    F1-Score: {ensemble_f1:.4f}")

# 保存集成模型
joblib.dump(ensemble, 'models/ensemble_voting_3class.pkl')
print("  ✓ 模型已保存: models/ensemble_voting_3class.pkl")

# ============================================================================
# 9. 生成评估报告和可视化
# ============================================================================
print("\n[9/9] 生成评估报告...")

# 保存元数据
metadata = {
    "created_at": datetime.now().isoformat(),
    "label_mapping": label_mapping,
    "num_classes": len(label_mapping),
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "feature_dim": X_train.shape[1],
    "models": {
        "xgboost": {
            "accuracy": float(xgb_acc),
            "f1_score": float(xgb_f1),
            "training_time_sec": float(xgb_time),
            "file": "xgboost_3class.json"
        },
        "random_forest": {
            "accuracy": float(rf_acc),
            "f1_score": float(rf_f1),
            "training_time_sec": float(rf_time),
            "file": "random_forest_3class.pkl"
        },
        "ensemble_voting": {
            "accuracy": float(ensemble_acc),
            "f1_score": float(ensemble_f1),
            "file": "ensemble_voting_3class.pkl"
        }
    }
}

if cnn_trained:
    metadata["models"]["genesis_cnn"] = {
        "accuracy": float(cnn_acc),
        "f1_score": float(cnn_f1),
        "training_time_sec": float(cnn_time),
        "file": "genesis_cnn_3class.keras"
    }

with open('models/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("  ✓ 元数据已保存: models/metadata.json")

# 生成混淆矩阵可视化
print("\n  生成混淆矩阵...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Confusion Matrices - 3-Class Kepler Exoplanet Detection', fontsize=16, fontweight='bold')

models_viz = [
    ('XGBoost', y_pred_xgb, 'Blues'),
    ('Random Forest', y_pred_rf, 'Greens'),
    ('Ensemble Voting', y_pred_ensemble, 'Purples')
]

for ax, (name, y_pred, cmap) in zip(axes, models_viz):
    cm = confusion_matrix(y_test_labels, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=[label_mapping[i] for i in range(len(label_mapping))],
                yticklabels=[label_mapping[i] for i in range(len(label_mapping))],
                cbar_kws={'label': 'Count'})
    ax.set_title(name, fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('figures/confusion_matrices_3class.png', dpi=300, bbox_inches='tight')
print("  ✓ 混淆矩阵已保存: figures/confusion_matrices_3class.png")

# 性能比较图
print("  生成性能比较图...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

model_names = ['XGBoost', 'Random Forest', 'Ensemble']
accuracies = [xgb_acc, rf_acc, ensemble_acc]
f1_scores = [xgb_f1, rf_f1, ensemble_f1]
colors = ['#3498db', '#2ecc71', '#9b59b6']

# Accuracy
bars1 = ax1.bar(model_names, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 1.05)
ax1.grid(axis='y', alpha=0.3)
for bar, val in zip(bars1, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# F1-Score
bars2 = ax2.bar(model_names, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('F1-Score (Weighted)', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 1.05)
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars2, f1_scores):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/performance_comparison_3class.png', dpi=300, bbox_inches='tight')
print("  ✓ 性能比较图已保存: figures/performance_comparison_3class.png")

# ============================================================================
# 完成总结
# ============================================================================
print("\n" + "=" * 80)
print("*** TRAINING COMPLETED! ***")
print("=" * 80)

print("\n*** Model Performance Summary ***")
print(f"  XGBoost        - 准确率: {xgb_acc:.4f}, F1: {xgb_f1:.4f}, 时间: {xgb_time:.2f}s")
print(f"  Random Forest  - 准确率: {rf_acc:.4f}, F1: {rf_f1:.4f}, 时间: {rf_time:.2f}s")
if cnn_trained:
    print(f"  Genesis CNN    - 准确率: {cnn_acc:.4f}, F1: {cnn_f1:.4f}, 时间: {cnn_time:.2f}s")
print(f"  Ensemble       - 准确率: {ensemble_acc:.4f}, F1: {ensemble_f1:.4f}")

# 找出最佳模型
best_acc = max(xgb_acc, rf_acc, ensemble_acc)
if best_acc == ensemble_acc:
    best_model_name = "Ensemble Voting"
elif best_acc == xgb_acc:
    best_model_name = "XGBoost"
else:
    best_model_name = "Random Forest"

print(f"\n*** BEST MODEL: {best_model_name} (Accuracy: {best_acc:.4f}) ***")

print("\n*** Output Files ***")
print("  models/")
print("    ├── xgboost_3class.json")
print("    ├── random_forest_3class.pkl")
if cnn_trained:
    print("    ├── genesis_cnn_3class.keras")
print("    ├── ensemble_voting_3class.pkl")
print("    ├── feature_scaler.pkl")
print("    └── metadata.json")
print("  figures/")
print("    ├── confusion_matrices_3class.png")
print("    └── performance_comparison_3class.png")

print("\n" + "=" * 80)
print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# 打印分类报告
print("\n*** Detailed Classification Reports ***")
print("\n【XGBoost】")
print(classification_report(y_test_labels, y_pred_xgb,
                          target_names=[label_mapping[i] for i in range(len(label_mapping))]))

print("\n【Random Forest】")
print(classification_report(y_test_labels, y_pred_rf,
                          target_names=[label_mapping[i] for i in range(len(label_mapping))]))

print("\n【Ensemble Voting】")
print(classification_report(y_test_labels, y_pred_ensemble,
                          target_names=[label_mapping[i] for i in range(len(label_mapping))]))

print("\n*** ALL TASKS COMPLETED! Models saved and ready for inference. ***")
