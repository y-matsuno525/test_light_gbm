import lightgbm as lgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Irisデータセットの読み込み
data = load_iris()
X = data.data
y = data.target

# トレーニングデータとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# モデルのトレーニング

bst = lgb.train(
    params,
    train_data,
    valid_sets=[test_data],
    num_boost_round=100,
    callbacks=[
        lgb.early_stopping(stopping_rounds=10, verbose=True),
        lgb.log_evaluation(period=100)
    ]
)

y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
y_pred_max = [list(x).index(max(x)) for x in y_pred]

# 精度の計算
accuracy = accuracy_score(y_test, y_pred_max)
print(f'精度: {accuracy * 100:.2f}%')

# 予測結果の表示（前10件）
print("予測ラベル:", y_pred_max[:10])
print("正解ラベル:", y_test[:10])

# 正解数（精度）の表示
print(f'精度: {accuracy * 100:.2f}%')

# 簡単な散布図（2つの特徴量だけを使用）
import matplotlib.pyplot as plt

plt.figure()
plt.title("予測結果（簡易プロット）")
plt.xlabel("特徴量1")
plt.ylabel("特徴量2")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_max, cmap='viridis', edgecolors='k')
plt.grid(True)
plt.show()