import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target  # 添加目标列到 DataFrame 中 [^1]

# 查看数据的前三行
print("数据的前三行：")
print(df.head(3))

# 查看数据的后三行
print("数据的后三行：")
print(df.tail(3))

# 使用 seaborn 根据目标类别着色绘制散点图矩阵
sns.pairplot(df, hue='target')
plt.show()
