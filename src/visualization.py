import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Настройка стиля
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# 1. Загрузка данных
df = pd.read_csv('../data/engineered_cbb.csv')

# 2. Создание папки results/task4 (если её нет)
os.makedirs('../results/task4', exist_ok=True)

# ======================
# 3. График 1: Bar chart (сравнение категорий)
# ======================
plt.figure(figsize=(10, 6))
ax1 = sns.barplot(
    x='EFFICIENCY_GROUP',
    y='W',
    data=df,
    errorbar=None,  # Замена устаревшего ci=None
    hue='EFFICIENCY_GROUP',  # Добавлено для palette
    palette="viridis",
    legend=False
)
plt.title('Среднее количество побед по группам эффективности')
plt.xlabel('Группа эффективности')
plt.ylabel('Среднее количество побед (W)')
plt.savefig('../results/task4/barplot_efficiency_vs_wins.png', bbox_inches='tight')
plt.close()

# ======================
# 4. График 2: Heatmap (корреляции)
# ======================
plt.figure(figsize=(12, 8))
corr_matrix = df[['ADJOE', 'ADJDE', 'W', 'OFF_DEF_RATIO']].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title('Корреляции между ключевыми признаками')
plt.savefig('../results/task4/heatmap_correlations.png', bbox_inches='tight')
plt.close()

# ======================
# 5. График 3: Violin plot (распределения)
# ======================
plt.figure(figsize=(10, 6))
sns.violinplot(
    x='EFFICIENCY_GROUP',
    y='OFF_DEF_RATIO',
    data=df,
    palette="Set2",
    hue='EFFICIENCY_GROUP',
    legend=False
)
plt.title('Распределение OFF_DEF_RATIO по группам эффективности')
plt.savefig('../results/task4/violinplot_efficiency.png', bbox_inches='tight')
plt.close()

print("Все графики успешно сохранены в ../results/task4/!")