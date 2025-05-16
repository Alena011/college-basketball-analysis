import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 1. Загрузка данных
df = pd.read_csv('../data/cleaned_cbb.csv')

# 2. Создаем папку results/task3
os.makedirs('../results/task3', exist_ok=True)

# 3. Создание новых признаков
## Отношение атаки к защите
df['OFF_DEF_RATIO'] = df['ADJOE'] / df['ADJDE']

## Группы по эффективности
df['EFFICIENCY_GROUP'] = pd.cut(df['ADJOE'],
                               bins=[0, 0.3, 0.6, 1.0],
                               labels=['Low', 'Medium', 'High'])

## Логарифм количества игр
df['LOG_GAMES'] = np.log1p(df['G'])

# 4. Визуализация новых признаков
## Гистограмма для OFF_DEF_RATIO
plt.figure(figsize=(10, 6))
sns.histplot(df['OFF_DEF_RATIO'], kde=True, bins=20, color='royalblue')
plt.title('Распределение OFF_DEF_RATIO', fontsize=14)
plt.xlabel('Соотношение атаки/защиты', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.savefig('../results/task3/off_def_ratio_hist.png', bbox_inches='tight', dpi=300)
plt.close()

## Boxplot для групп эффективности
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='EFFICIENCY_GROUP',
    y='W',
    data=df,
    palette='Set2',
    hue='EFFICIENCY_GROUP',
    legend=False
)
plt.title('Распределение побед по группам эффективности', fontsize=14)
plt.xlabel('Группа эффективности', fontsize=12)
plt.ylabel('Количество побед (W)', fontsize=12)
plt.savefig('../results/task3/efficiency_vs_wins.png', bbox_inches='tight', dpi=300)
plt.close()

## Тепловая карта корреляций
plt.figure(figsize=(12, 8))
new_features = ['OFF_DEF_RATIO', 'LOG_GAMES', 'W', 'ADJOE', 'ADJDE']
corr_matrix = df[new_features].corr()
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    center=0,
    fmt=".2f",
    linewidths=0.5,
    annot_kws={'size': 12}
)
plt.title('Корреляция новых признаков с ключевыми показателями', fontsize=14, pad=20)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('../results/task3/new_features_corr.png', bbox_inches='tight', dpi=300)
plt.close()

# 5. Дополнительная визуализация: Violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(
    x='EFFICIENCY_GROUP',
    y='OFF_DEF_RATIO',
    data=df,
    palette='pastel',
    hue='EFFICIENCY_GROUP',
    legend=False
)
plt.title('Распределение OFF_DEF_RATIO по группам эффективности', fontsize=14)
plt.xlabel('Группа эффективности', fontsize=12)
plt.ylabel('Соотношение атаки/защиты', fontsize=12)
plt.savefig('../results/task3/violinplot_efficiency.png', bbox_inches='tight', dpi=300)
plt.close()

# 6. Сохранение данных с новыми признаками
df.to_csv('../data/engineered_cbb.csv', index=False)

print("""
Feature Engineering выполнена! 
""")