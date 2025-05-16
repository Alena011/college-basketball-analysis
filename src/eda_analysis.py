# Импорт библиотек
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Загрузка очищенных данных
df = pd.read_csv('../data/cleaned_cbb.csv')

# 2. Создаем папку results/task2
os.makedirs('../results/task2', exist_ok=True)

# 3. Базовые статистики
numeric_cols = ['ADJOE', 'ADJDE', 'W', 'G']  # Числовые столбцы
stats = df[numeric_cols].describe()

# Добавляем моду
stats.loc['mode'] = df[numeric_cols].mode().iloc[0]

# 4. Сохранение статистик в CSV
stats.to_csv('../results/task2/basic_stats.csv')
print("Базовые статистики сохранены в ../results/task2/basic_stats.csv")

# 5. Визуализация
## Гистограмма для ADJOE
plt.figure(figsize=(10, 6))
sns.histplot(df['ADJOE'], kde=True, bins=20, color='skyblue')
plt.title('Распределение эффективности атаки (ADJOE)')
plt.savefig('../results/task2/adjoie_hist.png', bbox_inches='tight', dpi=300)
plt.close()

## Box-plot для ADJDE
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['ADJDE'], color='lightgreen')
plt.title('Распределение эффективности защиты (ADJDE)')
plt.savefig('../results/task2/adjde_boxplot.png', bbox_inches='tight', dpi=300)
plt.close()

## Scatter plot: ADJOE vs W
plt.figure(figsize=(10, 6))
sns.scatterplot(x='ADJOE', y='W', data=df, alpha=0.6, color='coral')
plt.title('Зависимость побед (W) от эффективности атаки (ADJOE)')
plt.savefig('../results/task2/adjoie_vs_w.png', bbox_inches='tight', dpi=300)
plt.close()

## Тепловая карта корреляций
plt.figure(figsize=(12, 8))
sns.heatmap(
    df[numeric_cols].corr(),
    annot=True,
    cmap='coolwarm',
    center=0,
    fmt=".2f",
    linewidths=0.5
)
plt.title('Корреляция между признаками', pad=20)
plt.savefig('../results/task2/correlation_heatmap.png', bbox_inches='tight', dpi=300)
plt.close()

# 6. Дополнительная визуализация (pair plot)
plt.figure(figsize=(12, 8))
sns.pairplot(df[numeric_cols], corner=True, diag_kind='kde')
plt.suptitle('Парные распределения числовых признаков', y=1.02)
plt.savefig('../results/task2/pair_plot.png', bbox_inches='tight', dpi=300)
plt.close()

print("""
EDA выполнена! 
""")