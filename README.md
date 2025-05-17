# Аналіз даних коледжного баскетболу (NCAA)


Професійний аналіз даних коледжного баскетболу з використанням Python (Pandas, Seaborn, Scikit-learn).

## 📌 Про проект
**Мета**: Комплексний аналіз ефективності команд NCAA через:
- Розвідувальний аналіз даних (EDA)
- Feature engineering
- Статистичне моделювання
- Інтерактивну візуалізацію

**Датасет**: [College Basketball Dataset](https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset) з Kaggle (сезони 2013-2021)

**Ключові особливості**:
-  Повний pipeline обробки даних
-  Інтерактивні візуалізації
-  Генерація нових ознак
-  Предиктивні моделі

## 🛠 Встановлення та використання
1. Клонуйте репозиторій:
```bash
git clone https://github.com/Alena011/college-basketball-analysis.git
cd college-basketball-analysis
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

Всановіть залежності
pip install -r requirements.txt

Запустіть
jupyter notebook


# Лабораторна робота №2: Прогнозування віку краба

## Опис
Розв'язання задачі регресії для прогнозування віку краба. Використані методи:
- Gradient Boosting
- Support Vector Machine (SVM)
- Bayesian Ridge

## Інструкція

### 1. Встановлення залежностей
```bash
pip install pandas scikit-learn matplotlib jupyter

2. Запуск проєкту
Склонуйте репозиторій:
git clone https://github.com/Alena011/college-basketball-analysis.git

3 Запустіть Jupyter Notebook:
jupyter notebook notebooks/lab_2_crab_age.ipynb


3. Дані
Датасет CrabAgePrediction.csv містить ознаки крабів. Файл розташований у папці data/.

Результати
Графіки та метрики збережено у папці results/crab_age/.

Порівняння моделей:

Gradient Boosting: RMSE = 2.10

SVM: RMSE = 2.21

Bayesian Ridge: RMSE = 2.15


# Лабораторна робота 3: Класифікація грибів

## Опис проекту

Ця лабораторна робота присвячена вирішенню задачі бінарної класифікації грибів на їстівні та отруйні за допомогою різних алгоритмів машинного навчання.


## Використані технології

- Python 3.x
- Бібліотеки:
  - pandas, numpy - обробка даних
  - matplotlib, seaborn - візуалізація
  - scikit-learn - машинне навчання
- Jupyter Notebook - інтерактивне середовище

## Методи класифікації

У роботі використано такі алгоритми:
1. Логістична регресія
2. Метод k-найближчих сусідів (KNN)
3. Дерево прийняття рішень

Для оптимізації моделей застосовано:
- Стандартизацію даних
- Кодування категоріальних ознак
- Підбір гіперпараметрів за допомогою GridSearchCV

## Як запустити проект

1. Клонуйте репозиторій
2. Встановіть необхідні бібліотеки:
pip install -r requirements.txt

3. Запустіть Jupyter Notebook:
jupyter notebook

4. Відкрийте файл `notebooks/lab_3.ipynb`
5. Виконуйте комірки послідовно

## Результати

Найкращі результати показали:
- Логістична регресія: точність ~95%
- Оптимізований KNN: точність ~100%
- Дерево рішень: точність ~100%

Графіки та матриці плутанини збережено у папці `results/images/`.

