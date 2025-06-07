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

# Лабораторна робота №4: Класифікація з використанням нейронних мереж


---

## Завдання

### Частина 1 — TensorFlow Playground
- Проведено 10 експериментів з різними параметрами:
  - Змінювались: кількість шарів, кількість нейронів, функції активації (Tanh, ReLU, Sigmoid), learning rate.
  - Збережено результати навчання у вигляді скріншотів (у папці `screenshots/`).
- Всі експерименти детально описані в ноутбуці `lab_4_classification.ipynb`.

### Частина 2 — Класифікація грибів
- Використано датасет `mushrooms.csv` (класифікація грибів на їстівні/отруйні).
- Проведено:
  - Первинний аналіз даних (EDA)
  - Попередню обробку (енкодинг категоріальних ознак)
  - Побудову нейронної мережі (змінювались кількість нейронів, функція активації, learning rate)
  - Навчання моделі з використанням `Keras` та `Tensorflow`
  - Оцінку точності, побудову графіків (loss/accuracy)
  - Побудову `confusion matrix` і класифікаційного звіту (`classification report`)

---

## Інструменти

- Python 3.x
- Jupyter Notebook
- Pandas, Numpy, Seaborn, Matplotlib
- TensorFlow / Keras
- scikit-learn

---

## Результати

- Модель досягла точності близько **...%** (уточнити за ноутбуком).
- Найкращі результати отримано при **[activation=..., layers=..., learning_rate=...]**.

---

## Інструкція для запуску

1. Встановити залежності:
```bash
pip install -r requirements.txt
jupyter notebook notebooks/lab_4_classification.ipynb



# Лабораторная работа №5: Классификация изображений с использованием EuroSAT

## Цель

Построить нейросетевую модель для классификации спутниковых снимков из датасета [EuroSAT](https://www.tensorflow.org/datasets/catalog/eurosat) с использованием:
- Полносвязной нейросети (Fully Connected Neural Network)

## Датасет

- **Название:** EuroSAT
- **Источник:** `tensorflow_datasets`
- **Классы:** 10 типов ландшафтов:
  - AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake
- **Размер изображений:** изначально 64x64, масштабируются до 224x224

## Этапы

1. Загрузка и разбиение датасета (70% train / 30% test)
2. Предобработка изображений (`resize`, нормализация `mobilenet_v2`)
3. Обучение Fully Connected модели:
    - 3 скрытых слоя по 64 нейрона
    - Активация: ReLU
    - Выходной слой: softmax
4. Визуализация:
    - графики точности и потерь
    - примеры предсказаний

##  Результаты

- Оценка точности модели: (указать точность, например: `0.85`)
- Графики обучения:
  - Accuracy vs Epochs
  - Loss vs Epochs
- Визуальные предсказания на 25 тестовых изображениях

##  Зависимости

```bash
pip install tensorflow==2.14 tensorflow-datasets matplotlib numpy
