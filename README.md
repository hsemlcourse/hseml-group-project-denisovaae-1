# ML Project — Electricity Load Forecasting

**Студент:** Денисова Алиса Евгеньевна

**Группа:** БИВ231

## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуск](#запуск)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Отчёт](#отчёт)

## Описание задачи

**Задача:** регрессия (прогноз почасового спроса на электроэнергию)

**Датасет:** [Electricity Load Forecasting (Kaggle)](https://www.kaggle.com/datasets/saurabhshahane/electricity-load-forecasting)

**Целевая метрика:** `MAE` (основная), `RMSE` (дополнительная)

## Структура репозитория

```text
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы
├── models                      # Сохранённые модели 
├── notebooks
│   ├── 01_eda.ipynb            # EDA
│   ├── 02_baseline.ipynb       # Baseline-модель
│   └── 03_experiments.ipynb    # Эксперименты и ablation study
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   ├── report.md               # Финальный отчёт
│   ├── data_summary.json
│   └── experiments.csv
├── src
│   ├── preprocessing.py        # Предобработка данных
│   ├── modeling.py             # Обучение и оценка моделей
│   ├── data_tools.py           # утилиты по данным
│   └── settings.py             # пути и константы
├── tests
│   └── test.py                 # Тесты пайплайна
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Запуск

```bash
# 1. Клонировать репозиторий
git clone https://github.com/hsemlcourse/hseml-group-project-denisovaae.git
cd hseml-group-project-denisovaae

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Получить исходный датасет вручную
# Источник: https://www.kaggle.com/datasets/saurabhshahane/electricity-load-forecasting
# Положить файл под именем:
# data/raw/electricity_load.csv
#
# Или запустить скрипт:
python src/download_data.py

# 5. Подготовка данных
python src/preprocessing.py

# 6. Обучение и эксперименты
python src/modeling.py
```

Запуск через Docker:

```bash
docker compose up --build
```

## Данные

- `data/raw/` — исходные файлы
- `data/processed/` — предобработанные данные
- Основной файл для запуска пайплайна: `data/raw/electricity_load.csv`
- Временной split: 70% / 15% / 15% без shuffle

## Результаты

Сводка чистки и признаков: `report/data_summary.json`  
Таблица экспериментов: `report/experiments.csv`  
Графики EDA: `report/images/*.png`  
Лучшая модель: `models/best_model.joblib`  
Метаданные лучшей модели: `report/best_model_info.json`

| Модель              | Val MAE | Test MAE | Примечание      |
| ------------------- | ------- | -------- | --------------- |
| baseline_raw_linear | 114.78  | 137.17   | baseline без FE |
| xgb_small           | 15.61   | 22.35    | лучшая модель   |

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
