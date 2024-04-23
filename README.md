# Лабораторная система прогнозирования - СиПи Проект
## Описание проекта
   Лабораторная система прогнозирования (ЛСП) - это специализированное клиент-серверное приложение для анализа временных рядов и прогнозирования различных параметров, например, температуры, влажности, давления и т.д. Благодаря отказу от монолитной архитектуры впользу распределения функционала приложения на отдельные модули данное решение имеет преимущество в масштабируемости и гибкости. Другими словами, оно подойдет для широкого спектра применения: от промышленного мониторинга до внедрения в частные метеостанции.
## Реализация
> ![UserDocSiPI](https://private-user-images.githubusercontent.com/121139974/324762514-46ebb371-f756-4007-9e9d-65a62dad675c.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTM4NjI4NTYsIm5iZiI6MTcxMzg2MjU1NiwicGF0aCI6Ii8xMjExMzk5NzQvMzI0NzYyNTE0LTQ2ZWJiMzcxLWY3NTYtNDAwNy05ZTlkLTY1YTYyZGFkNjc1Yy5qcGc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNDIzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDQyM1QwODU1NTZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1mYWVmMGIyOGU2YzllODYxOTcyYzU5MTc4ZGM3NWE3Njg4NzZhMzRlMTRlMzAwNDQ5YmE3YTcxMThhZjY3OTQ3JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.Wqijx1xM7C3fhLcxlAmryUPoP7rDZMf3VC5gDR8TQwo)


  Система состоит из 3 основных компонентов: клиенсткого приложения с графическим интерфейсом, предсказательной модели и управляющего сервера с встроенной базой данных. Модель обучается отдельно от остальных модулей решения, тем не менее в дальнейшем будет предусмотрена возможность настройки внутренних параметров модели со стороны клиента. Графическое приложение и модель обмениваются данным с сервером при помощи протокола HTTP, отправляя полезные данные в формате JSON. Сервер, в свою очередь, преобразует приходящие запросы в запросы к базе данных SQLite и перенаправляет потоки данных между модулями.

  ## Программное обеспечение
  Далее представлена программная реализациия модели обработки и прогнозирования данных на языке Python.
  В проекте были использованы следующие библиотеки:
  ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.model_selection import train_test_split
   from catboost import CatBoostRegressor
   from lightgbm import LGBMRegressor
   import xgboost as xgb
   from sklearn.model_selection import TimeSeriesSplit
   from sklearn.model_selection import cross_val_score
   from sklearn.model_selection import GridSearchCV
   from sklearn.metrics import mean_squared_error
  ```
  Затем выполняются импортирование данных, сортировка и проверка индексов на монотонность:
  ```python
   # Uploading data
   data = pd.read_csv('/kaggle/input/daily-climate-time-series- 
   data/DailyDelhiClimateTrain.csv', index_col = [0],  parse_dates = [0])
   # Sorting indexes
   data.sort_index(ascending = True)
   # Checking whether indexes monotonic or not
   print(data.index.is_monotonic_increasing)
  ```
  Далее происходит подготовка датасета для определения скользящего среднего:
  ```python
  data_shift = data - data.shift()
  data_shift['mean'] = data_shift['temperature'].rolling(15).mean()
  data_shift['std'] = data_shift['temperature'].rolling(15).std()
  fig, ax = plt.subplots(figsize = (12, 9))
  ax.plot(data_shift['mean'])
  ax.plot(data_shift['std'])
  ax.plot(data_shift['temperature'])
  ax.legend(['mean', 'std', 'temp'])
  ax.set_xlabel('Date')
  ax.set_ylabel('Temp')
  ax.set_title("Temp by period");
  ```
  На данном этапе разработки самой высокой точности предсказания добился алгоритм CatBoost, поэтому именно он будет использован в текущей реализации. Кроме того, здесь выводятся параметры градиентного бустинга, при которых модель наиболее точна:
  ```python
 tscv = TimeSeriesSplit(n_splits=5).split(features_train)
 CB_model = CatBoostRegressor()
 cat_params = {
    'depth' : [2, 4, 6],
    'n_estimators' : [100, 250]
               }
  cat_GS = GridSearchCV(CB_model, 
                       cat_params, 
                       scoring = 'neg_root_mean_squared_error', 
                       cv = tscv)
  cat_GS.fit(features_train, target_train, verbose=100)
  %%time
  cat_best_params = cat_GS.best_params_
  predict_cat = cat_GS.predict(features_valid)
  cat_score = cat_GS.best_score_ * -1
  print(cat_best_params)
  print(cat_score)
  ```
  ## Обзор пользовательского интерфейса
  Первоначально пользователю необходимо аутентифицироваться и авторизоваться в системе. Требуется ввести Логин и Пароль после чего пользователь получит доступ к основному функционалу приложения.
  

