# Лабораторная система прогнозирования - СиПи Проект
## Описание проекта
   Лабораторная система прогнозирования (ЛСП) - это специализированное клиент-серверное приложение для анализа временных рядов и прогнозирования различных параметров, например, температуры, влажности, давления и т.д. Благодаря отказу от монолитной архитектуры впользу распределения функционала приложения на отдельные модули данное решение имеет преимущество в масштабируемости и гибкости. Другими словами, оно подойдет для широкого спектра применения: от промышленного мониторинга до внедрения в частные метеостанции.
## Реализация
> ![UserDocSiPI](https://private-user-images.githubusercontent.com/121139974/324762514-46ebb371-f756-4007-9e9d-65a62dad675c.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTM4NjI4NTYsIm5iZiI6MTcxMzg2MjU1NiwicGF0aCI6Ii8xMjExMzk5NzQvMzI0NzYyNTE0LTQ2ZWJiMzcxLWY3NTYtNDAwNy05ZTlkLTY1YTYyZGFkNjc1Yy5qcGc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNDIzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDQyM1QwODU1NTZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1mYWVmMGIyOGU2YzllODYxOTcyYzU5MTc4ZGM3NWE3Njg4NzZhMzRlMTRlMzAwNDQ5YmE3YTcxMThhZjY3OTQ3JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.Wqijx1xM7C3fhLcxlAmryUPoP7rDZMf3VC5gDR8TQwo)


  Система состоит из 3 основных компонентов: клиенсткого приложения с графическим интерфейсом, предсказательной модели и управляющего сервера с встроенной базой данных. Модель обучается отдельно от остальных модулей решения, тем не менее в дальнейшем будет предусмотрена возможность настройки внутренних параметров модели со стороны клиента. Графическое приложение и модель обмениваются данным с сервером при помощи протокола HTTP, отправляя полезные данные в формате JSON. Сервер, в свою очередь, преобразует приходящие запросы в запросы к базе данных SQLite и перенаправляет потоки данных между модулями.

  ## Программное обеспечение
  


   

