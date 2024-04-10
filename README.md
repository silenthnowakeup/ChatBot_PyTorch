# Простой чат-бот для кофейного магазина с графическим интерфейсом на Python

Этот проект представляет собой простого чат-бота с использованием библиотеки Tkinter для создания графического интерфейса и библиотеки PyTorch для обработки естественного языка.

## Зависимости

Для запуска этого проекта вам понадобятся следующие зависимости:

- Python 3.x
- Tkinter (обычно поставляется с Python, но может потребоваться дополнительная установка)
- PyTorch
- NLTK (Natural Language Toolkit)

Вы можете установить зависимости, выполнив следующую команду:

`pip install -r requirements.txt`


## Запуск

### Локально

Для запуска приложения локально выполните следующую команду:
`python app.py`

Это запустит GUI чат-бота, где вы сможете общаться с ботом.

### В Docker

Для запуска приложения в Docker выполните следующие шаги:

1. Установите Docker на вашем компьютере, если он еще не установлен.
2. Перейдите в корневую директорию проекта.
3. Соберите Docker-образ, выполнив следующую команду: `docker build -t chatbot_image`.
4. Запустите контейнер с помощью следующей команды: `docker run -d --name chatbot_container chatbot_image`

Теперь ваше приложение должно быть запущено в контейнере Docker.




## Структура проекта

- `app.py`: Основной файл приложения, содержащий код для создания графического интерфейса и логику обработки сообщений.
- `chatbot.py`: Модуль с логикой чат-бота, который анализирует введенные сообщения и отвечает на них.
- `model.py`: Модель нейронной сети для обработки естественного языка.
- `nltk_utils.py`: Вспомогательные функции для обработки текста с помощью NLTK.
- `intents.json`: Файл с данными о намерениях для обучения чат-бота.
- `train.py`: Файл обучения нейронной сети.

## Расширение проекта

- Вы можете добавить больше намерений и ответов в файл `intents.json`, чтобы ваш чат-бот был более разнообразным.
- Можно использовать другие методы обработки естественного языка или другие модели нейронных сетей для улучшения функциональности чат-бота.
- Реализовать возможность сохранения истории чатов и другие функции для улучшения пользовательского опыта.



