# simple-web-service
Simple web service with html markup
Веб-сервис реализован с помощью фреймворка FastAPI (https://fastapi.tiangolo.com/).
В данном веб-сервисе реализованы следующие задачи (все модели обучены на английском языке):
1. Emotion classification на основе языковой модели BERT
2. Image-text-recognition на основе фреймворка easyocr
3. similar-texts-recognition на основе SentenceTransformers модели 

HTML разметка добавлена для первых двух задач,а для третьей только один POST эндпоинт. 
Чтобы запустить веб-сервис, нужно, находясь в папке с проектом, в консоли написать следующую команду: uvicorn main:app --reload

Далее перейти по ссылке http://127.0.0.1:8000. Там выбираем нужную нам задачу.
![Image alt](https://github.com/ArtyomVorobev/images/blob/main/1.PNG)

Чтобы протестировать третью задачу переходим по ссылке http://127.0.0.1:8000/docs, там выбираем similar-text-recognition
![Image alt](https://github.com/ArtyomVorobev/images/blob/main/2.PNG)

Далее нажимаем кнопку try it out
![Image alt](https://github.com/ArtyomVorobev/images/blob/main/3.PNG)

Вводим нужные тексты в формы для ввода и нажимаем EXECUTE 
![Image alt](https://github.com/ArtyomVorobev/images/blob/main/4.PNG)

Получаем значение косинусного расстояния в Response body 
![Image alt](https://github.com/ArtyomVorobev/images/blob/main/5.PNG)



# Про модели

1. Для первой задачи я выбрал предобученную модель https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion.
Реализация находится в файле emotion_model.py. Сначала текст токенизируется с помощью предобученного для этой же модели токенизатора, потом подается на вход модели, которая выдает вероятности классов ( "sadness","joy","love","anger","fear","surprise").

2.Для второй предобученную на английском языке модель с фреймворка easyocr. На вход модели подается PIL Image. На выходе получаем надписи и координаты, в которых находится конкретный текст. Далее картинка и аутпут модели подается на вход функции get_image() реализованной в ocr_model.py. Функция возвращает картинку, на которой будут написаны тексты, которые предсказала модель. 

!!! Для корректной работы нужно будет изменить директорию сохранения результирующей картинки в main.py (изменить save_path), а также для отображения в браузере изменить в HTML шаблоне значение img src (/templates/itr_result_form.html)!!!

3. Для третьей предобученную модель bert-base-nli-mean-tokens, а в качестве меры схожести текстов выбрал косинусное расстояние, которое считал по формуле 

![Image alt](https://github.com/ArtyomVorobev/images/blob/main/cosine.svg)

Модель Принимает на вход текст, а на выходе отдает эмбеддинг нашего текста размерности 768*1. Далее считается косинусное расстояние между данными векторами  обоих текстов. 


# P.S. 
HTML шаблоны находятся в папке /templates

