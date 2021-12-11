from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates

from PIL import Image

import shutil
import easyocr

from emotion_model import get_model,get_emotion,dict_argmax
from ocr_model import get_ocr_reader,get_image
from sim_text import cosine,get_similarity_model


#путь, куда будут сохраняться картинки
save_path = "C:/Users/artyo/Desktop/result.jpg"



app = FastAPI()
templates = Jinja2Templates(directory="templates/")

#объявляем модели
ocr_reader =  get_ocr_reader()
model,tokenizer = get_model()
sentence_model = get_similarity_model()




@app.get('/')
def enter_page(request: Request):
	return templates.TemplateResponse('enter_page.html', context={'request': request})


@app.get("/emotion-detection")
def emotion_form(request: Request):
    result = "Введите ваш текст"
    return templates.TemplateResponse('emotion_form.html', context={'request': request, 'result': result})


@app.post("/emotion-detection")
def emotion_form(request: Request, text: str = Form(...)):
	probs = get_emotion(text,model,tokenizer)
	result = dict_argmax(probs)
	return templates.TemplateResponse('emotion_result_form.html', context={'request': request,'text':text,'result': result, 'probs' : probs})


@app.get("/image-text-recognition")
def ocr_form(request: Request):
    result = "Загрузите вашу картинку"
    return templates.TemplateResponse('itr_form.html', context={'request': request, 'result': result})


@app.post("/image-text-recognition")
def ocr_form(request: Request, file: UploadFile = File(...)):

	with open(f'{file.filename}', "wb+") as f:
		f.write(file.file.read())
		img = Image.open(f)
		img.load() 
		f.close()

	result = ocr_reader.readtext(img)
	text = [out[1] for out in result]
	img = get_image(img,result)

	img.save(save_path)
	return templates.TemplateResponse('itr_result_form.html', context={'request': request, 'result': text})




@app.post("/similar-texts-recognition")
def sim_form(request: Request, first_text: str = Form(...), second_text: str = Form(...)):
    first_encoded = sentence_model.encode(first_text)
    second_encoded = sentence_model.encode(second_text)


    similarity = cosine(first_encoded,second_encoded)
    return similarity