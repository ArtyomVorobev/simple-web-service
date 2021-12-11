from PIL import Image,ImageDraw,ImageFont
import easyocr

def get_ocr_reader():
	reader = easyocr.Reader(['en'],gpu=False) 

	return reader

def get_image(img,result):
	img_edit = ImageDraw.Draw(img)
	font = ImageFont.load_default()
	for out in result:
		start_coords = out[0][0]
		text = u' '.join(out[1])
		img_edit.text(start_coords, text, (250,0,0),font=font)

	

	return img 