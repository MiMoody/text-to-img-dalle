import io
import random 
from PIL import Image
from docarray import Document
from docarray.array.match import MatchArray


SERVER_URL = 'grpcs://dalle-flow.dev.jina.ai'
PROMT = 'running cat'


def save_images(d: Document):
    """ Сохранение картинок, полученных от нейросети """
    
    image_bytes = d.load_uri_to_image_tensor().convert_image_tensor_to_blob().blob
    # open("m.png", "wb").write(image_bytes)
    img = Image.open(io.BytesIO(image_bytes))
    img.save(f"img-{random.random()}.png")
    img.show()
    return d



def main():
    """ Точка входа """
    
    doc :Document = Document(text=PROMT).post(SERVER_URL, parameters={'num_images': 1})
    match_array :MatchArray = doc.matches
    match_array.apply(save_images)

if __name__ == "__main__":
    main()
