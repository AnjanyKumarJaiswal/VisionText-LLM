from PyPDF2 import PdfReader
from PIL import Image
import fitz
import io

class Image_Text_Extractor:
    def __init__(self, pdf_path: str = './test.pdf'):
        self.pdf_path = pdf_path

    def extract_text_and_images(self) -> tuple[list, list]:
        
        pdf_text = []
        pdf_img = []

        with open(self.pdf_path, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            doc = fitz.open(self.pdf_path)

            for page_num in range(len(pdf_reader.pages)):
                # Text Extraction
                page = pdf_reader.pages[page_num]
                pdf_text.append(page.extract_text())

                # Image Extraction
                fitz_page = doc[page_num]
                img_list = fitz_page.get_images(full=True)
                for img_index, img in enumerate(img_list):
                    xref = img[0]
                    base_img = doc.extract_image(xref=xref)
                    img_bytes = base_img['image']
                    image = Image.open(io.BytesIO(img_bytes))
                    pdf_img.append(image)

        return pdf_text, pdf_img 