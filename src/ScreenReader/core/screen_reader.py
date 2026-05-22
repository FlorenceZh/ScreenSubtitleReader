from .ocr_fither import OCR 
from .speaker import Speaker
from PIL import ImageGrab
from .string_fither import is_similar_str

class ScreenReader:
    def __init__(self) -> None:
        self.ocr = OCR()
        self.speaker = Speaker()
        self.last_time_readed_text = ''
        
    def capture_and_read_once(self):
        im = ImageGrab.grab()
        #获取当前字幕
        ocr_result = self.ocr.OCR_image(im)
        if not ocr_result:
            return
        text = ocr_result.text
        #若字幕改变，读出新字幕的并更新状态
        if not is_similar_str(text,self.last_time_readed_text):
            self.speaker.add_sentence(text)
            self.last_time_readed_text = text
        
    def main_loop(self):
        while True:
            self.capture_and_read_once()
        
    

if __name__ =="__main__":
    sr = ScreenReader()
    sr.main_loop()