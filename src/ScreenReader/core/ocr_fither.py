from typing import List

from ScreenReader.ocr_module import OcrHandle
from PIL import Image
import time

class OCRItem:
    """封装每条 OCR 识别结果的对象"""
    def __init__(self, text, box, img_width, img_height, score=0.0):
        self.text = text.strip()  # 文本内容，顺手去除首尾空格
        # 将 numpy array 的坐标转换成普通的 Python 嵌套列表 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        self.box = box.tolist() if hasattr(box, 'tolist') else box
        self.img_width = img_width
        self.img_height = img_height
        self.score = float(score)

    def __repr__(self):
        return f"<OCRItem text='{self.text}', score={self.score:.2f}>"

class OCR:
    def __init__(self) -> None:
        self.ocr = OcrHandle()

    def OCR_image(self,img):
        
        print("🔍 开始执行 OCR 识别...")
        start_time = time.time()
        
        # 调用识别接口 (如果是全屏高分辨率截图，short_size 可以保持 960，或者视情况调高)
        results = self.ocr.text_predict(img, short_size=960)
        
        cost_time = time.time() - start_time
        print(f"✅ 识别完成！耗时: {cost_time:.3f} 秒")
        print("=" * 50)
        
        return OCR.fither_master(img,results)
    
    @staticmethod
    def fither(ocr_obj:List[OCRItem])->OCRItem|None:
        rules = [
            lambda obj: obj.box[0][1] > (obj.img_height *0.75),
            lambda obj: abs((obj.box[0][0] + obj.box[1][0]) / 2 - obj.img_width / 2)  < obj.img_width * 0.1
        ]
        
        for obj in ocr_obj:
            if all(rule(obj) for rule in rules):
                return obj
    
    @staticmethod
    def fither_master(ori_im,result):
        img_width, img_height = ori_im.size

        # 2. 将原始杂乱的数据列表封装成对象列表
        ocr_objects:List[OCRItem] = []
        for item in result:
            # 提取数据并实例化
            obj = OCRItem(
                text=item[1], 
                box=item[0], 
                img_width=img_width, 
                img_height=img_height, 
                score=item[2]
            )
            ocr_objects.append(obj)
        return OCR.fither(ocr_objects)
    

if __name__ == '__main__':
    ocr=OCR()
    im = Image.open('./.test_imgs/t4.png')
    res = ocr.OCR_image(im)
    print(res)