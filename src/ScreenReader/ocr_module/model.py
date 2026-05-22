from .config import *
from .crnn import CRNNHandle
from .angnet import  AngleNetHandle
from .utils import draw_bbox, crop_rect, sorted_boxes, get_rotate_crop_image
from PIL import Image
import numpy as np
import cv2
import copy
from .dbnet.dbnet_infer import DBNET
import time
import traceback

class  OcrHandle(object):
    def __init__(self):
        self.text_handle = DBNET(model_path)
        self.crnn_handle = CRNNHandle(crnn_model_path)
        if angle_detect:
            self.angle_handle = AngleNetHandle(angle_net_path)


    def crnnRecWithBox(self,im, boxes_list,score_list):
        """
        crnn模型，ocr识别
        @@model,
        @@converter,
        @@im:Array
        @@text_recs:text box
        @@ifIm:是否输出box对应的img

        """
        results = []
        boxes_list = sorted_boxes(np.array(boxes_list))

        line_imgs = []
        for index, (box, score) in enumerate(zip(boxes_list[:angle_detect_num], score_list[:angle_detect_num])):
            tmp_box = copy.deepcopy(box)
            partImg_array = get_rotate_crop_image(im, tmp_box.astype(np.float32))
            partImg = Image.fromarray(partImg_array).convert("RGB")
            line_imgs.append(partImg)

        angle_res = False
        if angle_detect:
            angle_res = self.angle_handle.predict_rbgs(line_imgs)

        count = 1
        for index, (box ,score) in enumerate(zip(boxes_list,score_list)):

            tmp_box = copy.deepcopy(box)
            partImg_array = get_rotate_crop_image(im, tmp_box.astype(np.float32))


            partImg = Image.fromarray(partImg_array).convert("RGB")

            if angle_detect and angle_res:
                partImg = partImg.rotate(180)


            if not is_rgb:
                partImg = partImg.convert('L')

            try:
                if is_rgb:
                    simPred = self.crnn_handle.predict_rbg(partImg)  ##识别的文本
                else:
                    simPred = self.crnn_handle.predict(partImg)  ##识别的文本
            except Exception as e:
                print(traceback.format_exc())
                continue

            if simPred.strip() != '':
                results.append([tmp_box,simPred,score])
                count += 1

        return results


    def text_predict(self,img,short_size):
        boxes_list, score_list = self.text_handle.process(np.asarray(img).astype(np.uint8),short_size=short_size)
        result = self.crnnRecWithBox(np.array(img), boxes_list,score_list)

        return result


if __name__ == "__main__":
    from PIL import ImageGrab


    print("正在初始化 OCR 模型，请稍候...")
    try:
        ocr = OcrHandle()
        print("✅ 模型初始化成功！")
    except Exception as e:
        print(f"❌ 模型初始化失败。\n错误信息: {e}")
        exit(1)

    print("📸 正在截取屏幕...")
    # 使用 ImageGrab 截取全屏
    im = ImageGrab.grab()
    
    # 【关键修正】颜色通道转换：
    # ImageGrab 截取的是 RGB 格式的 PIL 图像，而原项目默认使用 cv2.imread (BGR 格式)。
    # 为了保证模型识别效果不出偏差，这里将其转为 OpenCV 标准的 BGR numpy 数组
    img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    
    print("🔍 开始执行 OCR 识别...")
    start_time = time.time()
    
    # 调用识别接口 (如果是全屏高分辨率截图，short_size 可以保持 960，或者视情况调高)
    results = ocr.text_predict(img, short_size=960)
    
    cost_time = time.time() - start_time
    print(f"✅ 识别完成！耗时: {cost_time:.3f} 秒")
    print("=" * 50)
    
    if not results:
        print("未能识别到任何文字。")
    else:
        print("部分识别结果如下 (最多显示前20条):")
        # 截屏识别出来的文本通常很多，测试时为了清爽只打印前 20 条
        for i, res in enumerate(results[:20]):
            text = res[1] 
            score = res[2]
            print(f"文本: {text.strip()}  |  置信度: {score:.4f}")
            
        if len(results) > 20:
            print(f"... (还有 {len(results) - 20} 条未显示)")
            
    print("=" * 50)
