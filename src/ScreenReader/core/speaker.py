import pyttsx3
import queue
import time
import threading

class Speaker:
    def __init__(self) -> None:
        self.voice_tasks = queue.Queue()
        
        # 启动工作线程
        self.thread = threading.Thread(target=self.speak_text_block_thread, daemon=True)
        self.thread.start()
        
    def add_sentence(self, sentence: str):
        self.voice_tasks.put(sentence)
    
    def speak_text_block_thread(self):
        # 【关键修改】在子线程内部初始化引擎
        engine = pyttsx3.init()
        
        while True:
            # queue.get() 默认是阻塞的，没有任务时会一直等在这里，不会消耗CPU
            speak_sentence = self.voice_tasks.get()
            
            if speak_sentence:
                print('run once')
                engine.say(speak_sentence)
                engine.runAndWait()
            
            # 通知队列当前任务已处理完毕（良好的队列使用习惯）
            self.voice_tasks.task_done()
                
if __name__ == "__main__":
    sp = Speaker()
    for i in range(99):
        sp.add_sentence(f"阿巴阿巴阿巴 {i}")
        
    time.sleep(100)