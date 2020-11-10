import cv2 
import numpy as np

# 빈 캔버스 만들기
def init() -> np.array: 
    blank = np.zeros([512,512],dtype=np.uint8)
    blank.fill(255)
    blank = cv2.resize(blank,(512,512))
  
    return blank
  