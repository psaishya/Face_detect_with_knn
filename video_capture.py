import cv2

cap=cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not capture video")
while (cap.isOpened()):
    ret,frame=cap.read()
    if ret == False:
        continue
    cv2.imshow("Frame",frame)
    
    key_pressed=cv2.waitKey(1)
    if key_pressed ==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
    
        