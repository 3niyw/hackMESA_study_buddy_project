import cv2

cap = cv2.VideoCapture(0)  # 0 是你的內建 webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Webcam Feed", frame)

    # 按 q 鍵結束
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

