import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend

if not cap.isOpened():
    print("Error: Camera not found or cannot be opened.")
else:
    print("Camera is working. Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        cv2.imshow('Test Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
