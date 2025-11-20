from collections import deque
import cv2
import imutils
import numpy as np


#video_path="/run/media/lurtz/Automethic/Reconocimiento/Ajolotes/Tarja 1_20251111125600.avi"
#video_path="/run/media/lurtz/Automethic/Reconocimiento/Ajolotes/Tarja 1_20251111125600.avi"

video_path="E:/Reconocimiento/Ajolotes/Tarja 1_20251111125600.avi"

ACCELERATED = False                # True = análisis rápido (sin mostrar todos los frames)

selectedLower = (10, 10, 10)
selectedUpper = (100, 100, 100)
pts1 = deque(maxlen=32)
pts2 = deque(maxlen=32)
pts3 = deque(maxlen=32)
pts4 = deque(maxlen=32)
counter = 0
movement = 15

(dX1, dY1) = (0, 0)
(dX2, dY2) = (0, 0)
(dX3, dY3) = (0, 0)
(dX4, dY4) = (0, 0)

direction1 = ""
direction2 = ""
direction3 = ""
direction4 = ""


cap = cv2.VideoCapture(video_path)      #abre el video

# === SELECCIONAR ROI (Zona del corazón) ===
ret, frame = cap.read()
if not ret:
    print("No se pudo leer el video.")
    cap.release()
    exit()

frame = imutils.resize(frame, width=620)

#roi_box1 = cv2.selectROI("Selecciona al sujeto", frame, fromCenter=False, showCrosshair=False)
#roi_box2 = cv2.selectROI("Selecciona al sujeto", frame, fromCenter=False, showCrosshair=False)
#roi_box3 = cv2.selectROI("Selecciona al sujeto", frame, fromCenter=False, showCrosshair=False)
#roi_box4 = cv2.selectROI("Selecciona al sujeto", frame, fromCenter=False, showCrosshair=False)
#cv2.destroyWindow("Selecciona a los sujetos")

#x1, y1, w1, h1 = map(int, roi_box1)
#x2, y2, w2, h2 = map(int, roi_box2)
#x3, y3, w3, h3 = map(int, roi_box3)
#x4, y4, w4, h4 = map(int, roi_box4)

h , w  = 150, 180
x1, y1 = 240, 110
x2, y2 = 410, 110
x3, y3 = 250, 270
x4, y4 = 420, 275

#======= obtener info del video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration = total_frames / fps if fps > 0 else 0
print('total: ',duration)

backSub = cv2.createBackgroundSubtractorMOG2()

if not cap.isOpened():
    print("No se pudo abrir el video")
else:
    print("Video abierto correctamente")

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=620)
        roi1  = frame[y1:y1+h, x1:x1+w]
        roi2  = frame[y2:y2+h, x2:x2+w]
        roi3  = frame[y3:y3+h, x3:x3+w]
        roi4  = frame[y4:y4+h, x4:x4+w]
        center = None

        #=============== 1

        fg_mask = backSub.apply(roi1)
        cnts1 = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts1 = imutils.grab_contours(cnts1)

        if len(cnts1) > 0:
            c = max(cnts1, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)

            center = (0,0)
            if int(M["m00"]) != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > 10:
                cv2.circle(roi1, (int(x), int(y)), 10, (0, 255, 255), 2)
                cv2.circle(roi1, center, 5, (0, 0, 255), -1)
                pts1.appendleft(center)

        for i in np.arange(1, len(pts1)):
            if pts1[i - 1] is None or pts1[i] is None:
                continue
            try:
                if counter >= 10 and i == 1 and pts1[-10] is not None:
                    dX1 = pts1[-10][0] - pts1[i][0]
                    dY1 = pts1[-10][1] - pts1[i][1]
                    (dirX, dirY) = ("", "")

                    if np.abs(dX1) > movement:
                        dirX = "D" if np.sign(dX1) == 1 else "I"

                    if np.abs(dY1) > movement:
                        dirY = "A" if np.sign(dY1) == 1 else "B"

                    if dirX != "" and dirY != "":
                        direction1 = "{}-{}".format(dirY, dirX)

                    else:
                        direction1 = dirX if dirX != "" else dirY
            except Exception as e:
                print('1: ',e)

            thickness = int(np.sqrt(32 / float(i + 1)) * 2.5)
            #cv2.line(roi1, pts1[i - 1], pts1[i], (255, 0, 0), thickness)

        cv2.putText(roi1, direction1, (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(roi1, "dx: {}, dy: {}".format(dX1, dY1),(5, roi1.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX,0.35, (0, 0, 255), 1)

        #=============== 2

        fg_mask = backSub.apply(roi2)
        cnts2 = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts2 = imutils.grab_contours(cnts2)

        if len(cnts2) > 0:
            c = max(cnts2, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)

            center = (0,0)
            if int(M["m00"]) != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > 10:
                cv2.circle(roi2, (int(x), int(y)), 10, (0, 255, 255), 2)
                cv2.circle(roi2, center, 5, (0, 0, 255), -1)
                pts2.appendleft(center)

        for i in np.arange(1, len(pts2)):
            if pts2[i - 1] is None or pts2[i] is None:
                continue
            try:
                if counter >= 10 and i == 1 and pts2[-10] is not None:
                    dX2 = pts2[-10][0] - pts2[i][0]
                    dY2 = pts2[-10][1] - pts2[i][1]
                    (dirX, dirY) = ("", "")

                    if np.abs(dX2) > movement:
                        dirX = "D" if np.sign(dX2) == 1 else "I"

                    if np.abs(dY2) > movement:
                        dirY = "A" if np.sign(dY2) == 1 else "B"

                    if dirX != "" and dirY != "":
                        direction2 = "{}-{}".format(dirY, dirX)

                    else:
                        direction2 = dirX if dirX != "" else dirY
            except Exception as e:
                print('2: ',e)

            thickness = int(np.sqrt(32 / float(i + 1)) * 2.5)
            #cv2.line(roi2, pts2[i - 1], pts2[i], (255, 0, 0), thickness)

        cv2.putText(roi2, direction2, (5,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(roi2, "dx: {}, dy: {}".format(dX2, dY2),(5, roi2.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX,0.35, (0, 0, 255), 1)

        #=============== 3

        fg_mask = backSub.apply(roi3)
        cnts3 = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts3 = imutils.grab_contours(cnts3)

        if len(cnts3) > 0:
            c = max(cnts3, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)

            center = (0,0)
            if int(M["m00"]) != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > 10:
                cv2.circle(roi3, (int(x), int(y)), 10, (0, 255, 255), 2)
                cv2.circle(roi3, center, 5, (0, 0, 255), -1)
                pts3.appendleft(center)

        for i in np.arange(1, len(pts3)):
            if pts3[i - 1] is None or pts3[i] is None:
                continue
            try:
                if counter >= 10 and i == 1 and pts3[-10] is not None:
                    dX3 = pts3[-10][0] - pts3[i][0]
                    dY3 = pts3[-10][1] - pts3[i][1]
                    (dirX3, dirY3) = ("", "")

                    if np.abs(dX3) > movement:
                        dirX3 = "D" if np.sign(dX3) == 1 else "I"

                    if np.abs(dY3) > movement:
                        dirY3 = "A" if np.sign(dY3) == 1 else "B"

                    if dirX3 != "" and dirY3 != "":
                        direction3 = "{}-{}".format(dirY3, dirX3)

                    else:
                        direction3 = dirX3 if dirX3 != "" else dirY3
            except Exception as e:
                print('3: ',e)

            thickness = int(np.sqrt(32 / float(i + 1)) * 2.5)
            #cv2.line(roi3, pts3[i - 1], pts3[i], (0, 0, 255), thickness)

        cv2.putText(roi3, direction3, (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(roi3, "dx: {}, dy: {}".format(dX3, dY3),(5, roi3.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX,0.35, (0, 0, 255), 1)

        #=============== 4

        fg_mask = backSub.apply(roi4)
        cnts4 = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts4 = imutils.grab_contours(cnts4)

        if len(cnts4) > 0:
            c = max(cnts4, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)

            center = (0,0)
            if int(M["m00"]) != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > 10:
                cv2.circle(roi4, (int(x), int(y)), 10, (0, 255, 255), 2)
                cv2.circle(roi4, center, 5, (0, 0, 255), -1)
                pts4.appendleft(center)

        for i in np.arange(1, len(pts4)):
            if pts4[i - 1] is None or pts4[i] is None:
                continue
            try:
                if counter >= 10 and i == 1 and pts4[-10] is not None:
                    dX4 = pts4[-10][0] - pts4[i][0]
                    dY4 = pts4[-10][1] - pts4[i][1]
                    (dirX4, dirY4) = ("", "")

                    if np.abs(dX4) > movement:
                        dirX4 = "D" if np.sign(dX4) == 1 else "I"

                    if np.abs(dY4) > movement:
                        dirY4 = "A" if np.sign(dY4) == 1 else "B"

                    if dirX4 != "" and dirY4 != "":
                        direction4 = "{}-{}".format(dirY4, dirX4)

                    else:
                        direction4 = dirX4 if dirX4 != "" else dirY4
            except Exception as e:
                print('4: ',e)

            thickness = int(np.sqrt(32 / float(i + 1)) * 2.5)
            #cv2.line(roi4, pts4[i - 1], pts4[i], (0, 0, 255), thickness)

        cv2.putText(roi4, direction4, (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(roi4, "dx: {}, dy: {}".format(dX4, dY4),(5, roi4.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX,0.35, (0, 0, 255), 1)

        if not ret:
            break
        if not ACCELERATED:
            cv2.imshow("Video", frame)
            #cv2.imshow("1",roi1)
            #cv2.imshow("2",roi2)
            #cv2.imshow("3",roi3)
            #cv2.imshow("4",roi4)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_time = frame_id / fps
        counter+=1
        if frame_id % int(fps * 0.2) == 0:
            #print(f"Tiempo del video: {frame_id / fps:.2f} s")

            line = str(frame_id)+','+direction1+','+direction2+','+direction3+','+direction4+'\n'

            with open("record.csv",'a') as f:
                f.write(line)

cap.release()
cv2.destroyAllWindows()

