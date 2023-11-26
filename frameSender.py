import cv2
import object_socket

s = object_socket.ObjectSenderSocket('127.0.0.1', 5000, print_when_awaiting_receiver=True, print_when_sending_object=True)

video = cv2.VideoCapture('Lane Detection Test Video-01.mp4')

while True:
    ret, frame = video.read()
    s.send_object((ret, frame))

    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
s.close()