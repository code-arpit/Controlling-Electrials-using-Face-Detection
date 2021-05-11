import cv2 as cv
import socket

ip = '192.168.1.118'
port = 4210

# Initialize udp connection
socket_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#initilize camera and face cascades
video = cv.VideoCapture(0)
cascade_face = cv.CascadeClassifier('/home/arpitjain/Desktop/Code/Python/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')

print(f'Sending messages to :{ip, port}')
while True:
    ret, frame = video.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    detections = cascade_face.detectMultiScale(frame_gray, 1.3, 5)
    count = bytes(f'{len(detections)}', 'utf-8')
    if(len(detections) > 0):
        (x,y,w,h) = detections[0]
        frame = cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=3)
        print(f'Number of faces detected = {len(detections)}')
        socket_udp.sendto(count, (ip,port))

    elif (len(detections) == 0):
        print('Number of faces detected = 0')   
        socket_udp.sendto(count, (ip,port))

    cv.imshow('frame', frame)
    if cv.waitKey(10) & 0xFF==ord('q'):
        break
print(f'Messages sent to :{ip, port}')

video.release()
cv.destroyAllWindows()
