import cv2
import object_socket


s = object_socket.ObjectSenderSocket('127.0.0.1', 5000, print_when_awaiting_receiver=True,
                                     print_when_sending_object=True)

video = cv2.VideoCapture(
    r'C:\Users\andre\OneDrive\Desktop\Materii\Anul_III\Semestrul_I\ISSA\Lab3\Lane Detection Test Video-01.mp4')

while True:
    ret, frame = video.read()
    s.send_object((ret, frame))

    if not ret:
        break

    try:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            s.close()
            break
    except ConnectionResetError:
        print('\nAi apasat q imbecilule\n')

video.release()
s.close()


# with open(r'C:\Users\andre\OneDrive\Desktop\Materii\Anul_III\Semestrul_I\ISSA\Conti_project\main.py') as main:
#     exec(main.read())



