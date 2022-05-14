import cv2

#img_file = 'Car Image.jpg'
#video = cv2.VideoCapture('Tesla Accident.mp4')
#video = cv2.VideoCapture('tesla dashcam highway.mp4')
video = cv2.VideoCapture('car_and_pedestrian.mp4')

car_tracker = cv2.CascadeClassifier('car_detector.xml')
pedestrian_tracker = cv2.CascadeClassifier('haarcascade_fullbody.xml')

while True:
    (read_successful, frame) = video.read()

    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    

    #detect cars
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    #detect pedestrains
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)



    #draw rectangle around cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    #draw rectangle around pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 225, 255), 2)

    cv2.imshow('self driving car', frame)

    key = cv2.waitKey(1)

    if key==81 or key==113:
        break
video.release()
'''
img = cv2.imread(img_file)

black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

car_tracker = cv2.CascadeClassifier('car_detector.xml')

#detect cars
coordinates = car_tracker.detectMultiScale(black_n_white)

#for a particular car detection
car4 = coordinates[3]
(x, y, w, h) = car4
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)


for (x, y, w, h) in coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imshow('Clever Programmer Car Detector', img)

#dont autoclose (wait here in the code and listen for a key to press)
cv2.waitKey()
'''

print("code completed")