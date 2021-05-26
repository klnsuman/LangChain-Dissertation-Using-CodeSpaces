# This is a sample Python script.
import cv2
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

'''
Check This
https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
'''

def videoDetect():
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)

    classNames = []

    classFile = 'coco.names'

    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    print(classNames)

    configPath: str = '/Users/klrao/Downloads/SSD-OBJDETECT/ssd_mobilenet_v3_large_coco_2020_01_14/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = '/Users/klrao/Downloads/SSD-OBJDETECT/ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    while True:
        success,img = cap.read()
        #img = cv2.resize(img, None, fx=0.25, fy=0.25,cv2.INTER_AREA)
        classIds, confs, bbox = net.detect(img, confThreshold=0.5)
        print(classIds, bbox)
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(0, 0, 255), thickness=2)
                cv2.putText(img, classNames[classId - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 255, 0), 2)
        cv2.imshow("Image", img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.destroyAllWindows()

def imageDetect(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

    img = cv2.imread("/Users/klrao/Downloads/images/lena.png")
    #cv2.imshow("lena",img)
    classNames = []
    classFile = 'coco.names'


    with open(classFile,'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    print(classNames)

    configPath: str = '/Users/klrao/Downloads/SSD-OBJDETECT/ssd_mobilenet_v3_large_coco_2020_01_14/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = '/Users/klrao/Downloads/SSD-OBJDETECT/ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb'


    net = cv2.dnn_DetectionModel(weightsPath,configPath)
    net.setInputSize(320,320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5,127.5,127.5))
    net.setInputSwapRB(True)

    #model = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')


    classIds , confs , bbox = net.detect(img,confThreshold=0.5)
    print(classIds,bbox)

    for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
        cv2.rectangle(img,box,color=(0,0,255),thickness=2)
        cv2.putText(img,classNames[classId-1],(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.imshow("lena",img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #imageDetect('PyCharm')

    videoDetect()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
