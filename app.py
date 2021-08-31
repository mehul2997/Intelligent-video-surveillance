from flask import Flask, render_template, request, session, url_for, redirect, jsonify,send_from_directory
import pymysql
import time
import cv2
import datetime
import imutils
UPLOAD_FOLDER = 'static/videos/'
ALLOWED_EXTENSIONS = {'mp4', 'pdf', 'png', 'jpg', 'jpeg', 'gif','avi'}
import threading
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'random string'

from imutils.video import VideoStream
# Raspberry Pi camera module (requires picamera package)
# from camera_pi import Camera
import threading
import time
outputFrame = None
lock = threading.Lock()
import tensorflow as tf
import os
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import Flask, render_template, Response

# Below code does the authentication
# part of the code

# Try to load saved client credentials


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB * 2 of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 2)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


def dbConnection():
    try:
        connection = pymysql.connect(host="localhost", user="root", password="root", database="cctv")
        return connection
    except:
        print("Something went wrong in database Connection")


def dbClose():
    try:
        dbConnection().close()
    except:
        print("Something went wrong in Close DB Connection")

# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0)
time.sleep(2.0)
from datetime import date

today = date.today()
print("Today's date:", today)


fourcc = cv2.VideoWriter_fourcc(*'XVID') 

dirname="static//cam_videos//"
ts = time.time()
filetowrite=str(ts)+'.mp4'
completeVideoname=dirname+filetowrite
time.sleep(1)
ts = time.time()
filetowrite=str(ts)+'.mp4'
completeVideoname1=dirname+filetowrite

out = cv2.VideoWriter(completeVideoname, fourcc, 20.0, (640, 480)) 
out1 = cv2.VideoWriter(completeVideoname1, fourcc, 20.0, (640, 480)) 
timeintaerval=5

con=dbConnection()
cursor=con.cursor()



sql = "INSERT INTO videodetails (objectnamefound,imagename,date) VALUES (%s, %s, %s)"
val = ('', completeVideoname,today)
cursor.execute(sql, val)
con.commit()
sql = "INSERT INTO videodetails (objectnamefound,imagename,date) VALUES (%s, %s, %s)"
val = ('', completeVideoname1,today)
cursor.execute(sql, val)
con.commit()



@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/logout')
def logout():
    session.pop('user')
    session.pop('userid')
    return redirect(url_for('index'))


@app.route('/login', methods=["GET","POST"])
def login():
    msg = ''
    if request.method == "POST":
        try:
            session.pop('user',None)
            username = request.form.get("email")
            password = request.form.get("pass")
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT * FROM userdetailscctv WHERE email = %s AND password = %s', (username, password))
            result = cursor.fetchone()
            if result:
                session['user'] = result[1]
                session['userid'] = result[0]
                return redirect(url_for('home'))
            else:
                return redirect(url_for('index'))
        except:
            print("Exception occured at login")
            return redirect(url_for('index'))
        finally:
            dbClose()
    #return redirect(url_for('index'))
    return redirect(url_for('index'))

@app.route('/uploadvideo', methods=["GET","POST"])
def uploadvideo():
    if request.method == "POST":
        #file1=request.form.get("file")
        import cv2
        #print(file1)
        f2= request.files['file']
        print(f2)
        filename_secure = secure_filename(f2.filename)
        f2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename_secure))
        print("print saved")
        filename = os.path.abspath(app.config['UPLOAD_FOLDER']+"//"+filename_secure)
        filename1 = filename_secure
        #print(filename1)
        count=0
        for filename in os.listdir(r'static\videos/'):
            print(filename)
            vidObj = cv2.VideoCapture(r"static\videos/"+str(filename))
            while True:
                success, image = vidObj.read() 
                #print(success)
                # Saves the frames with frame-count 
                if success:
                    cv2.resize(image,(250, 250));
                    #print(count)
                    #print(image)
                    cv2.imwrite(r"images/frame%d.jpg" % count, image) 
                    count += 1
                else:
                    break
            
        violence=[]
        nonviolence=[]
        a=0
        for i in os.listdir('images/'):
            print(i)
            import numpy as np
            from tensorflow.keras.preprocessing import image
            import tensorflow
            test_image = image.load_img("images/"+str(i), target_size = (250, 250))
            test_image = image.img_to_array(test_image)/255
            test_image = np.expand_dims(test_image, axis = 0)
            filename7='new_model12.hp5'
            loaded_model6= tensorflow.keras.models.load_model(filename7)
            result = loaded_model6.predict(test_image).round(3)
            print(result)
            
            pred = np.argmax(result)
            print(result, "--->>>", pred)
        
            if pred == 0:
                nonviolence.append(i)
                
                print('Predicted>>> Image is Not categorized in Violence')
            else:
                a+=1
                violence.append(i)
                import shutil
                shutil.move("Images/"+i, "static\onlyviolence/"+i)
                print('Predicted>>> Image is catgorized Violence')
        count=0
        for i in os.listdir(r'static\onlyviolence/'):
            import cv2
            print(i)
            img = cv2.imread(r'static\onlyviolence/'+str(i)) 
            #print(img)
            
        
            import cv2
            cv2.ocl.setUseOpenCL(False)
            import cv2
            import numpy as np
            
            # Load Yolo
            net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
            classes = []
            with open("coco.names", "r") as f:
                classes = [line.strip() for line in f.readlines()]
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            colors = np.random.uniform(0, 255, size=(len(classes), 3))
            # cv2.imread("room_ser.jpg")
        #img = cv2.resize(img, None, fx=0.4, fy=0.4)
            height, width, channels = img.shape
        
        # Detecting objects
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
            net.setInput(blob)
            outs = net.forward(output_layers)
        
        # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
        
                # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
        
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            print(indexes)
            font = cv2.FONT_HERSHEY_PLAIN
            for ps in range(len(boxes)):
                if ps in indexes:
                    x, y, w, h = boxes[ps]
                    label = str(classes[class_ids[ps]])
                    color = colors[class_ids[ps]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
        
        
            cv2.imshow("Image", img)
            
                #print(success)
                # Saves the frames with frame-count 
            tested = [in1 for in1 in indexes if in1 == 1]
            print(tested)
            a=0
            #print(tested)
            if  len(tested)>0  and  tested[0][0]==1:
                cv2.resize(img,(250, 250));
                count += 1
                print('count',count)
                #print(image)
                cv2.imwrite("static/PeopleImages/frame%d.jpg" % count, img)
                #import shutil
                #shutil.move(r"E:\C Desktop\Company Tasks\Projects Assigned to me\Violence Detection system\Dataset\ui\CCTV\static\violence/"+str(i), "static\PeopleImages/"+str(i))
                #image_names = os.listdir(r'PeopleImages/')
               # return render_template('output1.html',image_names=image_names,a=a)
               
            else:
                 continue
            #vidObj.release() 
        cv2.destroyAllWindows()
        image_names = os.listdir(r'static/PeopleImages/')
        b=os.listdir(r'static/PeopleImages/')
        if len(b)>0:
            import pygame
            pygame.init()
            pygame.mixer.music.load('100.mp3')
            pygame.mixer.music.play()
    
            clock = pygame.time.Clock()
            while pygame.mixer.music.get_busy():
                pygame.event.poll()
                clock.tick(10)
        return render_template('output1.html',image_names=image_names,a=count)
    return render_template("uploadvideo.html")
        
@app.route('/onlycheck',methods=["GET","POST"])
def onlycheck():
    count=0
    for i in os.listdir(r'static\onlyviolence/'):
        import cv2
        print(i)
        img = cv2.imread(r'static\onlyviolence/'+str(i)) 
        #print(img)
        
    
        import cv2
        cv2.ocl.setUseOpenCL(False)
        import cv2
        import numpy as np
        
        # Load Yolo
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        classes = []
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        # cv2.imread("room_ser.jpg")
    #img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape
    
    # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
        net.setInput(blob)
        outs = net.forward(output_layers)
    
    # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
            # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
    
            # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
    
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        for ps in range(len(boxes)):
            if ps in indexes:
                x, y, w, h = boxes[ps]
                label = str(classes[class_ids[ps]])
                color = colors[class_ids[ps]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
                if label=='person':
                    cv2.resize(img,(250, 250));
                    count += 1
                    cv2.imwrite("static/PeopleImages/frame%d.jpg" % count, img)
                else:
                    pass
    
    
        cv2.imshow("Image", img)
        
            #print(success)
            # Saves the frames with frame-count 
       
        #vidObj.release() 
    cv2.destroyAllWindows()
    image_names = os.listdir(r'static/PeopleImages/')
    b=os.listdir(r'static/PeopleImages/')
    if len(b)>0:
        import pygame
        pygame.init()
        pygame.mixer.music.load('100.mp3')
        pygame.mixer.music.play()

        clock = pygame.time.Clock()
        while pygame.mixer.music.get_busy():
            pygame.event.poll()
            clock.tick(10)
    return render_template('output1.html',image_names=image_names,a=count)
    
        
            
            
        
        
        #return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)
            #return send_from_directory('', path, as_attachment=True)
            
        
    return render_template("uploadvideo.html")
@app.route('/register', methods=["GET","POST"])
def register():
    if request.method == "POST":
        try:
            name = request.form.get("name")
            email = request.form.get("email")
            mobile = request.form.get("mobile")
            password = request.form.get("pass")
            con = dbConnection()
            cursor = con.cursor()
            sql = "INSERT INTO userdetailscctv (name, email, mobile, password) VALUES (%s, %s, %s, %s)"
            val = (name, email, mobile, password)
            cursor.execute(sql, val)
            con.commit()
            return redirect(url_for('index'))
        except:
            print("Exception occured at login")
            return redirect(url_for('index'))
        finally:
            dbClose()
    return redirect(url_for('index'))

@app.route('/recognitionofperson')
def recognitionofperson():
    import cv2
    
    #cap = cv2.VideoCapture(0)
    nameofuser=''
    a_dict = {}
    fps =10# int(vs.get(cv2.CAP_PROP_FPS))
    count=0
    #con = dbConnection()
    #cursor = con.cursor()
    nonviolence=[]
    violence=[]
    i=10
    frame = vs.read()
    ret=True
    if not ret:
        os.exit()
    out1.write(frame) 
    img =frame# cv2.imread("room_ser.jpg")
    #img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

# D

# Showing informations on the screen
   
    #time.sleep(5

    
    
    normalframe=frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    text = "Unoccupied"
    
    
    
    
    for i in range(10):
        dirname = 'static/start_video/'
        ts = time.time()
        filetowrite=str(ts)+'.png'
        completefilename=dirname+filetowrite
        cv2.imwrite(completefilename,img)
        #sql = "INSERT INTO objectdetails1 (objectnamefound,imagename,date) VALUES (%s, %s, %s)"
        #val = ('', completefilename,today)
        #cursor.execute(sql, val)
        #con.commit()
        print('successfully written  frame')
        print("in recognition Person")
    a=0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('done')
        os.exit()
    for i in os.listdir('static/start_video/'):
        
        print(i)
        import numpy as np
        from tensorflow.keras.preprocessing import image
        import tensorflow
        test_image = image.load_img("static\start_video/"+str(i), target_size = (250, 250))
        test_image = image.img_to_array(test_image)/255
        test_image = np.expand_dims(test_image, axis = 0)
        filename7='new_model12.hp5'
        loaded_model6= tensorflow.keras.models.load_model(filename7)
        result = loaded_model6.predict(test_image).round(3)
        print(result)
        
        pred = np.argmax(result)
        print(result, "--->>>", pred)
        
        if pred == 0:
            nonviolence.append(i)
            
            print('Predicted>>> Image is Not categorized in Violence')
        else:
            a+=1
            violence.append(i)
            import shutil
            shutil.move("static\start_video/"+i, "static//video_images/"+i)
            print('Predicted>>> Image is catgorized Violence')
    for i in os.listdir(r'static\video_images/'):
        
        import cv2
        print(i)
        img = cv2.imread(r'static\video_images/'+str(i)) 
        #print(img)
        
    
        import cv2
        cv2.ocl.setUseOpenCL(False)
        import cv2
        import numpy as np
        
        # Load Yolo
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        classes = []
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        # cv2.imread("room_ser.jpg")
    #img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape
    
    # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
        net.setInput(blob)
        outs = net.forward(output_layers)
    
    # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
            # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
    
            # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
    
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        for ps in range(len(boxes)):
            if ps in indexes:
                x, y, w, h = boxes[ps]
                label = str(classes[class_ids[ps]])
                color = colors[class_ids[ps]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
                if label=='person':
                    cv2.resize(img,(250, 250));
                    count += 1
                    cv2.imwrite("static/PeopleImages1/frame%d.jpg" % count, img)
                else:
                    pass
                    
                    
    
    
        cv2.imshow("Image", img)
        
            #print(success)
            # Saves the frames with frame-count
        #vidObj.release() 
    cv2.destroyAllWindows()
    image_names = os.listdir(r'static/PeopleImages1/')
    b=os.listdir(r'static/PeopleImages1/')
    if len(b)>0:
        import pygame
        pygame.init()
        pygame.mixer.music.load('100.mp3')
        pygame.mixer.music.play()

        clock = pygame.time.Clock()
        while pygame.mixer.music.get_busy():
            pygame.event.poll()
            clock.tick(10)
    image_names = os.listdir(r'static/PeopleImages1/')
    
   
    cv2.destroyAllWindows()
    
    return render_template('output.html',image_names=image_names,a=a)

@app.route('/detectobject')
def detectobject():
    print('hi')
    import cv2
    import numpy as np
    
    # Load Yolo
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    image_names = os.listdir(r'static/onlyviolence/')
    for i in os.listdir('static/onlyviolence/'):
        print(str(i))
        i=str(i)
        a=cv2.imread('static/onlyviolence/'+str(i))
        #print(a.shape)
        #print(a)
        #print(type(a.shape))
        height, width, channels = a.shape
        blob = cv2.dnn.blobFromImage(a, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
        net.setInput(blob)
        outs = net.forward(output_layers)
        #print(blob)
    # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
    
                # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
    
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(a, (x, y), (x + w, y + h), color, 2)
                cv2.putText(a, label, (x, y + 30), font, 3, color, 3)
    
    
        cv2.imshow('Image',a)
          
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    
    return render_template('output1.html',image_names=image_names)
          
  # Load Yolo
import numpy as np




@app.route('/recognitionofperson1')
def recognitionofperson1():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    global timeintaerval,today
    train_dir = 'facedata/'
    val_dir = 'facedata/'
    con = dbConnection()
    cursor = con.cursor()
    firstFrame = None
# emotion_model.load_weights('emotion_model.h5')

    cv2.ocl.setUseOpenCL(False)
    #cap = cv2.VideoCapture(0)
    nameofuser=''
    a_dict = {}
    fps =10# int(vs.get(cv2.CAP_PROP_FPS))
    count=0
    #con = dbConnection()
    #cursor = con.cursor()
    i=10
    while True:
    # Find haar cascade to draw bounding box around face
        frame = vs.read()
        ret=True
        if not ret:
            break
        out1.write(frame) 
        
        
        img =frame# cv2.imread("room_ser.jpg")
    #img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

# Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

# Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
            # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

            # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)


        cv2.imshow("Image", img)
        time.sleep(5)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        normalframe=img
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
        text = "Unoccupied"
    # output the frame 
        while i<=10 :
            dirname = 'static/cam_images/'
            ts = time.time()
            filetowrite=str(ts)+'.png'
            completefilename=dirname+filetowrite
            cv2.imwrite(completefilename,img)
            #sql = "INSERT INTO objectdetails1 (objectnamefound,imagename,date) VALUES (%s, %s, %s)"
            #val = ('', completefilename,today)
            #cursor.execute(sql, val)
            #con.commit()
            print('successfully written  frame')
            i=i-1
        count+=1
      
        bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
        if firstFrame is None:
            firstFrame = gray_frame
            continue

    # compute the absolute difference between the current frame and
    # first frame
        frameDelta = cv2.absdiff(firstFrame, gray_frame)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

    # loop over the contours
        for c in cnts:
        # if the contour is too small, ignore it
            if cv2.contourArea(c) < 150:
                continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            if text=="Occupied":
                out.write(normalframe) 
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Occupied"
            

    # draw the text and timestamp on the frame
        
        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.imshow('Video12', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        '''for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            namefound=namedir[maxindex]
            nameofuser=namefound
            print(namefound)
            if namefound in a_dict:
                a_dict[namefound] += 1
            else:
                a_dict[namefound] = 1
            result_count = cursor.execute('SELECT * FROM user_company_information WHERE username = %s',(namefound))
            print('SELECT * FROM user_company_information WHERE username = %s',(namefound))
            res = cursor.fetchone()
            #print(res)
            userinfo=''
            if result_count > 0:
                print('inside')
                print(result_count)
            
                userinfo =namefound+'\n'+ str(res[0])+'\n'+ res[1]+'\n'+ res[2]+'\n'+ res[3]+'\n'+ res[4]+'\n'+ res[5]
            #cv2.putText(frame,namedir[maxindex]+str(maxindex), (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            y0, dy = 50, 50
            for i, line in enumerate(userinfo.split('\n')):
                y = y0 + i*dy
                cv2.putText(frame, line, (50, y ), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2, cv2.LINE_AA)
            #cv2.putText(frame,userinfo, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            '''

    dbClose()
    out.release() 
    out1.release() 
    
    
    
    cv2.destroyAllWindows()
    vs.stop()
    li2=[]
    global drive,completeVideoname,completeVideoname1
    path = completeVideoname  
   
# iterating thought all the files/folder
# of the desired directory
#for x in os.listdir(path):
   
    f = drive.CreateFile({'title': completeVideoname})
    f.SetContentFile(path)
    f.Upload()
  
    # Due to a known bug in pydrive if we 
    # don't empty the variable used to
    # upload the files to Google Drive the
    # file stays open in memory and causes a
    # memory leak, therefore preventing its 
    # deletion
    f = None
    path = completeVideoname1  
   
# iterating thought all the files/folder
# of the desired directory
#for x in os.listdir(path):
   
    f = drive.CreateFile({'title': completeVideoname1})
    f.SetContentFile(path)
    f.Upload()
  
    # Due to a known bug in pydrive if we 
    # don't empty the variable used to
    # upload the files to Google Drive the
    # file stays open in memory and causes a
    # memory leak, therefore preventing its 
    # deletion
    f = None
    #response = send_from_directory(directory='your-directory', filename='your-file-name')
    #response.headers['my-custom-header'] = 'my-custom-status-0'

    #print('list is',li2)
    return render_template('output.html',data=li2)

@app.route("/get_file",methods=["GET","POST"])
def get_file():
    path=''
    UPLOAD_DIRECTORY=''
    if request.method == "GET":
        filename = request.args["filename"]
        UPLOAD_DIRECTORY = "static//cam_videos"
        path=filename
    """Download a file."""
    
    
    #return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)
    return send_from_directory('', path, as_attachment=True)

@app.route('/livefeed')
def livefeed():
    """Video streaming home page."""
    global usernamelist
    global vs
    global total
    total=0
   
    vs = VideoStream(src=0).start()
    return render_template('livefeed.html')    
@app.route('/home')
def home():
    if 'user' in session:
        return render_template('home.html', user=session['user'])
    return redirect(url_for('index'))


def inserImageDetails():
    con = dbConnection()
    cursor = con.cursor()
    dirname = 'static/'
    ts = time.time()
    filetowrite=str(ts)+'.png'
    #cv2.imwrite(dirname+"//"+filetowrite,frame)
    sql = "INSERT INTO objectdetails1 (objectnamefound,imagename) VALUES (%s, %s)"
    val = ('', '')
    cursor.execute(sql, val)
    con.commit()
    
@app.route('/getimages', methods=["GET","POST"])
def getimages():
    if 'user' in session:
        if request.method == "POST":
            date = request.form.get("date")
            time = request.form.get("time")
            con = dbConnection()
            cursor = con.cursor()
            print("SELECT * FROM objectdetails1 where date='"+str(date)+"'")
            cursor.execute("SELECT * FROM objectdetails1 where date='"+str(date)+"'")
            result = cursor.fetchall()
            
            return render_template('displayimage.html',result=result, user=session['user'])
        return render_template('getimages.html', user=session['user'])
    return redirect(url_for('index'))


@app.route('/getvideos', methods=["GET","POST"])
def getvideos():
    if 'user' in session:
        if request.method == "POST":
            date = request.form.get("date")
            time = request.form.get("time")
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute("SELECT * FROM videodetails where date='"+date+"'")
            result = cursor.fetchall()
            
            return render_template('displayvideo.html',result=result, user=session['user'])
        return render_template('getvideos.html', user=session['user'])
    return redirect(url_for('index'))

def detect_motion(frameCount):
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock,total

    # initialize the motion detector and the total number of frames
    # read thus far
    #md = SingleMotionDetector(accumWeight=0.1)


    # loop over frames from the video stream
    while True:
        global path
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        frame1 =  vs.read()
       
        
        #time.sleep(3.0)
        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            outputFrame = frame1.copy()       
def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            svPath=r"E:\C Desktop\Company Tasks\Projects Assigned to me\Violence Detection system\Dataset\ui\CCTV\livefeed/"
            # encode the frame in JPEG format
            numFrame = 0
            i=20
            while i<=30:
                (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
                newPath = svPath + str(numFrame) + ".jpg"
                cv2.imencode('.jpg', outputFrame)[1].tofile(newPath)
                i=i-1
                numFrame+=1
            #cv2.imencode('.jpg', outputFrame)[1].tofile(newPath)
            #import numpy as np
            #from tensorflow.keras.preprocessing import image
            #import tensorflow
            #test_image = image.load_img('encodedImage.jpg', target_size = (250, 250))
            #test_image = image.img_to_array(test_image)/255
            #test_image = np.expand_dims(test_image, axis = 0)
            #filename7='new_model.hp5'
            #loaded_model6= tensorflow.keras.models.load_model(filename7)
            #result = loaded_model6.predict(test_image).round(3)
            #print(result)
            #nonviolence=[]
            #violence=[]
            #pred = np.argmax(result)
            #print(result, "--->>>", pred)
        
            #if pred == 0:
            #    nonviolence.append(outputFrame)
                
            #    print('Predicted>>> Image is Not categorized in Violence')
            #else:
            #    #a+=1
            #    violence.append(outputFrame)
            #    import shutil
            #shutil.move(r"E:\C Desktop\Company Tasks\Projects Assigned to me\Violence Detection system\Dataset\ui\CCTV\Images/"+outputFrame, "static\onlyviolence/"+outputFrame)
             #   print('Predicted>>> Image is catgorized Violence')
            

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')
@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")
if __name__ == '__main__':
    t = threading.Thread(target=detect_motion, args=(
        1,))
    t.daemon = False
    t.start()
    #app.run(debug=True)
    app.run('0.0.0.0')