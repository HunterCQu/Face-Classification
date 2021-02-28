# coding=utf-8
import cv2
import tensorflow as tf

# 这里用数字0代表Hunter，数字1代表zheng，并将其转化为tensor便于之后与预测值做比较
Hunter = [0]
zheng = [1]
zheng = tf.convert_to_tensor(zheng, dtype=tf.int32)
Hunter = tf.convert_to_tensor(Hunter, dtype=tf.int32)

my_net = tf.keras.models.load_model('model.h5')

# 调用计算机自带的摄像头
camera = cv2.VideoCapture(0)
# 这个人脸特征识别文件我在上篇文章有提到，这里写下它的绝对路径就行
haar = cv2.CascadeClassifier('D:/Anacondad3/envs/tf2.1/Lib/site-packages/opencv_python-4.4.0.44.dist-info/haarcascade_frontalface_default.xml')
n = 1
while 1:
    if n <= 20000:
        print('It`s processing %s image.' % n)
        success, img = camera.read()

        # 做灰度转换
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 获取图片中的所有人脸信息
        faces = haar.detectMultiScale(gray_img, 1.3, 5)
        for f_x, f_y, f_w, f_h in faces:
            # 截取得到人脸图片
            face = img[f_y:f_y + f_h, f_x:f_x + f_w]
            # 修改图片大小为64*64
            face = cv2.resize(face, (64, 64))
            # 将图片数据类型转换为tensor类型，完成后shape为[64, 64, 3]
            face_tensor = tf.convert_to_tensor(face)
            # 在0维度左侧增加一个维度，即[64, 64, 3]=>[1, 64, 64, 3]
            face_tensor = tf.expand_dims(face_tensor, axis=0)
            # 将tensor类型从uint8转换为float32
            face_tensor = tf.cast(face_tensor, dtype=tf.float32)
            # print('face_tensor', face_tensor)
            # 输入至训练好的网络
            logits = my_net(face_tensor)
            # 将每一行进行softmax
            prob = tf.nn.softmax(logits, axis=1)
            print('prob:', prob)
            # 取出prob中每一行最大值对应的索引
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            print('pred:', pred)
            # 把预测值与Hunter和zheng的标签做比较，并将结果写在图像上
            if tf.equal(pred, zheng):
                cv2.putText(img, 'zheng', (f_x, f_y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, 255, 2)
            if tf.equal(pred, Hunter):
                cv2.putText(img, 'Hunter', (f_x, f_y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, 255, 2)
            img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 0, 0), 2)
            n += 1
        cv2.imshow('img', img)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
    else:
        break
camera.release()
cv2.destroyAllWindows()
