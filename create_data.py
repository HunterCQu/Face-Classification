# coding=utf-8
import tensorflow as tf
import glob
import random
import csv
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

root_img = './face_images'
img_mean = tf.constant([0.485, 0.456, 0.406])
img_std = tf.constant([0.229, 0.224, 0.225])


# root为我们之前获得图片数据的根目录face_images，filename为我们要加载的csv文件，
# name2label为我们获取的图片类型字典
def load_csv(root, filename, name2label):
    # 如果根目录root下不存在filename文件，那么创建一个filename文件
    if not os.path.exists(os.path.join(root, filename)):
        # 创建一个图片路径的列表images
        images = []
        # 遍历字典里所有的元素，例如我的第一个为'xu'，第二个为‘zheng’
        for name in name2label.keys():
            # 将路径下所有的jpg图片的路径写至images列表中
            images += glob.glob(os.path.join(root, name, '*.jpg'))
        print(len(images), images)
        # 对images进行随机打乱
        random.shuffle(images)
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images:
                # 获取路径最底层文件夹的名字
                # os.sep为路径的分隔符，split函数以给定分隔符进行切片（默认以空格切片），取从右往左数第二个
                # img = '...a/b/c/haha.jpg' =>['...a', 'b', 'c', 'haha.jpg'], -2指的是'c'
                name = img.split(os.sep)[-2]
                # 查找字典对应元素的值
                label = name2label[name]
                # 添加到路径的后面
                writer.writerow([img, label])
            print('written into csv file:', filename)
    # 如果存在filename文件，将其读取至imgs, labels这两个列表中
    imgs, labels = [], []
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:
            # 读取路径和对应的label值
            img, label = row
            label = int(label)
            # 将其分别压入列表中，并返回出来
            imgs.append(img)
            labels.append(label)
    return imgs, labels


def load_faceimg(root, mode='train'):
    # 创建图片类型字典，准备调用load_csv方法
    name2label = {}
    for name in sorted(os.listdir(os.path.join(root))):
        # 跳过root目录下不是文件夹的文件
        if not os.path.isdir(os.path.join(root, name)):
            continue
        # name为根目录下各个文件夹的名字
        # name2label.keys表示字典name2label里所有的元素，len表示字典所有元素的个数
        # 一开始字典是没有元素的，所以'xu'的值为0, 之后字典元素有个一个，所以'zheng'的值为1
        name2label[name] = len(name2label.keys())
    # 调用load_csv方法，返回值images为储存图片的目录的列表，labels为储存图片种类编码的列表
    images, labels = load_csv(root, 'images.csv', name2label)
    # 我们将前60%取为训练集，后20%取为验证集，最后20%取为测试集，并返回
    if mode == 'train':
        images = images[:int(0.6 * len(images))]
        labels = labels[:int(0.6 * len(labels))]
    elif mode == 'val':
        images = images[int(0.6 * len(images)):int(0.8 * len(images))]
        labels = labels[int(0.6 * len(labels)):int(0.8 * len(labels))]
    else:
        images = images[int(0.8 * len(images)):]
        labels = labels[int(0.8 * len(labels)):]
    return images, labels, name2label


def normalize(x, mean=img_mean, std=img_std):
    # 标准化
    # x: [64, 64, 3]
    # mean: [64, 64, 3], std: [3]
    x = (x - mean) / std
    return x


# x：图片的路径List, y: 图片种类的数字编码List
def get_tensor(x, y):
    # 创建一个列表ims
    ims = []
    for i in x:
        # 读取路径下的图片
        p = tf.io.read_file(i)
        # 对图片进行解码，RGB，3通道
        p = tf.image.decode_jpeg(p, channels=3)
        # 修改图片大小为64*64
        p = tf.image.resize(p, [64, 64])
        # 将图片压入ims列表中
        ims.append(p)
    # 将List类型转换为tensor类型，并返回
    ims = tf.convert_to_tensor(ims)
    y = tf.convert_to_tensor(y)
    return ims, y


# 预处理函数，x, y均为tensor类型
def preprocess(x, y):
    # 数据增强
    x = tf.image.random_flip_left_right(x)  # 左右镜像
    x = tf.image.random_crop(x, [64, 64, 3])  # 随机裁剪
    # x: [0,255]=>0~1，将其值转换为float32
    x = tf.cast(x, dtype=tf.float32) / 255.
    # 0~1 => D(0, 1)
    x = normalize(x)
    # 将其值转换为int32
    y = tf.cast(y, dtype=tf.int32)
    return x, y


# 加载图片，获得图片路径与图片种类编码的列表
images_train, labels_train, name2label = load_faceimg(root_img, mode='train')
images_val, labels_val, _ = load_faceimg(root_img, mode='val')
images_test, labels_test, _ = load_faceimg(root_img, mode='test')

# 从对应路径读取图片，并将列表转换为张量
x_train, y_train = get_tensor(images_train, labels_train)
x_val, y_val = get_tensor(images_val, labels_val)
x_test, y_test = get_tensor(images_test, labels_test)

# 可输出查看它们的shape
print('x_train:', x_train.shape, 'y_train:', y_train.shape)
print('x_val:', x_val.shape, 'y_val:', y_val.shape)
print('x_test:', x_test.shape, 'y_test:', y_test.shape)

# 切分传入参数的第一个维度，并进行随机打散，预处理和打包处理
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).map(preprocess).batch(10)

db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
db_val = db_val.map(preprocess).batch(10)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(10)

# 创建一个迭代器，可以查看其shape大小
sample_train = next(iter(db_train))
sample_val = next(iter(db_val))
sample_test = next(iter(db_test))
print('sample_train:', sample_train[0].shape, sample_train[1].shape)
print('sample_val:', sample_val[0].shape, sample_val[1].shape)
print('sample_test:', sample_test[0].shape, sample_test[1].shape)






