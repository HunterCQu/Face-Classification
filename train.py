from tensorflow.keras import layers, optimizers, datasets, Sequential
import tensorflow as tf
from create_data import db_train, db_test

# 由卷积层和全连接层组成，中间用Flatten层进行平铺, inputs: [None, 64, 64, 3]
my_layers = [
    layers.Conv2D(32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    layers.Flatten(),

    layers.Dense(512, activation=tf.nn.relu),
    layers.Dropout(rate=0.5),
    layers.Dense(2, activation=None)
]
my_net = Sequential(my_layers)
my_net.build(input_shape=[None, 64, 64, 3])
my_net.summary()
optimizer = optimizers.Adam(lr=1e-3)

acc_best = 0
patience_num = 10
no_improved_num = 0
for epoch in range(50):
    for step, (x, y) in enumerate(db_train):
        with tf.GradientTape() as tape:
            out = my_net(x)
            # print('out', out.shape)
            logits = out
            y_onehot = tf.one_hot(y, depth=2)
            loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, my_net.trainable_variables)
        optimizer.apply_gradients(zip(grads, my_net.trainable_variables))

        if step % 5 == 0:
            print(epoch, step, 'loss:', float(loss))

    total_num = 0
    total_correct = 0
    for x2, y2 in db_test:
        out = my_net(x2)
        logits = out
        prob = tf.nn.softmax(logits, axis=1)
        # tf.argmax() : axis=1 表示返回每一行最大值对应的索引, axis=0 表示返回每一列最大值对应的索引
        pred = tf.argmax(prob, axis=1)
        # 将pred转化为int32数据类型，便于后面与y2进行比较
        pred = tf.cast(pred, dtype=tf.int32)

        correct = tf.cast(tf.equal(pred, y2), dtype=tf.int32)
        correct = tf.reduce_sum(correct)

        total_num += x2.shape[0]
        total_correct += int(correct)
    acc = total_correct / total_num
    if acc > acc_best:
        acc_best = acc
        no_improved_num = 0
        my_net.save('model.h5')
    else:
        no_improved_num += 1
    print(epoch, 'acc:', acc, 'no_improved_num:', no_improved_num)
    if no_improved_num >= patience_num:
        break
