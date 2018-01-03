
import common
import imgtodata
import tensorflow as tf


# 评估
sess, y_conv, x, y_, keep_prob, train_step, correct_prediction, accuracy = common.preduce()
print('加载模型')
saver = tf.train.Saver()
save_path = './tmp/model.ckpt'
try:
    # arg:获取最近一次保存的变量文件名称
    module_file = tf.train.latest_checkpoint('./tmp')
    print(module_file)
    saver.restore(sess, module_file)
    print("模型加载完毕")
except:
    pass

print('评估开始')
images, labs = imgtodata.loaddata('./tmp/data/1.bin')
print("预测文件的准确率为： %g" % accuracy.eval(feed_dict={
    x: images, y_: labs, keep_prob: 1.0}))
