from keras import backend as K
import numpy as np
import tensorflow as tf
import math
y_true = tf.constant([[[2.,5.,3]]])
y_pred = tf.constant([[[0.5,2.,5]]])
bce = K.binary_crossentropy(y_true, y_pred)
class_loglosses = K.mean(bce, axis=[0, 1,2])

class_weights = [0.1, 0.9]#note that the weights can be computed automatically using the training smaples
weighted_bce = K.sum(class_loglosses * K.constant(class_weights))

    # return K.weighted_binary_crossentropy(y_true, y_pred,pos_weight) + 0.35 * (self.dice_coef_loss(y_true, y_pred)) #not work

sess = tf.Session()

print(sess.run([bce,class_loglosses]))

#print(-(target[0]*math.log(lossinput[0])+(1-target[0])*math.log(1-lossinput[0])))
print(math.log(10))
print(-(2.*math.log(0.5)+(1.-2.)*math.log(1.-0.5)))