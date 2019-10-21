# coding=utf-8

from keras.layers import *
from keras.models import Model, load_model
from keras.utils import plot_model, np_utils
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np

import sys
sys.path.append("..")
import getData


def getClassWeight(labels):
    classWeightsList = []
    for temp in labels:
        y_integers = np.array(temp)
        class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
        d_class_weights = dict(enumerate(class_weights))
        classWeightsList.append(d_class_weights)

    return classWeightsList


def ourModel():
    image_inputs = Input(shape=(10,2,28,28,), dtype='float32')
    user_inputs = Input(shape=(1,), dtype='int32')

    uerPre = Embedding(max(userId) + 1, 64)(user_inputs)
    uerPre = Flatten()(uerPre)

    uerPre1 = Dense(490, activation='relu')(uerPre)
    uerPre1 = Reshape([10,7,7])(uerPre1)

    uerPre2 = Dense(10, activation='relu')(uerPre)

    image_inputs1 = Convolution3D(filters=10, kernel_size=(2,4,4), strides=(1,4,4), activation='sigmoid', name='conv1_1', data_format="channels_first")(image_inputs)

    image_inputs1 = Reshape([10,7,7])(image_inputs1)

    # sub-region attention method
    a1 = Multiply()([image_inputs1, uerPre1])
    a1 = Reshape([10,49])(a1)
    a1 = Dense(49, activation='tanh')(a1)
    a1 = Dense(49, activation='softmax')(a1)
    a1 = Reshape([10, 7, 7], name="featureMap")(a1)
    combine = Multiply()([image_inputs1, a1])

    combine = Reshape([10,1,7,7])(combine)

    layerConv1 = ConvLSTM2D(filters=1, kernel_size=[2,2], strides=(1,1), padding='same', data_format="channels_first", dilation_rate=(1, 1),
               activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True,
               kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros',
               unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
               activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
               return_sequences=True, go_backwards=False, stateful=False, dropout=0.0, recurrent_dropout=0.0)

    forwardRes,reverseRes = Bidirectional(layerConv1, merge_mode=None, weights=None,)(combine)

    # sequence attention method
    a1 = RepeatVector(49)(uerPre2)
    a1 = Reshape([7, 7, 10])(a1)
    a1 = Permute((3, 1, 2))(a1)

    forwardRes = Reshape([10, 7, 7])(forwardRes)
    reverseRes = Reshape([10, 7, 7])(reverseRes)
    combine1_1 = Multiply()([forwardRes, a1])
    combine1_2 = Multiply()([reverseRes, a1])
    combine1 = Add()([combine1_1, combine1_2])
    combine1 = Reshape([10, 7, 7])(combine1)

    a1 = Lambda(lambda x: K.sum(x, axis=[2,3]))(combine1)
    a1 = Dense(10, activation='tanh')(a1)
    a1 = Dense(10, activation='softmax')(a1)
    a1 = RepeatVector(49)(a1)
    a1 = Reshape([7, 7, 10])(a1)
    a1 = Permute((3, 1, 2), name="featureMap1")(a1)

    combine1 = Multiply()([combine1, a1])


    conv1 = Conv2D(filters=5, kernel_size=[2,2], strides=(1, 1), data_format="channels_first")(combine1)

    x = Flatten()(conv1)

    # the feeling of width
    x_w = Dense(64, activation='relu')(x)
    x_w = Dropout(0.1)(x_w)
    x_w = Dense(16, activation='relu')(x_w)
    weight_outputs = Dense(output_dim=3, activation='softmax', name='ctg_out_1')(x_w)

    # the feeling of length
    x_l = Dense(64, activation='relu')(x)
    x_l = Dropout(0.1)(x_l)
    x_l = Dense(16, activation='relu')(x_l)
    length_outputs = Dense(output_dim=3, activation='softmax', name='ctg_out_2')(x_l)

    total_x = Dense(64, activation='relu')(x)
    total_x = Dropout(0.1)(total_x)
    total_x = Dense(16, activation='relu')(total_x)
    total_outputs = Dense(output_dim=2, activation='softmax', name='ctg_out_3')(total_x)

    model = Model(inputs=[image_inputs, user_inputs],
                  outputs=[weight_outputs, length_outputs, total_outputs])


    model.compile(optimizer='adam',
                  loss={
                      'ctg_out_1': 'categorical_crossentropy',
                      'ctg_out_2': 'categorical_crossentropy',
                      'ctg_out_3': 'categorical_crossentropy'},
                  loss_weights={
                      'ctg_out_1': 0.5136,
                      'ctg_out_2': 0.2982,
                      'ctg_out_3': 1},
                  metrics=['accuracy'])

    print('Train...')
    print(model.summary())

    return model


if __name__ == '__main__':
    imageFeatures, userId, label_weight, label_length, label_total = getData.features_new("align")

    classWeightsList = getClassWeight([label_weight, label_length, label_total])

    label_weight = np_utils.to_categorical(label_weight, num_classes=3)
    label_length = np_utils.to_categorical(label_length, num_classes=3)
    label_total = np_utils.to_categorical(label_total, num_classes=2)

    num = int(len(userId)*0.2)

    imageFeatures_train = imageFeatures[0:-num]
    userId_train = userId[0:-num]
    label_weight_train = label_weight[0:-num]
    label_length_train = label_length[0:-num]
    label_total_train = label_total[0:-num]

    imageFeatures_test = imageFeatures[-num:]
    userId_test = userId[-num:]
    label_weight_test = label_weight[-num:]
    label_length_test = label_length[-num:]
    label_total_test = label_total[-num:]

    model = ourModel()
    model.save('my_model_finally.h5')

    y_pred = model.predict([imageFeatures_test, userId_test])
    y_pred_num = np.argmax(y_pred[0], axis=1)
    label_weight_test_num = np.argmax(label_weight_test, axis=1)
    print(classification_report(label_weight_test_num, y_pred_num, digits=4))
    print(accuracy_score(label_weight_test_num, y_pred_num))

    y_pred = model.predict([imageFeatures_test, userId_test])
    y_pred_num = np.argmax(y_pred[1], axis=1)
    label_length_test_num = np.argmax(label_length_test, axis=1)
    print(classification_report(label_length_test_num, y_pred_num, digits=4))
    print(accuracy_score(label_length_test_num, y_pred_num))

    y_pred = model.predict([imageFeatures_test, userId_test])
    y_pred_num = np.argmax(y_pred[2], axis=1)
    label_total_test_num = np.argmax(label_total_test, axis=1)
    print(classification_report(label_total_test_num, y_pred_num,digits=4))
    print(accuracy_score(label_total_test_num, y_pred_num))
