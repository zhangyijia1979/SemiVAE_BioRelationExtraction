#coding=utf-8
'''
Created on 2018.1.3

@author: DUTIR
'''
import sys
import time

import h5py
from keras import layers
from keras import metrics
from keras.models import Model,load_model
from keras.layers import Dense, Dropout, Activation,Input,Convolution1D,LSTM,Embedding, Merge

from keras.layers.core import  Flatten,Lambda,Reshape
from keras.layers.convolutional import MaxPooling1D

from keras.optimizers import RMSprop

from keras.preprocessing import sequence
from keras.engine.topology import Layer

import pickle as pkl

from keras import backend as K
from keras import utils

import numpy as np

#evaluation of DDI extraction results. 4 DDI tpyes
def result_evaluation(y_test,pred_test):

    pred_matrix = np.zeros_like(pred_test, dtype=np.int8)

    y_matrix = np.zeros_like(y_test, dtype=np.int8)
    pred_indexs = np.argmax(pred_test, 1)
    y_indexs = np.argmax(y_test, 1)

    for i in range(len(pred_indexs)):
        pred_matrix[i][pred_indexs[i]] = 1
        y_matrix[i][y_indexs[i]] = 1

    count_matrix=np.zeros((4,3))
    for class_idx in range(4):

        count_matrix[class_idx][0] = np.sum(np.array(pred_matrix[:, class_idx]) * np.array(y_matrix[:, class_idx]))#tp
        count_matrix[class_idx][1] = np.sum(np.array(pred_matrix[:, class_idx]) * (1 - np.array(y_matrix[:, class_idx])))#fp
        count_matrix[class_idx][2] = np.sum((1 - np.array(pred_matrix[:, class_idx])) * np.array(y_matrix[:, class_idx]))#fn

    sumtp=sumfp=sumfn=0

    for i in range(4):
        sumtp+=count_matrix[i][0]
        sumfp+=count_matrix[i][1]
        sumfn+=count_matrix[i][2]

        precision=recall=f1=0

    if (sumtp + sumfp) == 0:
        precision = 0.
    else:
        precision = float(sumtp) / (sumtp + sumfp)

    if (sumtp + sumfn) == 0:
        recall = 0.
    else:
        recall = float(sumtp) / (sumtp + sumfn)

    if (precision + recall) == 0.:
        f1 = 0.
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision,recall,f1


def inst_layers(layers, in_layer):
    x = in_layer
    for layer in layers:
        if isinstance(layer, list):
            x = inst_layers(layer, x)
        else:
            x = layer(x)
    return x

def sampling(args):
    z_mean, z_log_var = args
    # z_mean= args[0]
    # z_log_var=args[1]
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 32), mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_var / 2) * epsilon


class labeled_semi_variational_layer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        self.alpha = 100

        self.entity_sequence_length = 30
        super(labeled_semi_variational_layer, self).__init__(**kwargs)

    def label_vae_loss(self, x_word, decoder_word_mean, _z_mean,_z_log_var,x_label,classify_output):

        self.word_loss = K.mean( self.entity_sequence_length * metrics.sparse_categorical_crossentropy(x_word, decoder_word_mean))

        self.kl_loss = K.mean(- 0.5 * K.sum(1 + _z_log_var - K.square(_z_mean) - K.exp(_z_log_var), axis=-1))

        self.cls_loss = K.mean(self.alpha * metrics.categorical_crossentropy(x_label, classify_output))

        return self.word_loss + self.kl_loss+self.cls_loss

    def call(self, inputs):
        x_word = inputs[0]
        decoder_word_mean = inputs[1]
        _z_mean=inputs[2]
        _z_log_var = inputs[3]
        x_label = inputs[4]
        classify_output = inputs[5]
        loss = self.label_vae_loss(x_word, decoder_word_mean, _z_mean,_z_log_var,x_label,classify_output )
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return decoder_word_mean

class unlabeled_semi_variational_layer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        #self.alpha = 100

        self.entity_sequence_length=30
        super(unlabeled_semi_variational_layer, self).__init__(**kwargs)

    def unlabel_vae_loss(self, x_word, decoder_word_mean, _z_mean,_z_log_var):

        self.word_loss = K.mean( self.entity_sequence_length * metrics.sparse_categorical_crossentropy(x_word, decoder_word_mean))

        self.kl_loss = K.mean(- 0.5 * K.sum(1 + _z_log_var - K.square(_z_mean) - K.exp(_z_log_var), axis=-1))

        return self.kl_loss + self.word_loss

    def call(self, inputs):
        x_word = inputs[0]
        decoder_word_mean = inputs[1]
        _z_mean=inputs[2]
        _z_log_var = inputs[3]

        loss = self.unlabel_vae_loss(x_word, decoder_word_mean, _z_mean,_z_log_var)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return decoder_word_mean

if __name__ == '__main__':

        s = {
             'emb_dimension':100, # dimension of word embedding
             'batch_size':64,
             'epochs':10,
             'class_num':5,
            'dropout':0.5,
            'train_file': "./train.pkl",
            'test_file': "./test.pkl",
            'wordvecfile': "./vec.pkl",
            'nb_filter':300,
            'vocsize':5000,
            'encoder_hidden_layer':300,
            'decoder_hidden_layer_1': 300,
            'decoder_hidden_layer_2': 600,
            'decoder_hidden_layer_3': 1000,
            'latent_dim':32,
            'sentence_length':150,
            'labeled_data_size':2000,
            'validate_rate':0.1,
            'entity_sequence_length':30
            }


        f_word2vec = open(s['wordvecfile'], 'rb')
        vec_table = pkl.load(f_word2vec)
        f_word2vec.close()
        #print(vec_table.shape)
        #print (vec_table)

        f_Train = open(s['train_file'], 'rb')

        train_labels_vec = pkl.load(f_Train)
        train_all_words = pkl.load(f_Train)

        train_all_dis1 = pkl.load(f_Train)
        train_all_dis2 = pkl.load(f_Train)
        train_entity_sequence = pkl.load(f_Train)

        f_Train.close()

        f_Test = open(s['test_file'], 'rb')
        test_labels_vec = pkl.load(f_Test)
        test_all_words = pkl.load(f_Test)

        test_all_dis1 = pkl.load(f_Test)
        test_all_dis2 = pkl.load(f_Test)
        test_entity_sequence = pkl.load(f_Test)

        f_Test.close()

        train_labels = np.argmax(train_labels_vec, axis=-1)
        test_labels=np.argmax(test_labels_vec,axis=-1)

        train_all_words = sequence.pad_sequences(train_all_words, maxlen=s['sentence_length'],truncating='post', padding='post')
        test_all_words = sequence.pad_sequences(test_all_words, maxlen=s['sentence_length'],truncating='post',padding='post')

        train_all_dis1 = sequence.pad_sequences(train_all_dis1, maxlen=s['sentence_length'],
                                               truncating='post', padding='post')
        test_all_dis1 = sequence.pad_sequences(test_all_dis1, maxlen=s['sentence_length'],
                                              truncating='post',
                                              padding='post')

        train_all_dis2 = sequence.pad_sequences(train_all_dis2, maxlen=s['sentence_length'],
                                               truncating='post', padding='post')
        test_all_dis2 = sequence.pad_sequences(test_all_dis2, maxlen=s['sentence_length'],
                                              truncating='post',
                                              padding='post')

        result_out = open("./semi_vae_output.txt", 'w+')

        sample_rate=s['labeled_data_size']/ len(train_labels)

        # for calculate the mean results
        p_list = []
        r_list = []
        f_list = []

        #### training repeat 10 times to reduce the selection bias
        for random_times in range(10):

            #############construct semi corpus##########
            indexes_train = np.array([i for i in range(len(train_all_words))], dtype=np.int32)
            indexes_train = np.random.permutation(indexes_train)
            train_all_words = np.array([np.copy(train_all_words[index]) for index in indexes_train])
            train_labels = np.array([np.copy(train_labels[index]) for index in indexes_train])

            train_all_dis1 = np.array([np.copy(train_all_dis1[index]) for index in indexes_train])
            train_all_dis2 = np.array([np.copy(train_all_dis2[index]) for index in indexes_train])

            train_entity_sequence=np.array([np.copy(train_entity_sequence[index]) for index in indexes_train])

            indexes_test = np.array([i for i in range(len(test_all_words))], dtype=np.int32)
            indexes_test = np.random.permutation(indexes_test)
            test_all_words = np.array([np.copy(test_all_words[index]) for index in indexes_test])
            test_labels = np.array([np.copy(test_labels[index]) for index in indexes_test])

            test_all_dis1 = np.array([np.copy(test_all_dis1[index]) for index in indexes_test])
            test_all_dis2 = np.array([np.copy(test_all_dis2[index]) for index in indexes_test])

            test_entity_sequence = np.array([np.copy(test_entity_sequence[index]) for index in indexes_test])


            #split the training data into labeled data, unlabeled data and validate data
            total_number_each_class = [0] * s['class_num']
            sample_number_each_class = [0] * s['class_num']
            validate_number_each_class = [0] * s['class_num']

            #print(len(train_labels))
            train_indices = np.arange(len(train_labels))
            # print (train_labels)
            for label_index in range(len(train_labels)):
                total_number_each_class[train_labels[label_index]] += 1

            for instance_indices in range(len(total_number_each_class)):
                sample_number_each_class[instance_indices] = int( total_number_each_class[instance_indices] * sample_rate)

            for instance_indices in range(len(total_number_each_class)):
                validate_number_each_class[instance_indices] = int( total_number_each_class[instance_indices] * s['validate_rate'])

            print(total_number_each_class)
            print(sample_number_each_class)


            i_labeled = []
            i_unlabeled = []
            i_validated = []
            for c in range(s['class_num']):

                i = train_indices[train_labels == c][:sample_number_each_class[c]]
                j = train_indices[train_labels == c][ sample_number_each_class[c]:sample_number_each_class[c] + validate_number_each_class[c]]
                k = train_indices[train_labels == c][ sample_number_each_class[c] + validate_number_each_class[c]:total_number_each_class[c]]
                i_labeled += list(i)
                i_validated += list(j)
                i_unlabeled += list(k)


            #generate labeled data
            semi_labeled_all_words = train_all_words[i_labeled]
            semi_labeled_labels = train_labels[i_labeled]

            semi_labeled_all_dis1 = train_all_dis1[i_labeled]
            semi_labeled_all_dis2 = train_all_dis2[i_labeled]

            semi_labeled_entity_sequence = train_entity_sequence[i_labeled]

            semi_labeled_index = np.arange(len(semi_labeled_all_words))

            np.random.shuffle(semi_labeled_index)

            semi_labeled_all_words = semi_labeled_all_words[semi_labeled_index]
            semi_labeled_labels = semi_labeled_labels[semi_labeled_index]

            semi_labeled_all_dis1 = semi_labeled_all_dis1[semi_labeled_index]
            semi_labeled_all_dis2 = semi_labeled_all_dis2[semi_labeled_index]
            semi_labeled_entity_sequence=semi_labeled_entity_sequence[semi_labeled_index]

            # generate unlabeled data
            semi_unlabel_all_words = train_all_words[i_unlabeled]

            ##we don't use the labels of unlabel data
            #semi_unlabel_labels = train_labels[i_unlabeled]
            semi_unlabel_all_dis1 = train_all_dis1[i_unlabeled]
            semi_unlabel_all_dis2 = train_all_dis2[i_unlabeled]
            semi_unlabel_entity_sequence = train_entity_sequence[i_unlabeled]

            # generate validate data
            ##the validate data is used to choose the hyper-parameters and the epoch number of the model
            semi_validated_all_words = train_all_words[i_validated]
            semi_validated_labels = train_labels[i_validated]

            semi_validated_all_dis1 = train_all_dis1[i_validated]
            semi_validated_all_dis2 = train_all_dis2[i_validated]

            semi_validated_entity_sequence = train_entity_sequence[i_validated]

            semi_labeled_y=utils.to_categorical(semi_labeled_labels, num_classes=s['class_num'])
            semi_validated_y = utils.to_categorical(semi_validated_labels, num_classes=s['class_num'])

            test_y = utils.to_categorical(test_labels, num_classes=s['class_num'])

            ##word embedding
            wordembedding = Embedding(vec_table.shape[0],
                                 vec_table.shape[1],
                                weights=[vec_table]
                                      )
            ##position embedding
            disembedding = Embedding(650,
                                     20,
                                     #weights=[dis_vec_table]
                                     )

            input_all_word = Input(shape=(s['sentence_length'],), dtype='int32', name='input_all_word')
            all_word_fea = wordembedding(input_all_word)  # trainable=False

            input_all_dis1 = Input(shape=(s['sentence_length'],), dtype='int32', name='input_all_dis1')
            all_dis_fea1 = disembedding(input_all_dis1)

            input_all_dis2 = Input(shape=(s['sentence_length'],), dtype='int32', name='input_all_dis2')
            all_dis_fea2 = disembedding(input_all_dis2)

            input_entity_sequence = Input(shape=(s['entity_sequence_length'],), dtype='int32', name='input_entity_sequence')
            entity_sequence_fea = wordembedding(input_entity_sequence)

            input_label = Input(shape=( s['class_num'],), dtype='float32', name='input_label')

            #emb_merge = Merge(mode='concat')([all_word_fea, all_dis_fea1,all_dis_fea2])
            emb_merge = layers.concatenate([all_word_fea, all_dis_fea1, all_dis_fea2], axis=-1)

            ##embed layer dropout

            emb_merge = Dropout(0.5)(emb_merge)

            cnn_word = Convolution1D(nb_filter=s['nb_filter'],
                                     filter_length=3,
                                     border_mode='same',
                                     activation='tanh',
                                     subsample_length=1)(emb_merge)

            print (cnn_word.shape)
            cnn_word = MaxPooling1D(pool_length=s['sentence_length'])(cnn_word)
            cnn_word = Flatten()(cnn_word)
            classify_drop = Dropout(0.5)(cnn_word)
            classify_output = Dense(s['class_num'])(classify_drop)
            classify_output = Activation('softmax')(classify_output)

            ##entity sequence dropout
            entity_sequence_fea = Dropout(0.5)(entity_sequence_fea)

            left_lstm = LSTM(output_dim=s['encoder_hidden_layer'],
                            init='orthogonal',
                            activation='tanh',
                            inner_activation='sigmoid')(entity_sequence_fea)

            right_lstm = LSTM(output_dim=s['encoder_hidden_layer'],
                             init='orthogonal',
                             activation='tanh',
                             inner_activation='sigmoid',
                             go_backwards=True)(entity_sequence_fea)

            #lstm_merge = Merge(mode='concat')([left_lstm, right_lstm])
            lstm_merge = layers.concatenate([left_lstm, right_lstm], axis=-1)
            lstm_merge = Dropout(0.5)(lstm_merge)

            _z_mean = Dense(s['latent_dim'])(lstm_merge)
            _z_log_var = Dense(s['latent_dim'])(lstm_merge)

            z = Lambda(sampling, output_shape=(s['latent_dim'],))([_z_mean, _z_log_var])

            labed_data_merged = Merge(mode='concat')([input_label, z])
            labed_data_merged = layers.concatenate([input_label, z], axis=-1)

            decoder_layers = [ Dense(s['entity_sequence_length'] * s['decoder_hidden_layer_1']),
                Reshape((s['entity_sequence_length'], s['decoder_hidden_layer_1'])),
                Convolution1D(nb_filter=s['decoder_hidden_layer_1'],
                              filter_length=3,
                              border_mode='same',
                              activation='tanh',
                              subsample_length=1),
                Convolution1D(nb_filter=s['decoder_hidden_layer_2'],
                              filter_length=3,
                              border_mode='same',
                              activation='tanh',
                              subsample_length=1),
                Convolution1D(nb_filter=s['decoder_hidden_layer_3'],
                              filter_length=3,
                              border_mode='same',
                              activation='tanh',
                              subsample_length=1),
                Dropout(0.5)
            ]


            label_dec_out = inst_layers(decoder_layers, labed_data_merged)

            label_decoder_word_mean = Dense(s['vocsize'], activation='softmax')(label_dec_out)


            #unlabel_data_merged = Merge(mode='concat')([classify_output, z])
            unlabel_data_merged = layers.concatenate([classify_output, z], axis=-1)

            unlabel_dec_out = inst_layers(decoder_layers, unlabel_data_merged)

            unlabel_decoder_word_mean = Dense(s['vocsize'], activation='softmax')(unlabel_dec_out)

            _output = labeled_semi_variational_layer()(
                [input_entity_sequence, label_decoder_word_mean, _z_mean, _z_log_var, input_label, classify_output])

            _unlabeled_output = unlabeled_semi_variational_layer()(
                [input_entity_sequence, unlabel_decoder_word_mean, _z_mean, _z_log_var])

            label_ddi_vae = Model(
                inputs=[input_all_word, input_all_dis1, input_all_dis2, input_entity_sequence, input_label],
                outputs=_output)

            keras_opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)

            label_ddi_vae.compile(optimizer=keras_opt, loss=None)
            label_ddi_vae.summary()

            unlabel_ddi_vae = Model(inputs=[input_all_word, input_all_dis1, input_all_dis2, input_entity_sequence],
                                    outputs=_unlabeled_output)

            unlabel_ddi_vae.compile(optimizer=keras_opt, loss=None)
            unlabel_ddi_vae.summary()

            print ("-----------Begin training the model--------------")
            batch_size=s['batch_size']

            ###align the unlabel data with the labeled data
            mag = len(semi_unlabel_all_words) // len(semi_labeled_all_words)
            print(len(semi_unlabel_all_words))
            semi_unlabel_all_words = semi_unlabel_all_words[0:mag * len(semi_labeled_all_words)]
            assert len(semi_unlabel_all_words) % len(semi_labeled_all_words) == 0
            # start = time.time()
            history = []
            max_f = max_p = max_r = 0
            #print (len(semi_unlabel_all_words))
            #print (len(semi_labeled_all_words))

            for epoch in range(s['epochs']):

                totalloss = loss = unlabeled_loss = labeled_loss = 0
                unlabeled_index = np.arange(len(semi_unlabel_all_words))
                np.random.shuffle(unlabeled_index)
                # Repeat the labeled data to match length of unlabeled data
                labeled_index = []
                for i in range(len(semi_unlabel_all_words) // len(semi_labeled_all_words)):
                    l = np.arange(len(semi_labeled_all_words))
                    np.random.shuffle(l)
                    labeled_index.append(l)
                labeled_index = np.concatenate(labeled_index)

                batches = len(semi_unlabel_all_words) // batch_size
                label_batches = len(semi_labeled_all_words) // batch_size

                for i in range(batches):
                    # Labeled data training
                    index_range = labeled_index[i * batch_size:(i + 1) * batch_size]
                    #print (index_range)
                    label_val = label_ddi_vae.train_on_batch([semi_labeled_all_words[index_range],
                                                              semi_labeled_all_dis1[index_range],
                                                                semi_labeled_all_dis2[index_range],
                                                                semi_labeled_entity_sequence[index_range],
                                                              semi_labeled_y[index_range]], y=None)

                    loss += label_val

                    # Unlabeled data training

                    index_range = unlabeled_index[i * batch_size:(i + 1) * batch_size]

                    unlabel_val = unlabel_ddi_vae.train_on_batch([semi_unlabel_all_words[index_range],
                                                                  semi_unlabel_all_dis1[index_range],
                                                                 semi_unlabel_all_dis2[index_range],
                                                                semi_unlabel_entity_sequence[index_range]], y=None)
                    
                    loss += unlabel_val

                    history.append(loss)

                    #labeled_loss = labeled_loss + label_val
                    #unlabeled_loss = unlabeled_loss + unlabel_val

                    if i % label_batches == 0:

                        # print('=', end="")
                        classifier = Model(inputs=[input_all_word, input_all_dis1, input_all_dis2],
                                           outputs=[classify_output])
                        y_pred = classifier.predict([semi_validated_all_words,semi_validated_all_dis1,semi_validated_all_dis2],
                                                    batch_size=s['batch_size'], verbose=1)

                        precision, recall, F1 = result_evaluation(semi_validated_y, y_pred)

                        print('training epochs:' + str(epoch) + ' precision:' + str(
                            np.round(precision, 5)) + ' recall:' + str(
                            np.round(recall, 5)) + ' F1:' + str(np.round(F1, 5)))

                        if F1 > max_f:
                            max_f = F1
                            max_p = precision
                            max_r = recall

                            classifier.save("./semi_vae_temp_model.h5")
            ###evaluate on test data
            test_classifier= load_model("./semi_vae_temp_model.h5")

            y_pred = test_classifier.predict([test_all_words, test_all_dis1, test_all_dis2],
                                        batch_size=s['batch_size'], verbose=1)

            precision, recall, F1 = result_evaluation(test_y, y_pred)

            print('random times:' + str(random_times) + ' precision:' + str(
                np.round(precision, 5)) + ' recall:' + str(
                np.round(recall, 5)) + ' F1:' + str(np.round(F1, 5)))

            result_out.write(
                'laten_size: ' + str(s['latent_dim']) + ' batch_size:' + str(batch_size) + ' sample_size:' + str(
                    s['labeled_data_size']) + ' random_time:' + str(random_times) + ' p:' + str( np.round(precision, 5)) +  ' r:' + str(np.round(recall, 5)) + ' F1:' + str(np.round(F1, 5)) + "\n")

            p_list.append(precision)
            r_list.append(recall)
            f_list.append(F1)

        p_array = np.array(p_list)
        r_array = np.array(r_list)
        f_array = np.array(f_list)
        avg_p = np.average(p_array)
        avg_r = np.average(r_array)
        avg_f = np.average(f_array)
        std_p = np.std(p_array)
        std_r = np.std(r_array)
        std_f = np.std(f_array)


        print('laten_size: ' + str(s['latent_dim']) + ' batch_size:' + str(
            s['batch_size']) + ' sample_size:' + str(s['labeled_data_size']) + ' average_precision:' + str(
            np.round(avg_p, 5)) + ' average_recall:' + str(
            np.round(avg_r, 5)) + ' average_F1:' + str(np.round(avg_f, 5)) + ' std_precision:' + str(
            np.round(std_p, 5)) + ' std_recall:' + str(
            np.round(std_r, 5)) + ' std_F1:' + str(np.round(std_f, 5)))

        result_out.write('laten_size: ' + str(s['latent_dim']) + ' batch_size:' + str(
            s['batch_size']) + ' sample_size:' + str(s['labeled_data_size']) + ' average_precision:' + str(
            np.round(avg_p, 5)) + ' average_recall:' + str(
            np.round(avg_r, 5)) + ' average_F1:' + str(np.round(avg_f, 5)) + ' std_precision:' + str(
            np.round(std_p, 5)) + ' std_recall:' + str(
            np.round(std_r, 5)) + ' std_F1:' + str(np.round(std_f, 5)))

        result_out.close()

