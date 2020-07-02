#-*-coding:utf-8-*-
import tensorflow as tf

from transformers import TFBertModel
from tokenization_kobert import KoBertTokenizer

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SEQ_LEN = 200

# Data
tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')

def convert_data(data_df):
    tokens, masks, segments, targets = [], [], [], []
    total_c = len(data_df)
    c = 1
    for i in data_df:
        token = tokenizer.encode(i.split('\t')[1], max_length=SEQ_LEN, pad_to_max_length=True)

        num_zeros = token.count(0)
        mask = [1]*(SEQ_LEN-num_zeros) + [0]*num_zeros

        segment = [0]*SEQ_LEN

        tokens.append(token)
        masks.append(mask)
        segments.append(segment)
        targets.append(i.split('\t')[2])
       
    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)
    targets = np.array(targets, dtype=np.float32)

    return [tokens, masks, segments], targets


def main():
	# Data
    data_f = './tr.txt'
    with open(data_f, 'r') as f:
        train = [l for l in f.read().splitlines()]
    train_x, train_y = convert_data(train)

    # Model
    model = TFBertModel.from_pretrained("monologg/kobert", from_pt=True)

    token_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_word_ids')
    mask_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_masks')
    segment_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_segment')

    bert_outputs = model([token_inputs, mask_inputs, segment_inputs])
    bert_outputs = bert_outputs[1]

    opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    sentiment_drop = tf.keras.layers.Dropout(0.5)(bert_outputs)
    sentiment_first = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(sentiment_drop)
    sentiment_model = tf.keras.Model([token_inputs, mask_inputs, segment_inputs], sentiment_first)

    sentiment_model.compile(optimizer=opt,
            loss=tf.keras.losses.binary_crossentropy,
            metrics=[tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])

    sentiment_model.fit(train_x, train_y, epochs=1, batch_size=16, validation_split=0.2)

    # Save model
    sentiment_model.save_weights('/home/mldev/call_sa/bert/model/model_weight_small.h5')


if __name__ == '__main__':
	main()
    