#-*-coding:utf-8-*-
import tensorflow as tf
from transformers import TFBertModel
from tokenization_kobert import KoBertTokenizer

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


SEQ_LEN = 200

bert_dir = '/home/mldev/call_sa/model/baseline/bert/'
tokenizer = KoBertTokenizer.from_pretrained(bert_dir)

def convert_data(test_sentence):
    tokens, masks, segments = [], [], []

    token = tokenizer.encode(test_sentence, max_length=SEQ_LEN, pad_to_max_length=True)

    num_zeros = token.count(0)
    mask = [1]*(SEQ_LEN-num_zeros) + [0]*num_zeros
    segment = [0]*SEQ_LEN

    tokens.append(token)
    segments.append(segment)
    masks.append(mask)

    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)
    return [tokens, masks, segments]

def main():
    # Model
    #model = TFBertModel.from_pretrained("monologg/kobert", from_pt=True)
    model = TFBertModel.from_pretrained(bert_dir, from_pt=True)

    token_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_word_ids')
    mask_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_masks')
    segment_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_segment')

    bert_outputs = model([token_inputs, mask_inputs, segment_inputs])[1]

    sentiment_first = tf.keras.layers.Dense(2, activation='softmax')(bert_outputs)
    #sentiment_first = tf.keras.layers.Dense(1, activation='sigmoid')(bert_outputs)
    sentiment_model = tf.keras.Model([token_inputs, mask_inputs, segment_inputs], sentiment_first)

    test_str = '너무 좋아요'
    input_list = convert_data(test_str)
    pred = sentiment_model.predict(input_list)

    print(test_str, ' ' , pred)


if __name__ == '__main__':
    main()
    
