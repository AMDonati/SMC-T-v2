import tensorflow as tf
# building the LSTM with keras functional API for MC Dropout: https://www.tensorflow.org/guide/keras/functional

def build_GRU_for_classification(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)])
  return model

def build_LSTM_for_regression(shape_input_1, shape_input_2, shape_output, rnn_units, dropout_rate, rnn_drop_rate=0.0, training=True):
  inputs = tf.keras.Input(shape=(shape_input_1, shape_input_2))
  h = tf.keras.layers.LSTM(rnn_units, recurrent_dropout=rnn_drop_rate, return_sequences=True)(inputs, training=training)
  outputs = tf.keras.layers.Dropout(rate=dropout_rate)(h, training=training)
  outputs = tf.keras.layers.Dense(shape_output)(outputs)
  lstm_model = tf.keras.Model(outputs=outputs, inputs=inputs, name='lstm_for_regression')

  return lstm_model

def build_LSTM_for_classification(seq_len, emb_size, shape_output, rnn_units, dropout_rate, rnn_drop_rate=0.0, training=True):
  inputs = tf.keras.Input(shape=(seq_len,)) #TODO: resolve this problem here. Use tf.keras.Sequential instead (cf SMC-T code.)
  embedding = tf.keras.layers.Embedding(input_dim=shape_output, output_dim=emb_size)(inputs)
  h = tf.keras.layers.LSTM(rnn_units, recurrent_dropout=rnn_drop_rate, return_sequences=True)(embedding, training=training)
  outputs = tf.keras.layers.Dropout(rate=dropout_rate)(h, training=training)
  outputs = tf.keras.layers.Dense(shape_output)(outputs)
  lstm_model = tf.keras.Model(outputs=outputs, inputs=inputs, name='lstm_for_classification')

  return lstm_model

if __name__ == '__main__':
    import numpy as np
    print("test LSTM for classification")
    X = tf.constant(np.random.randint(0, 50, size=(8, 30)))
    #X = tf.expand_dims(X, axis=-1)
    lstm_classif = build_LSTM_for_classification(seq_len=30, emb_size=8, shape_output=50, rnn_units=16, dropout_rate=0.0)
    outputs = lstm_classif(X)


    train_dataset = tf.random.uniform(shape=(8, 25, 3))
    train_data = train_dataset[:, :-1, :]
    train_labels = tf.expand_dims(train_dataset[:, 1:, 0], axis=-1)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    train_dataset = train_dataset.batch(8, drop_remainder=True)
    val_dataset = tf.random.uniform(shape=(8, 25, 3))
    val_data = val_dataset[:, :-1, :]
    val_labels = tf.expand_dims(val_dataset[:, 1:, 0], axis=-1)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
    val_dataset = val_dataset.batch(8, drop_remainder=True)

    shape_input_1 = tf.shape(train_data)[1].numpy()
    shape_input_2 = tf.shape(train_data)[-1].numpy()
    shape_output = 3
    rnn_units = 20
    dropout_rate = 0.1
    rnn_drop_rate = 0.1

    model = build_LSTM_for_regression(shape_input_1=shape_input_1,
                                      shape_input_2=shape_input_2,
                                      shape_output=shape_output,
                                      rnn_units=rnn_units,
                                      dropout_rate=dropout_rate,
                                      rnn_drop_rate=rnn_drop_rate,
                                      training=True)

    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer,
                  loss='mse')
    model.summary()
    # start_training = time.time()
    EPOCHS = 2
    rnn_history = model.fit(train_dataset,
                            epochs=EPOCHS,
                            validation_data=val_dataset,
                            verbose=2)
    predictions_val = model(val_data)
    print('done')