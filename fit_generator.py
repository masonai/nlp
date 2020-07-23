def generator(features, labels, size):
    while True:
        start, end = 0, size
        while end < len(features):
            s = slice(start, end)
            yield features[s], labels[s]
            start, end = end, end + size


def training(tr_data, tr_label, test_data, test_label, model):
    batch_size = 16
    early_stopping = EarlyStopping(verbose=1, patience=10)
    model.fit(
        generator(tr_data, tr_label, batch_size),
        steps_per_epoch=tr_data.shape[0] // batch_size,
        epochs=100,
        verbose=1,
        validation_data=generator(test_data, test_label, batch_size),
        validation_steps=test_data.shape[0] // batch_size,
        callbacks=[early_stopping]
    )

#    model.save('./model/concat_0716v2.h5')
