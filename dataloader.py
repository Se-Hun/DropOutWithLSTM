from torchtext import data

def getDataIterator(train, test):
    train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=32, device=-1)
    train_iter.repeat = False
    test_iter.repeat = False

    return train_iter, test_iter

