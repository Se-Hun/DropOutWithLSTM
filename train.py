import torch
from torchtext import data
from torchtext.vocab import GloVe
from torchtext.datasets.imdb import IMDB
import torch.optim as optim
import torch.nn.functional as F

from dataloader import getDataIterator
from model import IMDBLstm


def fit(epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    for batch_idx, batch in enumerate(data_loader):
        text, target = batch.text, batch.label
        if torch.cuda.is_available():
            text, target = text.cuda(), target.cuda()

        if phase == 'training':
            optimizer.zero_grad()
        output = model(text)
        loss = F.nll_loss(output, target)

        running_loss += F.nll_loss(output, target, size_average=False).data
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()

    loss = running_loss / len(data_loader.dataset)
    accuracy = 100. * running_correct.item() / len(data_loader.dataset)

    print(
        f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
    return loss, accuracy

if __name__ == "__main__":
    TEXT = data.Field(lower=True, fix_length=200, batch_first=False)
    LABEL = data.Field(sequential=False, )

    train, test = IMDB.splits(TEXT, LABEL)

    TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300), max_size=10000, min_freq=10)
    LABEL.build_vocab(train, )

    train_iter, test_iter = getDataIterator(train, test)

    n_vocab = len(TEXT.vocab)
    n_hidden = 100

    model = IMDBLstm(n_vocab, n_hidden, 3, bs=32)
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [], []

    for epoch in range(1, 5):
        epoch_loss, epoch_accuracy = fit(epoch, model, train_iter, phase='training')
        val_epoch_loss, val_epoch_accuracy = fit(epoch, model, test_iter, phase='validation')
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)