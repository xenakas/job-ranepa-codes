from typing import Optional, Tuple
import time
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
from torch.autograd import Variable

from prepare_auto_ru_dataset_for_slot_filling import prepare_auto_ru_dataset
from sklearn.metrics import f1_score
torch.manual_seed(1)
now = datetime.now()
now = "_".join([str(n_date) for n_date in [now.year, now.month, now.day]])
model_name = ""


class CNN(nn.Module):
    def __init__(
            self,
            kernel_sizes: Tuple[int] = (3, 4, 5),
            num_filters: int = 100,
            vocab_size: int = 1000,
            embedding_dim: int = 50,
            embedding_mtx: Optional[list] = None,
            pretrained_embeddings: Optional[np.ndarray] = None,
            use_cuda: bool = True,
            trainable_embeddings: bool = False,
            sentence_len: int = 1000,
            output_chars: int = 50, ):

        super(CNN, self).__init__()
        self.kernel_sizes = kernel_sizes

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(
            torch.from_numpy(pretrained_embeddings))
        self.embedding.weight.requires_grad = trainable_embeddings

        if use_cuda:
            self.embedding = self.embedding.cuda()

        conv_blocks = []
        print(kernel_sizes)
        for kernel_size in kernel_sizes:
            # maxpool kernel_size must <= sentence_len - kernel_size+1,
            # otherwise, it could output empty
            maxpool_kernel_size = sentence_len - kernel_size + 1

            conv1d = nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=kernel_size,
                stride=1)
            component = nn.Sequential(
                conv1d,
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=maxpool_kernel_size)
            )
            if use_cuda:
                component = component.cuda()
            conv_blocks.append(component)

        # ModuleList is needed for registering parameters in conv_blocks
        self.conv_blocks = nn.ModuleList(conv_blocks)
        # if not one-hot
        # self.fc = nn.Linear(num_filters * len(kernel_sizes),
        #                     output_chars)
        # one-hot
        self.fc = nn.Linear(num_filters * len(kernel_sizes),
                            output_chars * vocab_size)

    def forward(self, x):       # x: (batch, sentence_len)
        # embedded x: (batch, sentence_len, embedding_dim)
        x = self.embedding(x)

        # input:  (batch, in_channel=1, in_length=sentence_len*embedding_dim),
        # output: (batch, out_channel=num_filters, out_length=sentence_len-...)
        # needs to convert x to (batch, embedding_dim, sentence_len)
        x = x.transpose(1, 2)

        x_list = [conv_block(x) for conv_block in self.conv_blocks]
        out = torch.cat(x_list, 2)
        out = out.view(out.size(0), -1)
        feature_extracted = out
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.fc(out)
        # out = F.softmax(out, dim=1)
        return out, feature_extracted


def compute_loss(model, loss_fn, inputs, labels, optimizer=None, train=True):
    inputs, labels = Variable(inputs), Variable(labels)
    inputs, labels = inputs.cuda(), labels.cuda()

    preds, _ = model(inputs)
    preds = preds.cuda()

    loss = loss_fn(preds.double(), labels.double())
    if train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss = float(loss)
    f1 = f1_score(
        torch.sigmoid(preds).data.cpu() > 0.5,
        labels.cpu(), average='samples')
    f1 = round(f1, 3)
    loss = round(loss, 8)
    return loss, f1


def train():
    dataset_train, dataset_test, emb_matrx = prepare_auto_ru_dataset()
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 4}
    max_epochs = 100
    training_generator = data.DataLoader(dataset_train, **params)
    test_generator = data.DataLoader(dataset_test, **params)
    output_len = dataset_train.Y.shape[1]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN(pretrained_embeddings=emb_matrx, output_chars=output_len)
    model = model.to(device)

    adam_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(adam_params, lr=0.0002)
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MultiLabelSoftMarginLoss()
    train_len = int(len(dataset_train.X) / params["batch_size"]) + 1
    test_len = int(len(dataset_test.X) / params["batch_size"]) + 1

    min_val_loss: int = None
    tolerance_left = 5
    for epoch in range(max_epochs):
        tic = time.time()
        avg_train_loss = 0
        model.train()
        # inputs, labels = next(iter(training_generator))
        # Train
        print("Train\n")
        for i, (inputs, labels) in enumerate(training_generator):
            # Transfer to GPU
            # labels = labels.long()
            loss, f1 = compute_loss(model, loss_fn, inputs, labels, optimizer)
            print(
                f"Epoch: {epoch}; sample {i} out of {train_len} samples; "
                f"F1: {f1}; Loss: {loss}", end="\r")
            avg_train_loss += loss
        # var `i` is left from the previous loop
        avg_train_loss = avg_train_loss / i + 1
        tic = time.time() - tic
        tic = round(tic, 2)
        print(f"\navg loss: {avg_train_loss}; time elapsed: {tic} secs\n")
        # TODO
        """
        сделать нормальную модель
        """
        # Validation
        avg_f1 = 0
        with torch.set_grad_enabled(False):
            avg_val_loss = 0
            print("Validation\n")
            for i, (inputs, labels) in enumerate(test_generator):
                # Transfer to GPU
                loss, f1 = compute_loss(
                    model, loss_fn, inputs, labels, train=False)
                print(
                    f"Epoch: {epoch}; sample {i} out of {test_len} samples; "
                    f"F1: {f1}; Loss: {loss}", end="\r")
                avg_val_loss += loss
                avg_f1 += f1
            # i is saved from the loop
            avg_val_loss /= i + 1
            avg_f1 /= i + 1
        print(f"\navg_val_loss: {avg_val_loss}; avg_f1: {avg_f1}\n")
        if epoch != 0:
            if avg_val_loss > min_val_loss:
                tolerance_left -= 1
                if tolerance_left <= 0:
                    print("The training is finished")
                    break
            else:
                min_val_loss = avg_val_loss
        else:
            min_val_loss = avg_val_loss
        if model_name:
            local_name = model_name
        else:
            local_name = "default"
        torch.save(model.state_dict(), f"nn_models/{now}_{local_name}")
    # #Later to restore:


if __name__ == "__main__":
    train()
