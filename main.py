import os
import argparse

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from model import Net
from utils import AverageMeter
from tensorboardX import SummaryWriter


class DisturbLabel(torch.nn.Module):
    def __init__(self, alpha, C):
        super(DisturbLabel, self).__init__()
        self.alpha = alpha
        self.C = C
        # Multinoulli distribution
        self.p_c = (1 - ((C - 1)/C) * (alpha/100))
        self.p_i = (1 / C) * (alpha / 100)

    def forward(self, y):
        # convert classes to index
        y_tensor = y
        y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)

        # create disturbed labels
        depth = self.C
        y_one_hot = torch.ones(y_tensor.size()[0], depth) * self.p_i
        y_one_hot.scatter_(1, y_tensor, self.p_c)
        y_one_hot = y_one_hot.view(*(tuple(y.shape) + (-1,)))

        # sample from Multinoulli distribution
        distribution = torch.distributions.OneHotCategorical(y_one_hot)
        y_disturbed = distribution.sample()
        y_disturbed = y_disturbed.max(dim=1)[1]  # back to categorical

        return y_disturbed


def main():
    # parameters
    parser = argparse.ArgumentParser(description='PyTorch DisturbLabel')
    parser.add_argument('--mode', type=str, default='bothreg')
    parser.add_argument('--alpha', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    global writer
    writer = SummaryWriter(os.path.join('logs', args.mode, 'tb'))

    # GPU/CPU
    device = torch.device('cuda' if args.device == 'gpu' else 'cpu')

    # Reading MNIST
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                           ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    # Model
    model = Net(args.mode).to(device)

    # Optimizer + Loss
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss().to(device)
    disturb = None
    if args.mode == 'disturblabel' or args.mode == 'bothreg':
        disturb = DisturbLabel(alpha=args.alpha, C=10)

    # Train and Test
    for epoch in range(1, args.epochs + 1):
        torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60, 80], gamma=0.1)
        train(args, model, device, train_loader, optimizer, criterion, epoch, disturb)
        test(args, model, device, test_loader, criterion, epoch)


def train(args, model, device, train_loader, optimizer, criterion, epoch, disturb):
    model.train()
    correct = 0
    losses = AverageMeter()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # disturb labels
        if args.mode == 'disturblabel' or args.mode == 'bothreg':
            target = disturb(target).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # calculate error rate
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        losses.update(loss.item(), data.size(0))

    train_error = 100 - (100. * correct / len(train_loader.dataset))
    print('Epoch [{0}] Train Loss: {1:.4f} | Error: {2:.2f}%'.format(epoch, losses.avg, train_error))
    writer.add_scalar('{0}/train_error'.format(args.mode), train_error, epoch)
    writer.add_scalar('{0}/train_loss'.format(args.mode), losses.avg, epoch)


def test(args, model, device, test_loader, criterion, epoch):
    model.eval()
    correct = 0
    losses = AverageMeter()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)  # sum up batch loss

            # calculate error rate
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            losses.update(loss.item(), data.size(0))

        test_error = 100 - (100. * correct / len(test_loader.dataset))
        print('Epoch [{0}] Test Loss: {1:.4f} | Error: {2:.2f}%\n'.format(epoch, losses.avg, test_error))
        writer.add_scalar('{0}/test_error'.format(args.mode), test_error, epoch)
        writer.add_scalar('{0}/test_loss'.format(args.mode), losses.avg, epoch)


if __name__ == '__main__':
    main()

