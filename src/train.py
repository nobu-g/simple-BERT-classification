import os
from argparse import ArgumentParser

import torch
from torch.utils.data.dataloader import DataLoader
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from modeling import BertClassifier
from data_loader import LabeledDocDataset


def main():
    # Training settings
    parser = ArgumentParser()
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: None)')
    parser.add_argument('-b', '--batch-size', type=int, default=1024,
                        help='number of batch size for training')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--bert-model', type=str,
                        help='path to BERT model directory')
    parser.add_argument('--save-path', type=str, default='result/model.pth',
                        help='path to trained model to save')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.device is not None else 'cpu')

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    # setup data_loader instances
    train_dataset = LabeledDocDataset(tokenizer)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # build model architecture
    model = BertClassifier(args.bert_model, 4)
    model.to(device)

    # build optimizer
    optimizer = BertAdam(model.parameters(), lr=args.lr)

    best_valid_acc = -1

    for epoch in range(1, args.epochs + 1):
        print(f'*** epoch {epoch} ***')
        # train
        model.train()
        total_loss = 0
        total_correct = 0
        for batch_idx, (source, mask, target) in enumerate(train_data_loader):
            source = source.to(device)      # (b, len)
            mask = mask.to(device)          # (b, len)
            target = target.to(device)      # (b)

            # Forward pass
            output = model(source, mask)    # (b, 2)
            loss = loss_fn(output, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += metric_fn(output, target)
        print(f'train_loss={total_loss / train_data_loader.n_samples:.3f}', end=' ')
        print(f'train_accuracy={total_correct / train_data_loader.n_samples:.3f}')

        # validation
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_correct = 0
            for batch_idx, (source, mask, target) in enumerate(valid_data_loader):
                source = source.to(device)    # (b, len)
                mask = mask.to(device)        # (b, len)
                target = target.to(device)    # (b)

                output = model(source, mask)  # (b, 2)

                total_loss += loss_fn(output, target)
                total_correct += metric_fn(output, target)
        valid_acc = total_correct / valid_data_loader.n_samples
        print(f'valid_loss={total_loss / valid_data_loader.n_samples:.3f}', end=' ')
        print(f'valid_accuracy={valid_acc:.3f}\n')
        if valid_acc > best_valid_acc:
            torch.save(model.state_dict(), args.save_path)
            best_valid_acc = valid_acc


if __name__ == '__main__':
    main()
