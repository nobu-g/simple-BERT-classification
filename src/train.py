import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from transformers import BertTokenizer
from transformers import AdamW
from sklearn.metrics import f1_score

from modeling import BertClassifier
from dataset import LabeledDocDataset


def prepare_device(n_gpu_use):
    """
    setup GPU device if available, move model into configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
              "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


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
    parser.add_argument('--train-file', type=str,
                        help='path to train data file')
    parser.add_argument('--valid-file', type=str,
                        help='path to validation data file')
    parser.add_argument('--save-path', type=str, default='result/model.pth',
                        help='path to trained model to save')
    parser.add_argument('--num-labels', type=int, default=4,
                        help='number of document class labels')
    parser.add_argument('--max-seq-length', type=int, default=512,
                        help='The maximum total input sequence length after WordPiece tokenization.')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False, tokenize_chinese_chars=False)

    # setup data_loader instances
    train_dataset = LabeledDocDataset(args.train_file, args.max_seq_length, tokenizer)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_dataset = LabeledDocDataset(args.valid_file, args.max_seq_length, tokenizer)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device, device_ids = prepare_device(len(args.device.split(',')) if args.device else 0)

    # build model architecture
    model = BertClassifier(args.bert_model, args.num_labels)
    model.to(device)
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    # build optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    best_valid_f1 = -1

    for epoch in range(1, args.epochs + 1):
        print(f'*** epoch {epoch} ***')
        # train
        model.train()
        total_loss = 0
        total_target, total_prediction = [], []
        for batch_idx, (input_ids, segment_ids, input_mask, target) in enumerate(train_data_loader):
            input_ids = input_ids.to(device)      # (b, seq)
            segment_ids = segment_ids.to(device)  # (b, seq)
            input_mask = input_mask.to(device)    # (b, seq)
            target = target.to(device)            # (b)

            # Forward pass
            output = model(input_ids, segment_ids, input_mask)  # (b, label)
            loss = F.cross_entropy(output, target)    # ()
            prediction = torch.argmax(output, dim=1)  # (b)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_target += target.tolist()
            total_prediction += prediction.tolist()
        train_f1 = f1_score(total_target, total_prediction, average='macro')
        print(f'train_loss={total_loss / len(train_data_loader.dataset):.3f}', end=' ')
        print(f'train_f1_score={train_f1:.3f}')

        # validation
        model.eval()
        total_loss = 0
        total_target, total_prediction = [], []
        with torch.no_grad():
            for batch_idx, (input_ids, segment_ids, input_mask, target) in enumerate(valid_data_loader):
                input_ids = input_ids.to(device)      # (b, seq)
                segment_ids = segment_ids.to(device)  # (b, seq)
                input_mask = input_mask.to(device)    # (b, seq)
                target = target.to(device)            # (b)

                # Forward pass
                output = model(input_ids, segment_ids, input_mask)  # (b, label)
                loss = F.cross_entropy(output, target)    # ()
                prediction = torch.argmax(output, dim=1)  # (b)

                total_loss += loss.item()
                total_target += target.tolist()
                total_prediction += prediction.tolist()
        valid_f1 = f1_score(total_target, total_prediction, average='macro')
        print(f'valid_loss={total_loss / len(valid_data_loader.dataset):.3f}', end=' ')
        print(f'valid_f1_score={valid_f1:.3f}\n')
        if valid_f1 > best_valid_f1:
            torch.save(model.state_dict(), args.save_path)
            best_valid_f1 = valid_f1


if __name__ == '__main__':
    main()
