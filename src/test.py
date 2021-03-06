import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import f1_score

from modeling import BertClassifier
from dataset import LabeledDocDataset
from train import prepare_device


def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: None)')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help='number of batch size for training')
    parser.add_argument('--bert-model', type=str,
                        help='path to BERT model directory')
    parser.add_argument('--test-file', type=str,
                        help='path to test data file')
    parser.add_argument('--load-path', type=str, default='result/model.pth',
                        help='path to trained model to load')
    parser.add_argument('--num-labels', type=int, default=4,
                        help='number of document class labels')
    parser.add_argument('--max-seq-length', type=int, default=512,
                        help='The maximum total input sequence length after WordPiece tokenization.')
    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    # setup data_loader instance
    test_dataset = LabeledDocDataset(args.test_file, args.max_seq_length, tokenizer)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device, device_ids = prepare_device(len(args.device.split(',')))

    # build model architecture
    model = BertClassifier(args.bert_model, args.num_labels)
    # load state dict
    state_dict = torch.load(args.load_path, map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    model.to(device)
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    model.eval()
    total_loss = 0
    total_target, total_prediction = [], []
    with torch.no_grad():
        for batch_idx, (input_ids, segment_ids, input_mask, target) in enumerate(test_data_loader):
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
    test_f1 = f1_score(total_target, total_prediction, average='macro')
    print(f'test_loss={total_loss / len(test_data_loader.dataset):.3f}', end=' ')
    print(f'test_f1_score={test_f1:.3f}')


if __name__ == '__main__':
    main()
