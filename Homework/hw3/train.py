import argparse
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import tqdm
from miditok import REMI, CPWord, TokenizerConfig
from model import X_LEN, GPT2Model
from torch import nn
from torch.utils.data.dataloader import DataLoader, Dataset

warnings.filterwarnings("ignore")


BATCH_SIZE = 8
SRC_FILE_IDX = [1, 2, 3]


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, help="the midi dataset path.", default="./Pop1K7/midi_analyzed/"
    )
    parser.add_argument(
        "--ckp-folder", type=str, help="the checkpoint folder.", default="./model_checkpoints_v2/"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, help="gpu device.", default="cuda")
    parser.add_argument("--seed", type=int, default=42, metavar="S", help="random seed (default: 42)")

    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="less verbose during training",
    )
    parser.add_argument(
        "--is-continue",
        action="store_true",
        default=False,
        help="continue training from model checkpoint",
    )
    parser.add_argument("--checkpoints", type=str, help="the checkpoint to continue.", default="")

    args = parser.parse_args()
    return args


args = parse_arg()


class Pop1K7Dataset(Dataset):
    def __init__(self, midi_dataset_path="./Pop1K7/midi_analyzed/", prompt=""):
        self.x_len = X_LEN
        self.tokenizer = None
        self.midi_paths = []
        for i in SRC_FILE_IDX:
            self.midi_paths += list(Path(midi_dataset_path).glob(f"src_00{i}/*.mid"))
        self.parser = self.prepare_data(self.midi_paths)

    def get_tokenizer(self):
        if self.tokenizer is None:
            raise NotImplementedError("The tokenizer has not been initialized")
        return self.tokenizer

    def __len__(self):
        return len(self.parser)

    def __getitem__(self, index):
        return self.parser[index]

    def initialize_tokenizer(self):
        if self.tokenizer is not None:
            print("The tokenizer has already been initialized.")

        TOKENIZER_PARAMS = {
            "pitch_range": (21, 109),
            "beat_res": {(0, 4): 8, (4, 12): 4},
            "num_velocities": 50,
            "special_tokens": ["PAD", "BOS", "EOS", "MASK", "OOV"],
            "use_chords": True,
            "use_rests": False,
            "use_tempos": True,
            "use_time_signatures": False,
            "use_programs": False,
            "num_tempos": 32,  # nb of tempo bins
            "tempo_range": (40, 250),  # (min, max)
            "rest_range": (2, 8),  # (half, 8 beats)
        }
        config = TokenizerConfig(**TOKENIZER_PARAMS)
        self.tokenizer = REMI(config)  # REMI encoding
        os.makedirs(args.ckp_folder, exist_ok=True)
        torch.save(
            {
                "tokenizer": self.tokenizer,
                "tokenizer_params": TOKENIZER_PARAMS,
            },
            os.path.join(args.ckp_folder, "tokenizer.pkl"),
        )

    def prepare_data(self, midi_paths):
        """extract events(tokens) and words(ids)"""
        if self.tokenizer is None:
            self.initialize_tokenizer()

        all_tokens = []
        all_ids = []
        pbar = tqdm.tqdm(midi_paths, disable=False)  # disable=args.quiet
        pbar.set_description("Tokenizing MIDI Files")
        for path in pbar:
            tok_sequence = self.tokenizer(path)[0]
            tokens = tok_sequence.tokens
            ids = tok_sequence.ids
            all_tokens.append(tokens)
            all_ids.append(ids)

        # all_ids is a list containing ids list of all midi files
        # all_ids = [[token_ids of midi], [token_ids of midi], ...]

        # TODO: cut the data into what you want to feed into model
        # Warning : this example cannot use in transformer_XL, you must implement group segments by yourself
        segments = []
        pbar = tqdm.tqdm(all_ids, disable=False)  # disable=args.quiet
        pbar.set_description("Preparing Sample Pairs")
        for words in pbar:
            pairs = []
            for i in range(0, len(words) - self.x_len - 1, self.x_len):
                x = words[i : i + self.x_len]
                y = words[i + 1 : i + self.x_len + 1]
                pairs.append([x, y])
            # abandon last segments in a midi
            pairs = pairs[0 : len(pairs) - (len(pairs) % 5)]
            segments = segments + pairs
        segments = torch.tensor(segments)
        print(segments.shape)
        return segments


def train(is_continue=False, checkpoints_path=""):
    epochs = args.epochs

    # create data list
    # use glob to get all midi file path
    midi_dataset_path = args.dataset

    # dataset
    train_dataset = Pop1K7Dataset(midi_dataset_path)
    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Dataloader is created")

    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    tokenizer = train_dataset.get_tokenizer()
    # create model
    if not is_continue:
        start_epoch = 1
        model = GPT2Model(tokenizer).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    else:
        # wheather checkpoint_path is exist
        if os.path.isfile(checkpoints_path):
            checkpoint = torch.load(checkpoints_path)
            model_state_dict = {}
            for key, value in checkpoint["model"].items():
                new_key = key.replace("module.", "")  # Remove "model." prefix
                model_state_dict[new_key] = value

        else:
            os._exit()
        start_epoch = checkpoint["epoch"] + 1

        model = GPT2Model(tokenizer).to(device)
        model.load_state_dict(model_state_dict)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.load_state_dict(checkpoint["optimizer"])

    print("Model is created \nStart training")
    model = torch.nn.DataParallel(model)
    model.train()
    losses = []
    try:
        os.makedirs(args.ckp_folder, exist_ok=True)
        print("dir is created")
    except:
        pass

    t = tqdm.trange(start_epoch, start_epoch + epochs + 1)
    for epoch in t:
        t.set_description("Training Epochs")
        single_epoch = []
        pbar = tqdm.tqdm(train_dataloader, disable=args.quiet)
        for i in pbar:
            pbar.set_description("Training batch")
            # x, y shape = (batch_size, length)
            x = i[:, 0, :].to(device).long()
            y = i[:, 1, :].to(device).long()
            optimizer.zero_grad()
            output = model(x)
            # loss = nn.CrossEntropyLoss()(output.permute(0, 2, 1), y)
            loss = nn.CrossEntropyLoss()(output.permute(0, 2, 1)[..., -1], y[..., -1])
            loss.backward()
            single_epoch.append(loss.to("cpu").mean().item())
            optimizer.step()
            pbar.set_postfix(loss="{:.3f}".format(np.mean(single_epoch)))
        single_epoch = np.array(single_epoch)
        losses.append(single_epoch.mean())
        t.set_postfix(epoch=epoch, loss="{:.5f}".format(losses[-1]))
        # print('>>> Epoch: {}, Loss: {:.5f}'.format(epoch,losses[-1]))
        if epoch <= 5 or epoch % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": losses[-1],
                },
                os.path.join(args.ckp_folder, "epoch_%03d.pkl" % epoch),
            )
            np.save(os.path.join(args.ckp_folder, "training_loss"), np.array(losses))


def main():
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    train(is_continue=args.is_continue, checkpoints_path=args.checkpoints)
    return


if __name__ == "__main__":
    main()
