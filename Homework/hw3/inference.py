import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import tqdm
from model import X_LEN, GPT2Model

warnings.filterwarnings("ignore")


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help="the checkpoint path.")
    parser.add_argument("--output-folder", type=str, help="the result path.", default="./results/")
    parser.add_argument("--prompt", type=str, default=None, help="the prompt file path.")
    parser.add_argument("--n-sample", type=int, default=20, metavar="S", help="num of samples to generate")
    parser.add_argument("--device", type=str, help="gpu device.", default="cuda")
    parser.add_argument("--seed", type=int, default=42, metavar="S", help="random seed (default: 42)")
    parser.add_argument("--temperature", type=float, default=1, help="inference temperature")
    parser.add_argument(
        "--n-target-bar", type=int, default=32, metavar="S", help="num of target bar to generate"
    )
    parser.add_argument("--topk", type=int, default=5, metavar="S", help="topk in sampling")

    args = parser.parse_args()
    return args


args = parse_arg()


def temperature_sampling(logits, temperature, topk):
    #################################################
    # Ref: https://github.com/YatingMusic/remi/blob/6d407258fa5828600a5474354862353ef4e4e8ae/model.py#L104
    # 1. adjust softmax with the temperature parameter
    # 2. choose top-k highest probs
    # 3. normalize the topk highest probs
    # 4. random choose one from the top-k highest probs as result by the probs after normalize
    #################################################
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    if topk == 1:
        prediction = np.argmax(probs)
    else:
        sorted_index = np.argsort(probs)[::-1]
        candi_index = sorted_index[:topk]
        candi_probs = [probs[i] for i in candi_index]
        # normalize probs
        candi_probs /= sum(candi_probs)
        # choose by predicted probs
        prediction = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return prediction


def test(
    n_target_bar=32, temperature=1.2, topk=5, output_path="./results/test.mid", model_path="", prompt=None
):
    with torch.no_grad():
        # load tokenizer and model
        tokenizer_ckp = torch.load("./model_checkpoints_v3/tokenizer.pkl")
        tokenizer = tokenizer_ckp["tokenizer"]
        event2word = tokenizer.vocab
        checkpoint = torch.load(model_path, map_location=args.device)
        model_state_dict = {}
        for key, value in checkpoint["model"].items():
            new_key = key.replace("module.", "")  # Remove "module." prefix
            model_state_dict[new_key] = value

        model = GPT2Model(tokenizer).to(args.device)
        model.load_state_dict(model_state_dict)
        model.eval()

        batch_size = 1
        words = []

        if prompt:
            # If prompt, load prompt file, tokenize it.
            tok_sequence = tokenizer(prompt)[0]
            ids = tok_sequence.ids
            words.append(ids)

        else:
            # Or, random select prompt to start
            for _ in range(batch_size):
                ws = [event2word["Bar_None"]]
                tempo_values = [v for k, v in event2word.items() if k.startswith("Tempo")]
                chords = [v for k, v in event2word.items() if k.startswith("Chord")]
                ws.append(event2word["Position_0"])
                ws.append(np.random.choice(chords))
                ws.append(event2word["Position_0"])
                ws.append(np.random.choice(tempo_values))
                words.append(ws)

        # generate
        original_length = len(words[0])
        initial_flag = 1
        current_generated_bar = 0
        cnt = 0
        print("Start generating")
        while current_generated_bar < n_target_bar:
            print("\r", current_generated_bar, end="")
            # input
            if initial_flag:
                temp_x = np.zeros((batch_size, original_length))
                for b in range(batch_size):
                    for z, t in enumerate(words[b]):
                        temp_x[b][z] = t
                initial_flag = 0
            else:
                temp_x_new = np.zeros((batch_size, 1))
                for b in range(batch_size):
                    temp_x_new[b][0] = words[b][-1]
                temp_x = np.array([np.append(temp_x[0], temp_x_new[0])])

            temp_x = torch.Tensor(temp_x).long()

            output_logits = model(temp_x[:, -X_LEN:].to(args.device))

            # sampling
            _logit = output_logits[0, -1].to("cpu").detach().numpy()
            word = temperature_sampling(logits=_logit, temperature=temperature, topk=topk)

            words[0].append(word)

            # stop
            cnt += 1
            if word == event2word["Bar_None"]:
                current_generated_bar += 1
                cnt = 0

            if cnt == 50:
                words[0].append(event2word["Bar_None"])
                current_generated_bar += 1
                cnt = 0

        generated_tokens = np.array(words)
        generated_midi = tokenizer(generated_tokens)
        generated_midi.dump_midi(output_path)


def main():
    # check path folder
    os.makedirs(args.output_folder, exist_ok=True)

    if args.prompt:
        filename = (args.prompt.split('/')[-1]).split('.')[0]
        output_path = Path(args.output_folder, f"{filename}_generated.mid")
        test(
            n_target_bar=args.n_target_bar,
            temperature=args.temperature,
            topk=args.topk,
            output_path=output_path,
            model_path=args.model_path,
            prompt=args.prompt,
        )
        pass
    else:
        for i in tqdm.tqdm(range(args.n_sample)):
            output_path = Path(args.output_folder, f"{i:02d}.mid")
            test(
                n_target_bar=args.n_target_bar,
                temperature=args.temperature,
                topk=args.topk,
                output_path=output_path,
                model_path=args.model_path,
                prompt=None,
            )
    return


if __name__ == "__main__":
    main()
