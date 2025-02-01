## Homework-3 Symbolic Music Generation

### Summary
- Implemented midi tokenizers (e.g. REMI, CPWord) to represent symbolic music as tokens
- Leveraged hugging face transformers to train a piano music generator on symbolic domain
- Experimented on different generation config (model, representation, inference sampling strategy, etc.)
- Performed evaluation using Pitch-Class Histogram Entropy and Grooving Pattern Similarity 

### Dataset
We use **Pop1K7** to train piano generation model. The dataset can be download from ([here](https://zenodo.org/records/13167761)) \
note: Use `Pop1K7/midi_analyzed/*/*.mid` for training

### The File Tree

```
|
|-- train.py
|-- inference.py
|-- model.py
|-- midi2wav.ipynb
|-- requirements.txt
|-- soundfonts
|  |-- Yamaha CFX Grand.sf2
|  
|-- model_checkpoints_v3
|  |-- epoch_100.pkl
|  |-- tokenizer.pkl
|  |-- training_loss.npy
|  
|-- listening_samples
|  |-- task1
|  |  |-- 00.wav
|  |  |-- 00.mid
|  |  |-- 01.wav
|  |  |-- 01.mid
|  |  |-- ...
|  |
|  |-- task2
|  |  |-- setting_0
|  |  |  |-- song_1_generated.wav
|  |  |  |-- song_2_generated.wav
|  |  |  |-- song_3_generated.wav
|  |  |-- setting_1
|  |  |-- setting_2(best)
```

---
### Get Started

This codebase train a transformer to generate piano symbolic using Pop1K7 midi_analyzed dataset.

1. Install requirements
2. Download datasets into the same directory
3. (optional) Download soundfonts to be used at inference

**To start training, run**
```
python train.py --epochs 100 --dataset "./Pop1K7/midi_analyzed/" --ckp-folder "./model_checkpoints/"
```

**To randomly generate midi samples without prompt, run**
```
python inference.py --n-sample 20 --n-target-bar 32 --model-path "./model_checkpoints_v3/epoch_100.pkl" --output-folder "./results_v3/" --temperature 1.2 --topk 5
```

**To continue generating based on prompt midi files, run**
```
python inference.py --n-target-bar 24 --model-path "./model_checkpoints_v3/epoch_100.pkl" --prompt "./prompt_song/song_1.mid" --output-folder "./prompt_song/setting_2" --temperature 1.2 --topk 5
```

### Reference
- [Pop1K7 Dataset](https://zenodo.org/records/13167761)
- [MidiTok Github Repo](https://github.com/Natooz/MidiTok)
- [GPT2LMHeadModel Huggingface](https://huggingface.co/docs/transformers/v4.46.2/en/model_doc/gpt2#transformers.GPT2LMHeadModel)
- [MusDr Eval](https://github.com/slSeanWU/MusDr)
