## Homework-2 Source Separation

### Summary
- Leveraged open source code to implement open-unmix source separation model
- Experimented on different model config (e.g. data augmentation, LSTM vs. BiLSTM, predicting phase with Griffin-Lim algorithm)
- Evaluated performance using SDR, Source-to-Distortion Ratio (median of median)

### Dataset
We use MUSDB18-HQ (.wav) (22.7GB) to train our model (Train: 100 tracks, Test: 50 tracks). \
The dataset can be downloaded from ([here](https://zenodo.org/records/3338373))

### The File Tree
```
|
|-- requirements.txt
|-- wav2npy.ipynb
|-- open-unmix-pytorch (main codebase, modified from `openunmix` repo)
|  |
|  |-- open-unmix **(best model chkpnt, config)**
|  |-- open-unmix-epoch25 
|  |-- open-unmix-epoch50
|  |-- open-unmix-epoch150 
|  |-- open-unmix-estimate-25 **(listening samples)**
|  |-- open-unmix-estimate-50
|  |-- open-unmix-estimate-150
|  |-- open-unmix-estimate-griffinlim 
|  |
|  |-- scripts
|  |  |-- train.py **(code to run training)**
|  |-- openunmix
|  |  |-- evaluate.py **(code to run source separation and perform evaluation)**
|  |  |-- ...
```
---

### Get started
This codebase is mainly inhereted from the public code `open-unmix-pytorch` repo. (https://github.com/sigsep/open-unmix-pytorch/tree/master)
I modified it to fit in my own dataset.
To be more specific, the revised files are: 
- `data.py`: create and load preprocessed numpy datasets
- `model.py`: add flags and griffinlim logic for Separator
- `train.py`: load preprocessed train/valid datasets and exclude encoder from training
- `evaluate.py`: minor fix to run on local directory
- `utils.py`: add griffinlim arguments at funcion calling

**To get the preprocessed npy data (data augmentation, random chunk, stft and complexnorm transformation)**
run `wav2npy.ipynb` (takes about 15 mins for train set, on GPU with nb_workers=4)

**To run inference code on single audio input**
run `inference.ipynb`

**To run training and evaluation with command line:**
First, go to project directory
```cd open-unmix-pytorch```

1. To start training, run
```
python scripts/train.py --epochs 150 --is-wav --root "../musdb18hq/" --np-root "../musdb18hq_np" --nb-workers 4 --output open-unmix
```

2. To evaluate on test set, run
```
python openunmix/evaluate.py --targets "vocals" --residual true --root "../musdb18hq" --is-wav --model "open-unmix" --outdir "open-unmix-estimate" --evaldir "open-unmix-eval"
```

3. To evaluate on test set with griffinlim algo, run
```
python openunmix/evaluate.py --targets "vocals" --residual true --griffinlim --root "../musdb18hq" --is-wav --model "open-unmix" --outdir "open-unmix-estimate-griffinlim" --evaldir "open-unmix-eval-griffinlim"
```


### Reference
- [MUSDB18-HQ Dataset](https://zenodo.org/records/3338373)
- [Open-Unmix Github Repo](https://github.com/sigsep/open-unmix-pytorch)
