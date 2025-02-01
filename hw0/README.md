## Homework-0 Musical note classification

### Summary
- Performed preprocessing and transformation on audio files using Librosa and torchaudio
- Implemented RandomForest and ShortChunkCNN_res, and analyze model performance

### Dataset
We use NSynth dataset to train our model. The dataset can be downloaded from ([here](https://magenta.tensorflow.org/datasets/nsynth)) \
note: 289,205 for training, 12,678 for validation, 4,096 for testing

### Code and Files
- `DeepMIR-hw0-Visualization.ipynb`: Task 1
- `DeepMIR-hw0-MLTrain.ipynb`: Task 2
- `DeepMIR-hw0-DLTrain.ipynb`: Task 3
- `DeepMIR-hw0-Inference.ipynb`: This file is for TA to run model inference, in which you can load the model checkpoints (both ML & DL) and evaluate on test set (Top1 Acc, Top3 Acc, Confusion Matrix).
- `model.py`: Include **ShortChunkCNN_Res** and other source code reference from MinzWon's [repo](https://github.com/minzwon/sota-music-tagging-models)

### Get started
1. install dependencies
    run `pip install -r requirements.txt` to install
2. open `DeepMIR-hw0-Inference.ipynb` notebook and click `Run All`

### Reference
- [NSynth Dataset](https://magenta.tensorflow.org/datasets/nsynth)
- [ShortChunkCNN_res from MinzWon Github Repo](https://github.com/minzwon/sota-music-tagging-models)
