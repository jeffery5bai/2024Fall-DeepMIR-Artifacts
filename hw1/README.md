## Homework-1 Instrument activity detection

### Summary
- Leveraged pretrained MERT-v1 as feature extractor, and trained ShortChunkCNN model to predict multi-label classification.
- Implement on-the-fly data streaming with DataLoader to handle massive dataset (101 GB)
- Visualized instruments activity to compare with ground truth

### Dataset
We use Slakh to train our model. The dataset can be downloaded ([here](http://www.slakh.com/))

### Code and Files
- `DeepMIR-hw1-ModelTrain.ipynb`: Train and validate model
- `DeepMIR-hw1-Inference.ipynb`: This file is for TA to run model inference, in which you can load the model checkpoints, evaluate on test set (classification report) and generate prediction result of Test Track (in .npy format).
- `model.py`: Include **ShortChunkCNN_Res** and other source code reference from MinzWon's [repo](https://github.com/minzwon/sota-music-tagging-models)
- `plot_pianoroll.py`: a little modification in the last cell to load prediction results.

### Get started
1. install dependencies
    run `pip install -r requirements.txt` to install
2. open `DeepMIR-hw1-Inference.ipynb` notebook and click `Run All`

### Reference
- [Slakh dataset](http://www.slakh.com/)
- [MERT-v1-330M Huggingface](https://huggingface.co/m-a-p/MERT-v1-330M)
- [ShortChunkCNN](https://github.com/minzwon/sota-music-tagging-models)