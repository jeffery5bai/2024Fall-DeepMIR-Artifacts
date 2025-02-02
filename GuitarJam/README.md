# 2024-Fall-DeepMIR-Final-Project
## Guitar Audio Generation via MusicGen: Fine-tuning Approach for High-Quality Guitar Synthesis
> *This is a school project from **DeepMIR***

Our research aims to build a guitar audio synthesizer that generate controllable, interactive guitar music for fun. We fine-tune SOTA text-to-music model, MusicGen (Copet et al., 2024) by Meta, through LoRA and other memory-efficient fine-tuning tricks on GuitarSet (Xi et al., 2018) and YouTube Music (web crawled). We mainly utilize the implementation from [musicgen-dreamboothing](https://github.com/ylacombe/musicgen-dreamboothing) GitHub repo.
For details, please take a look at our research paper `Guitar Audio Generation via MusicGen- Fine-tuning Approach for High-Quality Guitar Synthesis.pdf`

### Get Started
1. **Collect & Prepare Datasets:** run`youtube2wav.ipynb` and `prepare_datasets.ipynb` to prepare customized datasets for fune-tuning. \
You also can get the GuitarSet data using the following commands 
    ```
    !wget https://zenodo.org/records/3371780/files/audio_mono-mic.zip`
    ```
2. **Fine-tune on customized dataset**: run `finetune.ipynb` to continue train on model checkpoints


### Contribution
-----------
| Contributor | Work |
|-------------|------|
| *Jih-Ming Bai* | Problem Formulation, Literature Review, Data Collection, Model Tuning and Analysis |
| *Ting-An Yin* | Data augmentation, Model Tuning and Analysis   |
| *I-Pei Lee*    | Prompt Engineering, Model Tuning and Analysis       |