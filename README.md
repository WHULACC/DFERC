# Disentanglement & Fusion on Multimodal Emotion Recognition in Conversation

This repository contains the code for the ACM MM 2023 research paper "Revisiting Disentanglement and Fusion on Modality and Context in Conversational Multimodal Emotion Recognition."

### Data Preparation 

You can download the IEMOCAP dataset from the [official website](https://github.com/declare-lab/dialogue-understanding/blob/master/glove-end-to-end/datasets/_original/iemocap/IEMOCAP_features_raw.pkl).
And then name the file `IEMOCAP_features_raw.pkl` and place it in the `data/preprocessed` folder.

### Running the Model

You can clone the repository and install the necessary dependencies by executing:

`pip install -r requirements.txt`

Run the following command from the project's root directory to train and test the model:

`python main.py`


This directory is actively being updated...

## Citation
If you use this repository, please kindly cite the following paper:

```bib
@inproceedings{rd23mm,
  author       = {Bobo Li and Hao Fei and Lizi Liao and Yu Zhao and Chong Teng and Tat{-}Seng Chua and Donghong Ji and Fei Li},
  title        = {Revisiting Disentanglement and Fusion on Modality and Context in Conversational
                  Multimodal Emotion Recognition},
  booktitle    = {Proceedings of ACM MM},
  pages        = {5923--5934},
  year         = {2023}
}
```