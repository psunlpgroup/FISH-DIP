# FISH-DIP
Source code and relevant scripts for our EMNLP 2023 paper: "Unified Low-Resource Sequence Labeling by Sample-Aware Dynamic Sparse Finetuning".

## Requirements

- Python 3.8.5
- PyTorch (tested version 1.8.1)
- Transformers (tested version 4.20.0)
- networkx 2.5

## Datasets
For downloading and preprocessing all datasets please refer to [TANL](https://github.com/amazon-science/tanl). You might need permission to gain access to some datasets.

## Running the Source Code
- For all the associated tests, run the relevant script in the [scripts](scripts) directory. 
- The scripts assume the availability of 4X 48GB GPUs (except multiwoz - 8X) to get comparable sizes as original TANL. For lower number and memory of GPUs, adjust accordingly. When changing batch sizes, you should probably also adjust the number of epochs and also expect some performance differences.
- All the finetuned output logs and output models will be available in this [link](https://pennstateoffice365-my.sharepoint.com/:f:/g/personal/sfd5525_psu_edu/EtRlflUX0pRCrcQCN-3nz9oB0OYn-BDKpr-V1-0SagUf1g?e=0MzkGs)

## Citation
```bibtex
@inproceedings{das2023unified,
  title={Unified Low-Resource Sequence Labeling by Sample-Aware Dynamic Sparse Finetuning},
  author={Das, Sarkar Snigdha Sarathi and Zhang, Ranran Haoran and Shi, Peng and Yin, Wenpeng and Zhang, Rui},
  booktitle={EMNLP},
  year={2023}
}
```