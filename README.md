# Mean-field parsing
Source code of "[Modeling Label Correlations for Second-Order Semantic Dependency Parsing with Mean-Filed Inference](https://arxiv.org/abs/2204.03619)

## Setup
setup environment 
```
conda create -n parsing python=3.7
conda activate parsing
while read requirement; do pip install $requirement; done < requirements.txt 
```

download dataset: [link](https://github.com/wangxinyu0922/Second_Order_Parsing/issues/1#issuecomment-894643605). 


# Run
```
python train.py +exp=ft_30  datamodule=dm model=cpd_sdp 
python train.py +exp=ft_20  datamodule=pas model=cpd_sdp 
python train.py +exp=ft_20  datamodule=psd model=cpd_sdp

Run baseline:
python train.py +exp=ft_30  datamodule=dm model=mf_sdp 
python train.py +exp=ft_20  datamodule=pas model=mf_sdp 
python train.py +exp=ft_20  datamodule=psd model=mf_sdp
```



# Contact
Feel free to contact bestsonta@gmail.com if you have any questions.

# Citation
```


@misc{yang2022modeling,
      title={Modeling Label Correlations for Second-Order Semantic Dependency Parsing with Mean-Filed Inference}, 
      author={Songlin Yang and Kewei Tu},
      year={2022},
      eprint={2204.03619},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

# Acknowledge
The code is based on [lightning+hydra](https://github.com/ashleve/lightning-hydra-template) template. I use [FastNLP](https://github.com/fastnlp/fastNLP) as dataloader. I use lots of built-in modules (LSTMs, Biaffines, Triaffines, Dropout Layers, etc) from [Supar](https://github.com/yzhangcs/parser/tree/main/supar).  



