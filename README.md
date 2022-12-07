# NAS_transformer
Evolutionary Neural Architecture Search on Transformers for RUL Prediction

![check](transformer_resize.png)
This work introduces a custom genetic algorithm (GA) based neural architecture search (NAS) technique that automatically finds the optimal architectures of Transformers for RUL predictions. Our GA provides a fast and efficient search, finding high-quality solutions based on performance predictor that is updated at every generation, thus reducing the needed network training. Note that the backbone architecture is [the Transformer for RUL predictions](https://arxiv.org/abs/2106.15842) and the preformance predictor is [the NGBoost model](https://arxiv.org/abs/1910.03225)  <br/>

The proposed algorithm explores the below combinatorial parameter space defining the architecture of the Transformer model.
<p align="center">
  <img height="600" src="/params.png">
</p>

<p align="center">
  <img height="500" src="/ea_predictor_resize.png">
</p>

## Prerequisites
Our work has the following dependencies:
```bash
pip install -r requirements.txt
```

## Descriptions
- launcher.py: launcher for the experiments.
- experiments.py: Evaluation of the discovered network by ENAS-PdM on unobserved data during EA & Training.

## Run
Please launch ENAS-PdM by 
```bash
python3 launcher.py
```
After each generation, the information of the best individual is displayed
```bash
50      11      11.5005 0.153568        11.2976 11.9282
pickle dump
log saved
Best individual:
[2, 4, 1, 13, 12]
Best individual is saved
37873.60605573654
```


## References
