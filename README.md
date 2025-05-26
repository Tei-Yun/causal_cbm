# Causally Reliable Concept Bottleneck Models

[![PDF](https://img.shields.io/badge/%E2%87%A9-PDF-orange.svg?style=flat-square)](https://arxiv.org/pdf/2503.04363?)

**Authors**: [Giovanni De Felice](mailto:giovanni.de.felice@usi.ch), Arianna Casanova Flores, Francesco De Santis, Silvia Santini, Johannes Schneider, Pietro Barbiero, Alberto Termine

---

We propose Causally reliable Concept Bottleneck Models (C2BMs), a class of concept-based architectures that enforce reasoning through a bottleneck of concepts structured according to a model of the real-world causal mechanisms. We also introduce a pipeline to automatically learn this structure from observational data and unstructured background knowledge (e.g., scientific literature). 

<p align="center">
<img src="https://github.com/gdefe/causally-reliable-cbm/blob/main/visual_abstract.pdf" style="width: 18cm">
<br>
    
Experimental evidence suggest that C2BM are more interpretable, causally reliable, and improve responsiveness to interventions w.r.t. standard opaque and conceptbased models, while maintaining their accuracy.

---


(Tested with Python 3.12 on Ubuntu 22.04.3 LTS)


1) install dependencies with conda:
```
conda update conda
conda env create -f environment.yml
conda activate c2bm
```

2) eventually download the datasets:
download the required datasets and place them in your CACHE folder `~/.cache/c2bm/{dataset_name}/`:
    - asia: is downloaded automatically and saved in `~/.cache/c2bm/asia`
    - sachs: is downloaded automatically and saved in `~/.cache/c2bm/sachs`
    - alarm: is downloaded automatically and saved in `~/.cache/c2bm/alarm`
    - insurance: download .bif file from https://www.bnlearn.com/bnrepository/discrete-medium.html#insurance and save it in `~/.cache/c2bm/bnlearn_bif_datasets/`
    - hailfinder: download .bif file from https://www.bnlearn.com/bnrepository/discrete-medium.html#insurance and save it in `~/.cache/c2bm/bnlearn_bif_datasets/`
    - celeba: download all required files from https://www.kaggle.com/jessicali9530/celeba-dataset and save them all in `~/.cache/c2bm/CelebA/celeba/`
    - penumothorax: dowload the ResNet50 from https://github.com/Soombit-ai/cxr-clip and place the tar in `~/.cache/c2bm/siim_pneumothorax/pretrained_models/`
                    download the test_png train_png folders from https://www.kaggle.com/datasets/abhishek/siim-png-images and place them in `~/.cache/c2bm/siim_pneumothorax/`
                    move the siim_train.csv from this zip to `~/.cache/c2bm/siim_pneumothorax/`

3) have a look at the configuration files inside the `conf` folder. 
Every configuration file sweep_*.yaml is a sweep configuration used to run experiments on different dataset in sequence. 
A different sweep is available for each model (e.g., c2bm, cem).
A different sweep is available for 2 groups of datasets: synthetic datasets from the bnlearn repository (`sweep_{model_name}_bn.yaml`) and image datasets (`sweep_{model_name}_rw.yaml`).
The configuration files set the parameters to the ones used in the paper.
Comment out the runs you do not want to execute, e.g., less seeds or/and less datasets.

**(NOTE: configuring the sweeper using the `hydra-list-sweeper` package ('grid_params' and 'list_params' notation) prevents from showing possible caught errors during execution. Please use the standard hydra sweeper ('params' notation) for debuggin purposes.)**

4) double-check the other settings inside the configuration file, e.g., number of epochs, etc.

5) CAUSAL DISCOVERY:
   - to discover the graph using the automated pipeline:
        - set `dataset.load_graph: false`. 
        - set the `override causal_discovery: ges` in the configuration defaults. 
        - set the `override llm: gpt` and `override rag: standard` in the configuration defaults. API-keys are required. Place them in the `env.py` file. 
   - instead, to load the same graph as we discovered in the paper (which we provide for ease of reproducibility):
        - copy the `learned_graph/{dataset_name}/graph.pkl` file to the respective folder in `~/.cache/c2bm/{dataset_name}/`.
        - set `dataset.load_true_graph: false`. 
        - set `load_graph: true`.
   - instead, to load the true graph, if available: 
        - set: `dataset.load_true_graph: true`.
        - set: `dataset.load_graph: false`.

6) (optional) to enable logging with wandb, set `trainer.logger: wandb` and insert you wandb entity in env.py:

7) run the code:  `python main.py --config-name sweep_{}.yaml`

8) Visualization of results:
Results will be printed in the terminal. 
The results are saved in the `output` folder, organized by the date and time the sweep started.
To display the results and the intervention plots:
    - insert the output run paths in the `make_plot.py` script, following the many (commented out) examples already present in the script.
    - execute `python make_plot.py`
    - accuracies will be printed in the terminal and the plots will be saved in the `plots` folder.



## Bibtex reference

If you find this code useful please consider citing our paper:

```
@article{de2025causally,
  title={Causally reliable concept bottleneck models},
  author={De Felice, Giovanni and Flores, Arianna Casanova and De Santis, Francesco and Santini, Silvia and Schneider, Johannes and Barbiero, Pietro and Termine, Alberto},
  journal={arXiv preprint arXiv:2503.04363},
  year={2025}
}
```
