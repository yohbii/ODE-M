# Unlocking the Potential of Continual Model Merging: An ODE Perspective
## Abstract
> Continual Model Merging (CMM) enables rapid customization of foundation models across sequentially arriving tasks, offering a scalable alternative to repeated retraining. However, existing merging rules lack explicit controllability over the allocation of learning capacity between previously learned capabilities and newly merged models. Consequently, as tasks are merged sequentially, this deficiency accumulates into severe forgetting, particularly in scenarios with heterogeneous task importance, where performance allocation becomes highly inconsistent. The key reason can be attributed to the fact that previous methods treat each task model as an isolated parameter point and apply fixed algebraic combinations, rather than explicitly constructing a transition that respects how independently trained models can be connected in parameter space. Motivated by mode connectivity, we assume that desirable merged models lie on low loss connecting paths, and that continual merging should follow such paths without crossing loss barriers that induce forgetting. Grounded in these insights, we propose a novel ODE-driven Merging (ODE-M) tailored for CMM that traces such a path by integrating a time-dependent velocity field and enforcing barrier constraints to prevent loss-increasing steps. Extensive experiments demonstrate that ODE-M achieves state-of-the-art performance compared to its competitors across mainstream CMM benchmarks.


## 📦 Installation
### Create environment:
```bash
conda create -n ode_merging python=3.11
conda activate ode_merging
```

### Install FusionBench

> This project is built on top of [FusionBench](https://github.com/tanganke/fusion_bench) (commit:  `a2d3fa99cc8fe088981715b30e36a7a268b7a999`) and follows its configuration / modelpool / taskpool abstractions.

Install FusionBench:
```bash
pip install -e .
```

> [!NOTE] 
> During execution, the dataset and model will be downloaded from Hugging Face.

## Run

### Main Experiments
```bash
bash examples/ode_merging/vit_exp.sh
```

### Ablation Studies
```bash
bash examples/ode_merging/ablations.sh
```

## Citation
If you find our paper useful, you can consider citing our work:
```
@article{lin2026ode,
  title={Unlocking the Potential of Continual Model Merging: An ODE Perspective},
  author={Lihong Lin and Haidong Kang},
  journal={Forty-third International Conference on Machine Learning},
  year={2026}
}
```

## Acknowledgement
This project is built upon [FusionBench](https://github.com/tanganke/fusion_bench). We sincerely thank the FusionBench developers and contributors for open-sourcing and maintaining this valuable framework, whose infrastructure and tools greatly facilitated the development and evaluation of our work.
