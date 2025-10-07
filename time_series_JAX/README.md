# 📊 Experimental Results: Long Sequence Time-Series Classification

This submodule contains the code for reproducing the PD-SSM results in Table 3 from the paper, shown below.

<p align="center">
<img width="700" alt="pdssm_full_model" src=../assets/timeseries_table.png><br>
  <em>Time-Series Results.</em>
</p>

## Environment

To set up the environment, follow the procedure below:

```
conda create -n pdssm_jax python=3.10
conda activate pdssm_jax
conda install pre-commit=3.7.1 sktime=0.30.1 tqdm=4.66.4 matplotlib=3.8.4 -c conda-forge
# Substitue for correct Jax pip install: https://jax.readthedocs.io/en/latest/installation.html
pip install -U "jax[cuda12]==0.6.2" "jaxlib[cuda12]==0.6.2" equinox==0.12.2 optax==0.2.4 diffrax==0.7.0 signax==0.1.1 roughpy==0.2.0
```

## Experiments

### Download Data

The dataset should be downloaded and pre-processed by running the following commands:

```
python data_dir/download_uea.py
python data_dir/process_uea.py
```

### Reproduce Results

Running the following command will iterate over five seeds for the chosen hyperparameter configuration and store the results in ```./outputs```.

```
python run_experiment.py --task EigenWorms --lr 0.001 --n_blocks 2 --embed_dim 128 --ssm_dim 64
```

To analyse the results, run the command below with the appropriate task name.

```
python analyse_results.py -task EigenWorms
```

#### Hyperparameters

To fully reproduce the PD-SSM results from the table above, replace the relevant arguments in the above command by the values given in the table below:

| Task | Learning Rate | Num. Blocks | Embed Dim. | SSM Dim. |
|------------|---------------|------------|--------|-----------|
| EigenWorms       | 0.0001         | 6         | 128     | 16      | 
| EthanolConcentration       | 0.001        | 4         | 128    | 16       | 
| Heartbeat       | 0.001          | 2         | 64     | 128   | 
| MotorImagery       | 0.0001        | 2        | 64     | 16     | 
| SelfRegulationSCP1       | 0.00001         | 2        | 128     | 64   |
| SelfRegulationSCP2       | 0.001         | 6        | 128     | 16   | 


### Implementation details

This submodule is implemented in JAX, utilizing an efficient implementation of parallel associative scan.
Due to the incompatibility of ```argmax``` with the JAX autodifferentiation framework, a custom implementation of the backward pass is provided in ```models/PDSSM.py```.

## 🙏 Acknowledgment

The experimental code provided here is a modified version of the code https://github.com/Benjamin-Walker/log-neural-cdes provided by:

```bibtex
@article{Walker2024LogNCDE,
  title={Log Neural Controlled Differential Equations: The Lie Brackets Make a Difference},
  author={Walker, Benjamin and McLeod, Andrew D. and Qin, Tiexin and Cheng, Yichuan and Li, Haoliang and Lyons, Terry},
  journal={International Conference on Machine Learning},
  year={2024}
}
```