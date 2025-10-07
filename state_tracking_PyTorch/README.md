# 📊 Experimental Results: Length Generalization on FSA State Tracking

This submodule contains the code for reproducing the PD-SSM results in Table 2 from the paper, shown below.

<p align="center">
<img width="600" alt="pdssm_full_model" src=../assets/fsa_table.png><br>
  <em>FSA Emulation Results</em>
</p>

## Environment

The base environment only consists of a few packages.

```
conda create -n pdssm_pytorch python=3.10
conda activate pdssm_pytorch
pip install -r requirements.txt
```

## Experiments

### Reproduce Experiments

The PD-SSM experiment configurations are saved in JSON format in ```./experiment_configs```. 
Each JSON config file corresponds to a task-seed combination.
For example, to train PD-SSM on parity with the seed 2, execute the following command:

```
python run_experiment.py -c parity_2
```

The checkpoints are stored in ```./checkpoints``` and the results are stored in ```./results```.

To analyse the results, execute the following command which prints the best validation accuracy for each task/seed combination in ```./results```:

```
python analyse_results.py
```

### Implementation details

This submodule is implemented in PyTorch, with the PD-SSM model implemented in a fully recurrent way, treating the column-sparse matrices as if they were unstructured and utilizing the standard operation

```
toch.matmul(A_t, x_t_minus_1)
```

The second submodule (time_series_JAX) provides a JAX implementation which utilizes an efficient parallel associative scan function.

## 🙏 Acknowledgments

The experimental code provided here is a modified version of the code (https://github.com/Benjamin-Walker/structured-linear-cdes) provided by the authors of *Structured Linear Controlled Differential Equations: Maximally Expressive and Parallel-in-Time Sequence Models*, with the BibTex citation below:


```bibtex
@misc{walker2025slices,
  title        = {Structured Linear CDEs: Maximally Expressive and Parallel-in-Time Sequence Models},
  author       = {Walker, Benjamin and Yang, Lingyi and Muca Cirone, Nicola and Salvi, Cristopher and Lyons, Terry},
  year         = {2025},
  month        = {May},
  url          = {https://arxiv.org/abs/2505.17761},
}
```

The set of tasks we use were originally introduced in *Neural Networks and the Chomsky Hieararchy* at ICLR 2023, with the citation below:

```bibtex
@inproceedings{deletang2023neural,
  author       = {Gr{\'{e}}goire Del{\'{e}}tang and
                  Anian Ruoss and
                  Jordi Grau{-}Moya and
                  Tim Genewein and
                  Li Kevin Wenliang and
                  Elliot Catt and
                  Chris Cundy and
                  Marcus Hutter and
                  Shane Legg and
                  Joel Veness and
                  Pedro A. Ortega},
  title        = {Neural Networks and the Chomsky Hierarchy},
  booktitle    = {11th International Conference on Learning Representations},
  year         = {2023},
}
```
