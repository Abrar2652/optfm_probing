# OPTFM: A Scalable Multi-View Graph Transformer for Hierarchical Pre-Training in Combinatorial Optimization

Sample code and data. **Note that we will open-source the code and data, and will ensure that the publicly available code is easy to run and capable of reproducing the numerical results reported in the paper.**

## Environment

1. Pre-Training and Downstream Task I: lns_environment.yaml
2. Downstream Task II: solpre_environment.yaml


## Pretraining

All the code data are located in the "node_pretrain/" & "graph_pretrain/" dir. As the pretraining data (Optimization Problems) are extremely large, we will open-source upon acceptance of our paper. Now we provide some sample files.
> Node-level pretraining: 

Note that the sample problems and the generated bipartite graphs are located in "Graphs_SCIP/", "Graphs_valid_SCIP/" and "Graphs_test_SCIP/"
```python
cd node_pretrain/
python main_mip.py
```
If you want to generate bipartite graph files by yourself, you can:
```python
python generate_graph_scip.py
```

Some test files to generate the F1 score data for each instance.
```python
python test_mip.py
```

## Downstream Task I:

You can train the solution prediction GBDT model by yourself, as follows:

```python
cd sol_predict
python train_solpre.py
```

Tot test the solution prediction results:
```python
python test_solpre.py
```

The pretrained model was directly extracted from the "node_pretrain/" dir.


## Downstream Task II:

Note that as the RL-based model was trained with tensorflow, so you should update the environment. Then you can train the LNS RL-based model by yourself, as follows:

```python
cd lns
python train_lns.py
```

Tot test the LNS solution generation results:
```python
python test_lns.py
```

The pretrained model was directly extracted from the "node_pretrain/" dir.

## Downstream Task III:

All the codes are located in the "tune/" dir. As it depends on the node-level pre-trained model to generate numerous files, you can only test some sample files in 
```python
cd tune
python test_config.py
``` 