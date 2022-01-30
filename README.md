# Clique Densification
## Dependencies
### Compile Pivoter
```
cd examples
git clone https://github.com/sjain12/Pivoter.git
cd Pivoter
make
```

### Compile SNAP
```
cd examples
git clone https://github.com/snap-stanford/snap.git
cd snap
make all
mkdir ../bins
cp examples/forestfire/forestfire ../bins/
```

### Other Python Packages
```
pip install -r requirements.txt
```

## Experiments
### Empirical Observation
Clique statistics for empirical graphs on SNAP can be found under examples/_statistics.
The following command can be used to visualize it.
```
cd examples
python cli.py statistics num_cliques
```
(the following commands assume you're already under examples.)
or
```
python cli.py statistics num_cliques_combined
```

### Simulation
Perform ABC (approximate bayesian computation)

```
python cli.py statistics runABC --log_dir=outputs
```

Apply a different distance metric (Optional):
```
python cli.py statistics apply_custom_dist --log_dir=outputs
```

Parse the results from an ABC run
```
python cli.py statistics parse_results --log_dir=outputs
```

### Metrics
Before computing and comparing metrics, generate instances of the best parameters for each model:
```
python cli.py graph generate_based_on_best_paras --log_dir=outputs
```
then count the number of cliques with:
```
python cli.py statistics count_cliques
```
After that, you may run any command under the group statistics for a specific ABC run.
For example,
```
python cli.py statistics distance_edge --log_dir=outputs
```