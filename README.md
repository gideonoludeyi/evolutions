# evolutions--pso-2d-py

## Getting Started
1. Build and Install
```sh
$ pip install .
```
2. Run PSO
```sh
$ pso run --seed 123
```
3. Plot Results
```sh
$ pso plot --seed 123 -o plot.gif
```
4. Get Help
Run `pso run --help` to check which hyperparameters can be modified.
```sh
$ pso run --help
```

eg. to change the inertia weights to `(0.5, 0.8)`, cognitive factor to `0.25`, and social factor to `1.25`, run:
```sh
$ pso run -w 0.5 -w 0.8 -c 0.25 -s 1.25 --seed 123
```

You can also change the same hyperparameters for the plot.
Run `pso plot --help` to see the list of options hyperparameters you can modify.
