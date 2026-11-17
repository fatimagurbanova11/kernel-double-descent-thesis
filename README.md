
# The Double Descent Phenomenon in Kernel-Based Approximation

MATLAB numerical experiments accompanying the Master's thesis
"The Double Descent Phenomenon in Kernel-Based Approximation"
at the University of Padova.

## Numerical Experiments

The repository contains experiments investigating the
behaviour of kernel-based approximation as the number
of kernel centres increases.

Three kernel-function configurations are considered:

1. Gaussian kernel with the cone function.
2. Matern 3/2 kernel with the cone function.
3. Multiquadric kernel with the saddle function.

The experiments examine training and test errors in
the underparameterized, interpolation, and
overparameterized regimes.


## MATLAB Scripts

- `cone_gaussian.m` — Gaussian kernel with the cone function.
- `cone_matern32.m` — Matern 3/2 kernel with the cone function.
- `saddle_multiquadric.m` — Multiquadric kernel with the saddle function.
- `compare_gaussian_matern.m` — Comparison of Gaussian and Matern 3/2 test errors.


## Reproducing the Results

### Saving the numerical results

After running `cone_gaussian.m`, execute:

```matlab
save('gaussian_results.mat', 'M_list', 'err_test_fix', 'err_test_adap')
```

After running `cone_matern32.m`, execute:

```matlab
save('matern32_results.mat', 'M_list', 'err_test_fix', 'err_test_adap')
```

Then run `compare_gaussian_matern.m` to reproduce the
Gaussian versus Matern 3/2 comparison figure.

The repository also includes the saved `.mat` results files.


1. Open MATLAB.
2. Run `coneintwithallmodif.m`.
3. Save the resulting test-error arrays as
   `gaussian_results.mat`.
4. Run `cone_matern32.m`.
5. Save the resulting test-error arrays as
   `matern32_results.mat`.
6. Run `compare_gaussian_matern.m` to generate the
   comparison figure.

The experiments use a fixed random seed to support
reproducibility.

## Notes

The reported results correspond to individual numerical
realizations and should not be interpreted as universal
performance rankings of the kernels.
