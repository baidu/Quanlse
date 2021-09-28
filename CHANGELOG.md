# Changelog

## [2.1.0] - September-27-2021
### Added

- Add single-qubit calibration module;
- Add readout cavity calibration module;
- Add control-Z gate pulse calibration module;
- Add analysis toolkit for the two-qubit with coupler architecture;
- Add two-qubit `PulseModel` simulator;
- Add `ODESolver` modules, including adaptive ODE solver and `Numba` accelerator support;
- Add new preset waveform functions: `QWaveform.delay()` for inserting time delay and `QWaveform.virtualZ()` for virtual-Z phase shift;
- Add `QWaveform.appendWave()` function, simplify the usage for adding the waves to the Hamiltonian;
- Add `QWaveform.setLO()` function for local oscillator frequency setting;
- Add `freq`/`phase`/`phase0` parameters for `QWaveform` objects;
- Add `PulseModel.Simulate()` method;
- Add new SchedulerPipeline strategies: left-aligned and center aligned.

### Changed

- Improve `QHamiltonian.simulate()`, now support the state population return.

### Deprecated

- Remove `Plot.plotScheduler()`.


## [2.0.0] - July-1-2021
### Added

- Refactor the original Quanlse modules;
- Add the Multi-qubit noisy simulator module;
- Add the Randomized Benchmarking module;
- Add Zero-Noise Extrapolation module;
- Trapped ion: Add the general Mølmer-Sørensen gate module.


## [1.1.0] - April-19-2021
### Added

- New 1-qubit optimizer using GRAPE (GRadient Ascent Pulse Engineering) Algorithm;
- Noisy 1-qubit simulator including readout simulation;
- Trapped ion: 1-qubit optimizer and the two-qubit MS gate optimizer.


## [1.0.0] - Jan-23-2021
### Added

- First commit;
- Quanlse SDK contains toolkits, tutorials, examples.
