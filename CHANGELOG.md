# Changelog

## [2.2.0] - December-22-2021
### Added
- Add `Superconduct.Lab` package for rapidly designing superconducting experiments, including the scan or normal experiments; it also provides the interface `Superconduct.Lab.LabSpace` to access the data service, `Superconduct.Lab.Runner` to connect the experiment devices or simulator.
- Add `Utils.ControlGate` module to define the composition of the logic gate and the corresponding control pulses.
- Add `Utils.WaveFunction` module to define the preset waveform functions.
- Add `Simulator.SimulatorAgent` module to define the agent served for Quanlse Simulator, it provides quick access to all data and functions of Quanlse Simulator; it is based on the `Superconduct.Lab` package.
- Add `Utils.Functions.subspaceVec()` function to extract a population vector in a subspace.
- Add symbol QOperator shortcuts `QOperator.chXY()`, `QOperator.chX()`, `QOperator.chY()`, `QOperator.chZ()`, `QOperator.chRO()` which has no matrix definition in `QOperator` module.
- Add `TrappedIon.Optimizer.OptimizerIon` module to calculate robust laser sequence
- Add `TrappedIon.Optimizer.QIonEvolution` module to define basic evolution function of trapped ion system
- Add `TrappedIon.QIonSystem` module to define basic class of robust control laser and trapped ion Chip property.
- Add `TrappedIon.QIonTrajectory` module to define the phonon trajectory evolution and visualize the ion-ion coupling.
### Changed
- Improve module `QWaveform`, add `dump2Dict()`, `dump2Json()` and `parseJson()` methods to fully support JSON serialization and deserialization; `QWaveform.addWave()` method now is also supported to pass string-formatted waveform function name to argument `f` to call the preset waveforms.
- Newly add the `Superconduct` package, and move the Calibration/Simulator packages into `Superconduct`.

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
