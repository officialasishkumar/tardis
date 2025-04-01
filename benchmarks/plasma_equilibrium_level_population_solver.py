import numpy as np
import pandas as pd
import astropy.units as u


from benchmarks.benchmark_base import BenchmarkBase
from tardis.plasma.electron_energy_distribution import ThermalElectronEnergyDistribution
from tardis.plasma.equilibrium.level_populations import LevelPopulationSolver
from tardis.plasma.equilibrium.rate_matrix import RateMatrix
from tardis.plasma.radiation_field import DilutePlanckianRadiationField


class BenchmarkLevelPopulationSolver(BenchmarkBase):
    """
    Class to benchmark the LevelPopulationSolver class.
    """

    repeat = 2

    def setup(self):
        rate_matrix_solver = RateMatrix(
            [(self.radiative_rate_solver, "radiative")],
            self.atomic_dataset.levels,
        )

        rad_field = DilutePlanckianRadiationField(
            self.collisional_simulation_state.t_radiative,
            dilution_factor=np.zeros_like(
                self.collisional_simulation_state.t_radiative
            ),
        )
        electron_dist = ThermalElectronEnergyDistribution(
            0, self.collisional_simulation_state.t_radiative, 1e6 * u.g / u.cm**3
        )

        rates_matrices = rate_matrix_solver.solve(rad_field, electron_dist)
        self.solver = LevelPopulationSolver(
            rates_matrices, self.atomic_dataset.levels
        )


    def time_calculate_level_population_simple(self):
        rates_matrix = np.array([[1, 1], [2, -2]])
        result = self.solver._LevelPopulationSolver__calculate_level_population(
            rates_matrix
        )

    def time_calculate_level_population_empty(self):
        rates_matrix = np.array([[]])
        self.solver._LevelPopulationSolver__calculate_level_population(
            rates_matrix
        )

    def time_calculate_level_population_zeros(self):
        rates_matrix = np.array([[0, 0], [0, 0]])
        self.solver._LevelPopulationSolver__calculate_level_population(
            rates_matrix
        )

    def time_solve(self):
        self.solver.solve()