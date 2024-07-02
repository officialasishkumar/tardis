"""
Basic TARDIS Benchmark.
"""

import numpy as np
from asv_runner.benchmarks.mark import parameterize, skip_benchmark

import tardis.transport.montecarlo.formal_integral as formal_integral
from benchmarks.benchmark_base import BenchmarkBase
from tardis import constants as c
from tardis import run_tardis
from tardis.io.configuration.config_reader import Configuration
from tardis.model.geometry.radial1d import NumbaRadial1DGeometry

# Error in all functions: Terminating: fork() called from a process already using GNU OpenMP, this is unsafe. 
@skip_benchmark
class BenchmarkMontecarloMontecarloNumbaNumbaFormalIntegral(BenchmarkBase):
    """
    Class to benchmark the numba formal integral function.
    """

    def __init__(self):
        super().__init__()
        self.config = None
        filename = "data/tardis_configv1_benchmark.yml"
        path = self.get_relative_path(filename)
        self.config = Configuration.from_yaml(path)
        self.Simulation = run_tardis(
            self.config, log_level="ERROR", show_progress_bars=False
        )

        self.FormalIntegrator = formal_integral.FormalIntegrator(
            self.Simulation.simulation_state, self.Simulation.plasma, self.Simulation.transport
        )

    @parameterize(
        {
            "Parameters": [
                {
                    "nu": 1e14,
                    "temperature": 1e4,
                },
                {
                    "nu": 0,
                    "temperature": 1,
                },
                {
                    "nu": 1,
                    "temperature": 1,
                }
            ]
        }
    )
    def time_intensity_black_body(self, parameters):
        nu = parameters["nu"]
        temperature = parameters["temperature"]
        formal_integral.intensity_black_body(nu, temperature)

    @parameterize({"N": (1e2, 1e3, 1e4, 1e5)})
    def time_trapezoid_integration(self, n):
        h = 1.0
        data = np.random.random(int(n))

        formal_integral.trapezoid_integration(data, h)

    @staticmethod
    def calculate_z(r, p):
        return np.sqrt(r * r - p * p)

    TESTDATA = [
        np.linspace(1, 2, 3, dtype=np.float64),
        np.linspace(0, 1, 3),
        # np.linspace(1, 2, 10, dtype=np.float64),
    ]

    def formal_integral_geometry(self, r):
        # NOTE: PyTest is generating a full matrix with all the permutations.
        #       For the `time_calculate_z` function with values: [0.0, 0.5, 1.0]
        #       -  p=0.0, formal_integral_geometry0-0.0, param["r"]: [1.  1.5 2. ]
        #       -  p=0.5, formal_integral_geometry0-0.5, param["r"]: [1.  1.5 2. ]
        #       -  p=1.0, formal_integral_geometry0-1.0, param["r"]: [1.  1.5 2. ]
        #       -  p=0.0, formal_integral_geometry1-0.0, param["r"]: [0.  0.5 1. ]
        #       -  p=1.0, formal_integral_geometry1-1.0, param["r"]: [0.  0.5 1. ]
        #       Same for `test_populate_z_photosphere` function
        #       And for `test_populate_z_shells` function
        #       -  p=1e-05, formal_integral_geometry0-1e-05, param["r"]: [1.  1.5 2. ]
        #       -  p=0.5,   formal_integral_geometry0-0.5,   param["r"]: [1.  1.5 2. ]
        #       -  p=0.99,  formal_integral_geometry0-0.99,  param["r"]: [1.  1.5 2. ]
        #       -  p=1,     formal_integral_geometry0-1,     param["r"]: [1.  1.5 2. ]
        #       -  p=1e-05, formal_integral_geometry1-1e-05, param["r"]: [0.  0.5 1. ]
        #       -  p=0.5,   formal_integral_geometry1-0.5,   param["r"]: [0.  0.5 1. ]
        #       -  p=0.99,  formal_integral_geometry1-0.99,  param["r"]: [0.  0.5 1. ]
        #       -  p=1,     formal_integral_geometry1-1,     param["r"]: [0.  0.5 1. ]
        geometry = NumbaRadial1DGeometry(
            r[:-1],
            r[1:],
            r[:-1] * c.c.cgs.value,
            r[1:] * c.c.cgs.value,
        )
        return geometry

    @property
    def time_explosion(self):
        # previously used model value that passes tests
        # time taken for a photon to move 1 cm
        return 1 / c.c.cgs.value

    # Error: Terminating: fork() called from a process already using GNU OpenMP, this is unsafe.
    @skip_benchmark
    @parameterize({"p": [0.0, 0.5, 1.0], "Test data": TESTDATA})
    def time_calculate_z(self, p, test_data):
        func = formal_integral.calculate_z
        inv_t = 1.0 / self.verysimple_time_explosion
        r_outer = self.formal_integral_geometry(test_data).r_outer

        for r in r_outer:
            func(r, p, inv_t)

    @skip_benchmark
    @parameterize({"p": [0, 0.5, 1], "Test data": TESTDATA})
    def time_populate_z_photosphere(self, p, test_data):
        formal_integral.FormalIntegrator(
            self.formal_integral_geometry(test_data), None, None
        )
        r_inner = self.formal_integral_geometry(test_data).r_inner
        self.formal_integral_geometry(test_data).r_outer

        p = r_inner[0] * p
        oz = np.zeros_like(r_inner)
        oshell_id = np.zeros_like(oz, dtype=np.int64)

        formal_integral.populate_z(
            self.formal_integral_geometry(test_data),
            self.formal_integral_geometry(test_data),
            p,
            oz,
            oshell_id,
        )

    @skip_benchmark
    @parameterize({"p": [1e-5, 0.5, 0.99, 1], "Test data": TESTDATA})
    def time_populate_z_shells(self, p, test_data) -> None:
        func = formal_integral.populate_z

        size = len(self.formal_integral_geometry(test_data).r_inner)
        r_inner = self.formal_integral_geometry(test_data).r_inner
        r_outer = self.formal_integral_geometry(test_data).r_outer

        p = r_inner[0] + (r_outer[-1] - r_inner[0]) * p

        oz = np.zeros(size * 2)
        oshell_id = np.zeros_like(oz, dtype=np.int64)

        func(
            self.formal_integral_geometry(test_data),
            self.formal_integral_geometry(test_data),
            p,
            oz,
            oshell_id,
        )

    # Error: Terminating: fork() called from a process already using GNU OpenMP, this is unsafe.
    @skip_benchmark
    @parameterize(
        {
            "Parameters": [
                {
                    "N": [100, 1000, 10000]
                }
            ]
        }
    )
    def time_calculate_p_values(self, Parameters):
        r = 1.0
        formal_integral.calculate_p_values(r, Parameters["N"])

    # Benchmark for functions in FormalIntegrator class

    # Error: Terminating: fork() called from a process already using GNU OpenMP, this is unsafe.
    @skip_benchmark  
    def time_FormalIntegrator_functions(self):
        self.FormalIntegrator.calculate_spectrum(
            self.Simulation.transport.transport_state.spectrum.frequency
        )
        self.FormalIntegrator.make_source_function()
        self.FormalIntegrator.generate_numba_objects()
        self.FormalIntegrator.formal_integral(
            self.Simulation.transport.transport_state.spectrum.frequency,
            1000
        )
