"""
Generate Flux Samples from Un-constrained Model, and Split into Training, Validation, and Test sets
"""

# Setup
# Imports
# Standard Library Imports
import pathlib

# External Imports
from cobra.sampling import OptGPSampler
import metworkpy

# Local Imports

# Setup Path
BASE_PATH = pathlib.Path(__file__).parent.parent
FLUX_SAMPLE_PATH = BASE_PATH / "flux_samples"
MODEL_PATH = BASE_PATH.parent / "models"

# Read Base Model
BASE_MODEL = metworkpy.read_model(MODEL_PATH / "iEK1011_m7H10_media.json")

# Run Parameters
PROCESSES = 16
NUM_SAMPLES = 100_000
TEST_PROP = 0.2
VALIDATION_PROP = 0.1

# Create Sampler Object
sampler = OptGPSampler(model=BASE_MODEL, thinning=250, processes=PROCESSES)

# Generate Samples
samples = sampler.sample(NUM_SAMPLES)

# Filter for valid samples
valid_samples = samples[sampler.validate(samples) == "v"]

# Shuffle the samples
