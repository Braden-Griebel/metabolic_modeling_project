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
from sklearn.model_selection import train_test_split

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

# Split into test and train sets
train_samples, test_samples = train_test_split(
    valid_samples, test_size=TEST_PROP, shuffle=True
)  # split 80/20
train_samples, validate_samples = train_test_split(
    train_samples, test_size=VALIDATION_PROP / (1 - TEST_PROP), shuffle=True
)  # split 10% of total off as validation samples

# Save the samples
train_samples.to_parquet(FLUX_SAMPLE_PATH / "train.parquet", index=False)
validate_samples.to_parquet(FLUX_SAMPLE_PATH / "validate.parquet", index=False)
test_samples.to_parquet(FLUX_SAMPLE_PATH / "test.parquet", index=False)
