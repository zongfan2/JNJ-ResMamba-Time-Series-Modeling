# Import Mapping Guide

This document shows the mapping of old imports from the `Helpers/` directory to the new modular structure.

## Old vs New Import Paths

### Model Architectures

```python
# OLD
from Helpers.DL_models import setup_model, MBA_tsm_with_padding, ...

# NEW
from models import setup_model, MBA_tsm_with_padding, ...
```

### Data Loading and Preprocessing

```python
# OLD
from Helpers.DL_helpers import (
    load_data_tso,
    batch_generator,
    add_padding_TSO,
    add_padding_tso_patch,
    add_padding_tso_patch_h5,
    get_nb_steps,
    ...
)

# NEW
from data import (
    load_data_tso,
    batch_generator,
    add_padding_TSO,
    add_padding_tso_patch,
    add_padding_tso_patch_h5,
    get_nb_steps,
    ...
)
```

### Loss Functions

```python
# OLD
from Helpers.DL_helpers import measure_loss_tso, measure_loss_tso_with_continuity, ...

# NEW
from losses import measure_loss_tso, measure_loss_tso_with_continuity, ...
```

### DLR-TC Specific Losses

```python
# OLD
from Helpers.dlrtc_losses import (
    GeneralizedCrossEntropy,
    CompatibilityLoss,
    TemporalSmoothnessLoss,
    DLRTCLoss,
    SoftLabelManager,
)

# NEW
from losses.dlrtc import (
    GeneralizedCrossEntropy,
    CompatibilityLoss,
    TemporalSmoothnessLoss,
    DLRTCLoss,
    SoftLabelManager,
)
```

### Utility Functions

```python
# OLD
from Helpers.DL_helpers import (
    EarlyStopping,
    calculate_metrics_nn,
    smooth_predictions,
    smooth_predictions_combined,
    plot_tso_learning_curves,
    ...
)

# NEW
from utils import (
    EarlyStopping,
    calculate_metrics_nn,
    smooth_predictions,
    smooth_predictions_combined,
    plot_tso_learning_curves,
    ...
)
```

### Evaluation Functions

```python
# OLD
from Helpers.DL_helpers import some_evaluation_function, ...

# NEW
from evaluation import some_evaluation_function, ...
```

### Pretraining Methods

```python
# OLD
from Helpers.net.pretrainer import (
    AIMPretrainer,
    DINOPretrainer,
    RelConPretrainer,
    SimCLRPretrainer,
)

# NEW
from models.pretraining import (
    AIMPretrainer,
    DINOPretrainer,
    RelConPretrainer,
    SimCLRPretrainer,
)
```

### Data Parallel Pretraining

```python
# OLD
from Helpers.net.dataparallel_pretrainer import create_dataparallel_pretrainer

# NEW
from models.pretraining.dataparallel import create_dataparallel_pretrainer
```

### Data Preparation Utils

```python
# OLD
from Helpers.net.prepare_data import some_function, ...

# NEW
from data.prepare_data import some_function, ...
```

### Specialized Models (PatchTST, etc.)

```python
# OLD
from Helpers.net.patch_tst import PatchTST, ...

# NEW
from models import PatchTST, ...
# OR (depending on organization)
from models.specialized import PatchTST, ...
```

## Module Structure

### Models Package (`models/`)
- `__init__.py` - Main model imports and setup_model function
- `resmamba.py` - MBA_tsm and variants
- `pretraining/` - Self-supervised pretraining methods
  - `__init__.py` - AIMPretrainer, DINOPretrainer, etc.
  - `dataparallel.py` - Multi-GPU pretraining utilities
- `specialized/` - Specialized architectures
  - PatchTST
  - EfficientUNet
  - Other custom models

### Data Package (`data/`)
- `__init__.py` - Data loading functions
- `prepare_data.py` - Data preparation utilities
- `augmentation.py` - Data augmentation methods
- `h5_utils.py` - H5-specific utilities

### Losses Package (`losses/`)
- `__init__.py` - Standard loss functions
- `dlrtc.py` - Dynamic Label Refinement losses
- `contrastive.py` - Contrastive learning losses

### Utils Package (`utils/`)
- `__init__.py` - Utility functions (EarlyStopping, etc.)
- `metrics.py` - Metric calculation functions
- `smoothing.py` - Post-processing smoothing functions

### Evaluation Package (`evaluation/`)
- `__init__.py` - Evaluation functions
- `visualization.py` - Plotting and visualization utilities

## Wildcard Import Mapping

Some scripts use wildcard imports for convenience:

```python
# OLD
from Helpers.DL_models import *
from Helpers.DL_helpers import *

# NEW (NOT RECOMMENDED - use specific imports instead)
from models import *
from data import *
from losses import *
from utils import *
from evaluation import *
```

However, **explicit imports are recommended** for clarity and to avoid namespace conflicts:

```python
# RECOMMENDED
from models import setup_model, MBA_tsm_with_padding
from data import load_data_tso, batch_generator
from losses import measure_loss_tso
from utils import EarlyStopping
from evaluation import calculate_metrics_nn
```

## Update Checklist for Custom Scripts

When updating custom scripts to use the new import structure:

- [ ] Replace `from Helpers.DL_models import` with `from models import`
- [ ] Replace `from Helpers.DL_helpers import` with `from data/losses/utils/evaluation import` (as appropriate)
- [ ] Replace `from Helpers.dlrtc_losses import` with `from losses.dlrtc import`
- [ ] Replace `from Helpers.net.pretrainer import` with `from models.pretraining import`
- [ ] Replace `from Helpers.net.dataparallel_pretrainer import` with `from models.pretraining.dataparallel import`
- [ ] Replace `from Helpers.net.prepare_data import` with `from data.prepare_data import`
- [ ] Verify all imports are resolved
- [ ] Test script runs without import errors

## Pre-Converted Training Scripts

The following training scripts have been pre-converted and are ready to use:

1. `train_scratch.py` - Updated from `predict_scratch_segment.py`
2. `train_scratch_h5.py` - Updated from `predict_scratch_segment_h5.py`
3. `train_tso.py` - Updated from `predict_TSO_segment.py`
4. `train_tso_patch.py` - Updated from `predict_TSO_segment_patch.py`
5. `train_tso_patch_h5.py` - Updated from `predict_TSO_segment_patch_h5.py`
6. `train_tso_dlrtc.py` - Updated from `predict_TSO_segment_patch_dlrtc.py`
7. `pretrain.py` - Updated from `pretrain.py`
8. `convert_h5.py` - Copied from `convert_parquet_to_h5.py`

All import statements have been updated to use the new module structure.
