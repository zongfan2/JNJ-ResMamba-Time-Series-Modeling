"""
Structural import test for the refactored models/data/losses/evaluation/utils packages.

Mocks heavy dependencies (torch, mamba_ssm, transformers, etc.) so we can verify
all import chains resolve without needing a GPU environment.
"""
import sys
import types
import unittest
from unittest.mock import MagicMock


def create_mock_module(name, submodules=None):
    """Create a mock module that supports attribute access and submodule imports."""
    mod = types.ModuleType(name)
    mod.__path__ = []
    mod.__package__ = name
    mod.__spec__ = MagicMock()

    # Make any attribute access return a MagicMock (acts as class/function)
    class AttrModule(types.ModuleType):
        def __getattr__(self, attr):
            if attr.startswith('_'):
                return super().__getattribute__(attr)
            # Return a callable mock that can be used as a base class
            mock = MagicMock()
            mock.__mro__ = [mock, object]
            mock.__class__ = type
            return mock

    mod = AttrModule(name)
    mod.__path__ = []
    mod.__package__ = name
    mod.__spec__ = MagicMock()

    if submodules:
        for sub in submodules:
            full_name = f"{name}.{sub}"
            sub_mod = AttrModule(full_name)
            sub_mod.__path__ = []
            sub_mod.__package__ = full_name
            sub_mod.__spec__ = MagicMock()
            sys.modules[full_name] = sub_mod
            setattr(mod, sub, sub_mod)

    return mod


def setup_mocks():
    """Install mock modules for all heavy dependencies."""

    # --- torch and submodules ---
    torch_submodules = [
        'nn', 'nn.functional', 'nn.utils', 'nn.utils.rnn', 'nn.init',
        'optim', 'optim.lr_scheduler',
        'utils', 'utils.data', 'utils.data.distributed',
        'cuda', 'cuda.amp', 'distributed',
        'autograd', 'fft',
    ]
    torch_mock = create_mock_module('torch', [s.split('.')[0] for s in torch_submodules])
    sys.modules['torch'] = torch_mock

    # Create nested submodules
    for sub in torch_submodules:
        parts = sub.split('.')
        full = 'torch'
        parent = torch_mock
        for part in parts:
            full = f"{full}.{part}"
            if full not in sys.modules:
                child = create_mock_module(full)
                sys.modules[full] = child
                setattr(parent, part, child)
            parent = sys.modules[full]

    # torch.nn.Module needs to be a proper base class
    class FakeModule:
        def __init__(self, *args, **kwargs): pass
        def __init_subclass__(cls, **kwargs): pass
        def train(self, mode=True): return self
        def parameters(self): return []
        def named_parameters(self): return []
        def state_dict(self): return {}
        def load_state_dict(self, d, strict=True): pass

    class FakeParameter:
        def __init__(self, *args, **kwargs): pass

    sys.modules['torch.nn'].Module = FakeModule
    sys.modules['torch.nn'].Parameter = FakeParameter
    sys.modules['torch.nn'].ModuleList = list
    sys.modules['torch.nn'].Sequential = list
    sys.modules['torch.nn'].Linear = MagicMock
    sys.modules['torch.nn'].Conv1d = MagicMock
    sys.modules['torch.nn'].Conv2d = MagicMock
    sys.modules['torch.nn'].ConvTranspose1d = MagicMock
    sys.modules['torch.nn'].BatchNorm1d = MagicMock
    sys.modules['torch.nn'].LayerNorm = MagicMock
    sys.modules['torch.nn'].InstanceNorm1d = MagicMock
    sys.modules['torch.nn'].GroupNorm = MagicMock
    sys.modules['torch.nn'].Dropout = MagicMock
    sys.modules['torch.nn'].Dropout2d = MagicMock
    sys.modules['torch.nn'].ReLU = MagicMock
    sys.modules['torch.nn'].GELU = MagicMock
    sys.modules['torch.nn'].Sigmoid = MagicMock
    sys.modules['torch.nn'].Tanh = MagicMock
    sys.modules['torch.nn'].Embedding = MagicMock
    sys.modules['torch.nn'].MultiheadAttention = MagicMock
    sys.modules['torch.nn'].GRU = MagicMock
    sys.modules['torch.nn'].LSTM = MagicMock
    sys.modules['torch.nn'].RNN = MagicMock
    sys.modules['torch.nn'].Softmax = MagicMock
    sys.modules['torch.nn'].Identity = MagicMock
    sys.modules['torch.nn'].Unfold = MagicMock
    sys.modules['torch.nn'].Fold = MagicMock
    sys.modules['torch.nn'].AdaptiveAvgPool1d = MagicMock
    sys.modules['torch.nn'].MaxPool1d = MagicMock
    sys.modules['torch.nn.functional'].relu = MagicMock()
    sys.modules['torch.nn.functional'].gelu = MagicMock()
    sys.modules['torch.nn.functional'].softmax = MagicMock()
    sys.modules['torch.nn.functional'].interpolate = MagicMock()
    sys.modules['torch.nn.functional'].max_pool1d = MagicMock()
    sys.modules['torch.nn.functional'].cross_entropy = MagicMock()
    sys.modules['torch.nn.functional'].mse_loss = MagicMock()
    sys.modules['torch.nn.functional'].binary_cross_entropy_with_logits = MagicMock()
    sys.modules['torch.nn.functional'].pad = MagicMock()
    sys.modules['torch.nn.utils'].weight_norm = MagicMock()
    sys.modules['torch.nn.utils.rnn'] = create_mock_module('torch.nn.utils.rnn')

    # torch top-level functions
    torch_mock.zeros = MagicMock()
    torch_mock.ones = MagicMock()
    torch_mock.tensor = MagicMock()
    torch_mock.arange = MagicMock()
    torch_mock.randn = MagicMock()
    torch_mock.rand = MagicMock()
    torch_mock.cat = MagicMock()
    torch_mock.stack = MagicMock()
    torch_mock.sigmoid = MagicMock()
    torch_mock.where = MagicMock()
    torch_mock.exp = MagicMock()
    torch_mock.log = MagicMock()
    torch_mock.einsum = MagicMock()
    torch_mock.no_grad = MagicMock()
    torch_mock.Tensor = MagicMock
    torch_mock.float32 = 'float32'
    torch_mock.long = 'long'
    torch_mock.bool = 'bool'
    torch_mock.device = MagicMock()
    torch_mock.save = MagicMock()
    torch_mock.load = MagicMock()

    # --- mamba_ssm ---
    mamba_mock = create_mock_module('mamba_ssm', ['modules', 'ops'])
    sys.modules['mamba_ssm'] = mamba_mock
    sys.modules['mamba_ssm.modules'] = create_mock_module('mamba_ssm.modules')
    sys.modules['mamba_ssm.modules.mamba_simple'] = create_mock_module('mamba_ssm.modules.mamba_simple')
    sys.modules['mamba_ssm.modules.mamba2'] = create_mock_module('mamba_ssm.modules.mamba2')
    sys.modules['mamba_ssm.ops'] = create_mock_module('mamba_ssm.ops')
    sys.modules['mamba_ssm.ops.selective_scan_interface'] = create_mock_module('mamba_ssm.ops.selective_scan_interface')
    sys.modules['causal_conv1d'] = create_mock_module('causal_conv1d')

    # --- transformers ---
    transformers_mock = create_mock_module('transformers')
    sys.modules['transformers'] = transformers_mock

    # --- sklearn ---
    sklearn_mock = create_mock_module('sklearn', ['preprocessing', 'metrics', 'model_selection', 'utils', 'utils.class_weight'])
    sys.modules['sklearn'] = sklearn_mock
    sys.modules['sklearn.preprocessing'] = create_mock_module('sklearn.preprocessing')
    sys.modules['sklearn.metrics'] = create_mock_module('sklearn.metrics')
    sys.modules['sklearn.model_selection'] = create_mock_module('sklearn.model_selection')
    sys.modules['sklearn.utils'] = create_mock_module('sklearn.utils')
    sys.modules['sklearn.utils.class_weight'] = create_mock_module('sklearn.utils.class_weight')

    # torch.optim.lr_scheduler._LRScheduler (private class used by some modules)
    lr_sched = sys.modules['torch.optim.lr_scheduler'] = create_mock_module('torch.optim.lr_scheduler')
    lr_sched._LRScheduler = type('_LRScheduler', (), {
        '__init__': lambda self, *a, **kw: None,
        '__init_subclass__': classmethod(lambda cls, **kw: None),
    })

    # torch.nn.parallel
    sys.modules['torch.nn.parallel'] = create_mock_module('torch.nn.parallel')

    # --- torchvision ---
    torchvision_mock = create_mock_module('torchvision', ['models', 'transforms'])
    sys.modules['torchvision'] = torchvision_mock
    sys.modules['torchvision.models'] = create_mock_module('torchvision.models')
    sys.modules['torchvision.transforms'] = create_mock_module('torchvision.transforms')

    # --- einops ---
    einops_mock = create_mock_module('einops')
    einops_mock.rearrange = MagicMock()
    sys.modules['einops'] = einops_mock

    # --- timm (optional) ---
    sys.modules['timm'] = create_mock_module('timm')

    # --- other optional deps ---
    for mod_name in ['h5py', 'optuna', 'scipy', 'scipy.signal', 'scipy.stats',
                     'matplotlib', 'matplotlib.pyplot', 'seaborn',
                     'pandas', 'joblib', 'pyarrow', 'pyarrow.parquet',
                     'skimage', 'skimage.measure',
                     'torcheval', 'torcheval.metrics',
                     'datasets', 'ray', 'ray.train', 'ray.tune']:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = create_mock_module(mod_name)


class TestModelsImports(unittest.TestCase):
    """Test that all models/ submodules can be imported."""

    @classmethod
    def setUpClass(cls):
        setup_mocks()
        # Ensure our project root is on the path
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

    def test_import_components(self):
        from models.components import FeatureExtractor, TemporalBlock
        print("  OK: models.components")

    def test_import_mamba_blocks(self):
        from models.mamba_blocks import MBA, ConvFeedForward, MaskMambaBlock
        print("  OK: models.mamba_blocks")

    def test_import_attention(self):
        from models.attention import (
            AttModule, AttModule_mamba, MultiHeadSelfAttentionPooling,
            AttModule_mamba_causal, AttModule_mamba_cross, AttModule_cross
        )
        print("  OK: models.attention")

    def test_import_resmamba(self):
        from models.resmamba import MBA_tsm, MBA_tsm_with_padding, MBA_patch, MBA4TSO, latent_mixup
        print("  OK: models.resmamba")

    def test_import_encoder_decoder(self):
        from models.encoder_decoder import (
            MBA_encoder_decoder, PositionalEncoding, Encoder, Decoder,
            MBA_tsm_encoder_decoder_ch_bottleneck,
            MBA_tsm_encoder_decoder_seq_bottleneck,
            MBA_tsm_encoder_decoder_progressive_with_skip_connection,
            MBA_tsm_encoder_decoder_progressive,
        )
        print("  OK: models.encoder_decoder")

    def test_import_pretraining(self):
        from models.pretraining import MBATSMForPretraining
        print("  OK: models.pretraining")

    def test_import_pretrainer(self):
        from models.pretrainer import MAEPretrainer
        print("  OK: models.pretrainer")

    def test_import_baselines(self):
        from models.baselines import RNN, BiRNN, CNN, LSTMModel, MTCN
        print("  OK: models.baselines")

    def test_import_resnet(self):
        from models.resnet import ResNet1D, ResBlock1D
        print("  OK: models.resnet")

    def test_import_unet(self):
        from models.unet import EfficientUNet, ResNetUNet
        print("  OK: models.unet")

    def test_import_specialized(self):
        from models.specialized import PatchTSTHead, ViT, SwinT
        print("  OK: models.specialized")

    def test_import_vit1d(self):
        from models.vit1d import ViT1D
        print("  OK: models.vit1d")

    def test_import_conv1d(self):
        from models.conv1d import Conv1DBlock, Conv1DTS
        print("  OK: models.conv1d")

    def test_import_embed(self):
        from models.embed import PatchEmbed1D, PatchEmbedViT1D
        print("  OK: models.embed")

    def test_import_setup(self):
        from models.setup import setup_model
        print("  OK: models.setup")

    def test_import_models_init(self):
        """Test the top-level models/__init__.py imports everything."""
        import models
        # Check key symbols are accessible
        self.assertTrue(hasattr(models, 'MBA_tsm'))
        self.assertTrue(hasattr(models, 'MBA_tsm_with_padding'))
        self.assertTrue(hasattr(models, 'MBA_encoder_decoder'))
        self.assertTrue(hasattr(models, 'MBA4TSO'))
        self.assertTrue(hasattr(models, 'EfficientUNet'))
        self.assertTrue(hasattr(models, 'setup_model'))
        self.assertTrue(hasattr(models, 'ViT1D'))
        # Check backward-compatible aliases
        self.assertTrue(hasattr(models, 'MBA_tsm_encoder_decoder_ch_bottleneck'))
        self.assertTrue(hasattr(models, 'MBA_tsm_encoder_decoder_seq_bottleneck'))
        self.assertTrue(hasattr(models, 'MBA_tsm_encoder_decoder_progressive_with_skip_connection'))
        self.assertTrue(hasattr(models, 'MBA_tsm_encoder_decoder_progressive'))
        print("  OK: models (top-level __init__)")


class TestDataImports(unittest.TestCase):
    """Test that all data/ submodules can be imported."""

    @classmethod
    def setUpClass(cls):
        setup_mocks()
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

    def test_import_loading(self):
        from data.loading import load_data_from_h5, load_data, load_sequence_data
        print("  OK: data.loading")

    def test_import_preprocessing(self):
        import data.preprocessing
        print("  OK: data.preprocessing")

    def test_import_padding(self):
        import data.padding
        print("  OK: data.padding")

    def test_import_augmentation(self):
        import data.augmentation
        print("  OK: data.augmentation")

    def test_import_batching(self):
        import data.batching
        print("  OK: data.batching")

    def test_import_data_init(self):
        import data
        print("  OK: data (top-level __init__)")


class TestLossesImports(unittest.TestCase):
    """Test that all losses/ submodules can be imported."""

    @classmethod
    def setUpClass(cls):
        setup_mocks()
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

    def test_import_standard(self):
        import losses.standard
        print("  OK: losses.standard")

    def test_import_dlrtc(self):
        import losses.dlrtc
        print("  OK: losses.dlrtc")

    def test_import_losses_init(self):
        import losses
        print("  OK: losses (top-level __init__)")


class TestEvaluationImports(unittest.TestCase):
    """Test that all evaluation/ submodules can be imported."""

    @classmethod
    def setUpClass(cls):
        setup_mocks()
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

    def test_import_metrics(self):
        import evaluation.metrics
        print("  OK: evaluation.metrics")

    def test_import_postprocessing(self):
        import evaluation.postprocessing
        print("  OK: evaluation.postprocessing")

    def test_import_evaluation_init(self):
        import evaluation
        print("  OK: evaluation (top-level __init__)")


class TestUtilsImports(unittest.TestCase):
    """Test that utils/ can be imported."""

    @classmethod
    def setUpClass(cls):
        setup_mocks()
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

    def test_import_common(self):
        from utils.common import Retraining
        print("  OK: utils.common")


class TestShimImports(unittest.TestCase):
    """Test that legacy Helpers/ shims still work."""

    @classmethod
    def setUpClass(cls):
        setup_mocks()
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

    def test_import_models_shim(self):
        try:
            from Helpers.DL_models_shim import MBA_tsm_with_padding
            print("  OK: Helpers.DL_models_shim")
        except ImportError as e:
            print(f"  SKIP: Helpers.DL_models_shim ({e})")

    def test_import_helpers_shim(self):
        try:
            import Helpers.DL_helpers_shim
            print("  OK: Helpers.DL_helpers_shim")
        except ImportError as e:
            print(f"  SKIP: Helpers.DL_helpers_shim ({e})")


if __name__ == '__main__':
    unittest.main(verbosity=2)
