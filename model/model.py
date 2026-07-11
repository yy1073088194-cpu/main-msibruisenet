"""Full model assembly: Encoder + UNet decoder + optional modules.

Supports three encoder backends (via config `encoder_name`):
    - "mobilenetv2": MobileNetV2, channels [16,24,32,96,320], stride down to 1/32
    - "mobilenetv3": MobileNetV3-Large, channels [16,24,40,112,960], stride down to 1/16
    - "efficientnet_b0": EfficientNet-B0, channels [16,24,40,112,1280], stride down to 1/16

The decoder channel widths auto-adapt to the encoder's output channels.
"""

import torch.nn as nn
import torch.nn.functional as F

from .encoder import MobileNetV2Encoder, MobileNetV3Encoder, EfficientNetB0Encoder
from .decoder import UNetDecoder
from .modules import SpectralConv1D, DiagonalBandGate, ConcreteSelectorK


ENCODERS = {
    "mobilenetv2": MobileNetV2Encoder,
    "mobilenetv3": MobileNetV3Encoder,
    "efficientnet_b0": EfficientNetB0Encoder,
}


class SegmentationModel(nn.Module):
    """Encoder-UNet segmentation model for MSI data.

    Configurable components:
        - Encoder: MobileNetV2 / EfficientNet-B0 / MobileNetV3
        - Skip modules: none / se
        - SpectralConv1D after S1 (optional)
    """

    def __init__(self, num_classes=2, in_channels=9,
                 encoder_name="mobilenetv2", pretrained=True,
                 first_layer_pretrained=True,
                 skip_module="none", se_reduction=16,
                 use_spectral_conv=False, spectral_conv_kernel_size=3,
                 use_input_spectral_conv=False, input_spectral_conv_kernel_size=3,
                 band_gate=None, encoder_in_channels=None):
        super().__init__()

        # Optional static band-selection gate at the network input.
        # Receives the wide candidate input and passes exactly k bands.
        self.band_gate = band_gate

        # Optional spectral 1D conv on the RAW input bands (before the encoder).
        # Mixes adjacent bands of the full multispectral input; channel count is
        # preserved (in_channels -> in_channels), so the encoder is unchanged.
        self.use_input_spectral_conv = use_input_spectral_conv
        if use_input_spectral_conv:
            self.input_spectral_conv = SpectralConv1D(
                num_channels=in_channels,
                kernel_size=input_spectral_conv_kernel_size,
            )

        # Encoder
        encoder_cls = ENCODERS.get(encoder_name)
        if encoder_cls is None:
            raise ValueError(
                f"Unknown encoder '{encoder_name}'. "
                f"Available: {list(ENCODERS.keys())}"
            )
        # The encoder's input width may differ from the raw input width when the
        # band gate changes the channel count (ConcreteSelectorK maps B -> k).
        enc_in = encoder_in_channels if encoder_in_channels is not None else in_channels
        self.encoder = encoder_cls(in_channels=enc_in, pretrained=pretrained,
                                   first_layer_pretrained=first_layer_pretrained)

        enc_channels = self.encoder.get_output_channels()

        # Optional spectral conv after S1
        self.use_spectral_conv = use_spectral_conv
        if use_spectral_conv:
            self.spectral_conv = SpectralConv1D(
                num_channels=enc_channels[0],
                kernel_size=spectral_conv_kernel_size,
            )

        # Decoder
        self.decoder = UNetDecoder(
            encoder_channels=enc_channels,
            num_classes=num_classes,
            skip_module=skip_module,
            se_reduction=se_reduction,
        )

    def forward(self, x):
        input_size = x.shape[2:]

        if self.band_gate is not None:
            x = self.band_gate(x)

        if self.use_input_spectral_conv:
            x = self.input_spectral_conv(x)

        features = self.encoder(x)  # [S1, S2, S3, S4, S5]

        if self.use_spectral_conv:
            features[0] = self.spectral_conv(features[0])

        logits = self.decoder(features)
        logits = F.interpolate(logits, size=input_size, mode="bilinear",
                               align_corners=False)
        return logits

    def set_gate_progress(self, frac):
        """Update the band gate's tau schedule (no-op if no gate)."""
        if self.band_gate is not None:
            self.band_gate.set_progress(frac)

    def get_selected_bands(self):
        """Return the gate's selected band indices, or None if no gate."""
        if self.band_gate is not None:
            return self.band_gate.selected_bands()
        return None


# Backward-compatible alias
MobileNetV2UNet = SegmentationModel


def build_model(cfg):
    """Build model from config dict.

    Multi-architecture dispatch via cfg["model"]["architecture"]:
        - "default" (or absent): SegmentationModel
        - "smp": segmentation_models_pytorch wrapper
        - "topformer": TopFormer (CVPR 2022)
        - "seaformer": SeaFormer (ICLR 2023)
        - "pidnet": PIDNet (CVPR 2023)
    """
    model_cfg = cfg["model"]
    arch = model_cfg.get("architecture", "default")

    if arch == "smp":
        from .smp_models import build_smp_model
        return build_smp_model(cfg)
    if arch == "topformer":
        from .topformer import build_topformer
        return build_topformer(cfg)
    if arch == "seaformer":
        from .seaformer import build_seaformer
        return build_seaformer(cfg)
    if arch == "pidnet":
        from .pidnet import build_pidnet
        return build_pidnet(cfg)
    if arch != "default":
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            f"Available: default, smp, topformer, seaformer, pidnet"
        )

    in_channels = cfg["data"].get("num_channels", 9)

    # ConcreteSelectorK outputs k channels (B -> k), so the encoder must be
    # built for k inputs. The topk gate is width-preserving (B -> B).
    # NOTE: with "concrete", use_input_spectral_conv / use_spectral_conv must
    # stay False (they are sized for the raw B-channel input).
    enc_in = in_channels
    if (model_cfg.get("use_band_gate", False)
            and model_cfg.get("band_select_type", "topk") == "concrete"):
        enc_in = model_cfg.get("band_gate_k", 3)

    return SegmentationModel(
        num_classes=model_cfg.get("num_classes", 2),
        in_channels=in_channels,
        encoder_name=model_cfg.get("encoder_name", "mobilenetv2"),
        pretrained=model_cfg.get("encoder_pretrained", True),
        first_layer_pretrained=model_cfg.get("first_layer_pretrained", True),
        skip_module=model_cfg.get("skip_module", "none"),
        se_reduction=model_cfg.get("se_reduction", 16),
        use_spectral_conv=model_cfg.get("use_spectral_conv", False),
        spectral_conv_kernel_size=model_cfg.get("spectral_conv_kernel_size", 3),
        use_input_spectral_conv=model_cfg.get("use_input_spectral_conv", False),
        input_spectral_conv_kernel_size=model_cfg.get("input_spectral_conv_kernel_size", 3),
        band_gate=_build_band_gate(model_cfg, in_channels),
        encoder_in_channels=enc_in,
    )


def _build_band_gate(model_cfg, in_channels):
    """Construct an input-stage band selector from config, or None if disabled.

    Both selectors sit in the same ``band_gate`` slot and share the
    set_progress / selected_bands API, so training scheduling and band logging
    are identical regardless of which is chosen. They differ in output width:
    topk preserves the B input channels, concrete outputs k channels (the
    encoder is sized accordingly in ``build_model``).

    Config keys (under ``model``):
        use_band_gate: bool (default False)
        band_select_type: "topk" (default, DiagonalBandGate) | "concrete"
            (ConcreteSelectorK — supervised concrete selector baseline)
        band_gate_k: int — number of bands to keep
        band_gate_tau_start / band_gate_tau_end — temperature anneal endpoints
            (for concrete these are T0 / TB of the exponential schedule)
        band_gate_random_select: bool — freeze on a random k-subset (topk only)
    """
    if not model_cfg.get("use_band_gate", False):
        return None

    select_type = model_cfg.get("band_select_type", "topk")
    k = model_cfg.get("band_gate_k", 3)
    tau_start = model_cfg.get("band_gate_tau_start", 1.0)
    tau_end = model_cfg.get("band_gate_tau_end", 0.05)

    if select_type == "concrete":
        return ConcreteSelectorK(
            num_bands=in_channels, k=k,
            tau_start=tau_start, tau_end=tau_end,  # tau_start=T0, tau_end=TB
        )
    if select_type == "topk":
        return DiagonalBandGate(
            num_bands=in_channels, k=k,
            tau_start=tau_start, tau_end=tau_end,
            random_select=model_cfg.get("band_gate_random_select", False),
        )
    raise ValueError(
        f"Unknown band_select_type '{select_type}'. Available: topk, concrete"
    )
