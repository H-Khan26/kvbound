"""Quantized weight format: the safetensors header contract shared with the C++ loader.

A quantized checkpoint keeps the Hugging Face tensor names. Every matrix that activations are
multiplied by is stored as integers under its own name plus an F16 scale tensor named
"<name>.scale"; the other tensors stay F16: biases and norm parameters, and in a Whisper
checkpoint the convolution weights and the position tables. The header's __metadata__ says how to
read the matrices:

  int8_pc  per output channel: name I8 [out, in], scale F16 [out],
           W[o, i] = scale[o] * q[o, i], q in [-127, 127].
  int4_g   groups of group_size along the input dimension: name U8 [out, in / 2], two
           two's-complement nibbles per byte, element 2j in bits 0..3 and 2j + 1 in bits 4..7;
           scale F16 [out, in / group_size], W[o, i] = scale[o, i // group_size] * q[o, i],
           q in [-7, 7].

A file without the scheme key is a plain FP16/BF16/F32 checkpoint (scheme fp16).

act_scales "awq" marks an int4_g file whose matrices were quantized after rescaling their input
channels by activation statistics (kvbound.awq). The inverse scales are folded into the norm
weights and up_proj, so the file reads and runs like any other int4_g file.

rounding "gptq" marks an int4_g file whose integers were chosen by GPTQ (kvbound.gptq) instead of
round-to-nearest: on the same grid and against the same kind of F16 scales, so it too reads and
runs like any other int4_g file. Without the key the integers are rounded to nearest.

Usage: WeightFormat.from_metadata(safe_open(path, "pt").metadata())
"""

from dataclasses import dataclass

FORMAT_VERSION = "1"

FP16 = "fp16"
INT8_PC = "int8_pc"
INT4_G = "int4_g"
SCHEMES = (FP16, INT8_PC, INT4_G)

CLIPS = ("minmax", "mse", "awq")
ACT_SCALES = ("", "awq")
ROUNDINGS = ("", "gptq")
PACK_ORDER = "lo_nibble_first"
SCALE_DTYPE = "F16"
SCALE_SUFFIX = ".scale"
DEFAULT_GROUP_SIZE = 128

# Largest integer magnitude each quantized scheme writes (symmetric, so -8 is never used in int4).
QMAX = {INT8_PC: 127, INT4_G: 7}

KEY_VERSION = "kvbound.format_version"
KEY_SCHEME = "kvbound.scheme"
KEY_GROUP_SIZE = "kvbound.group_size"
KEY_PACK_ORDER = "kvbound.pack_order"
KEY_SCALE_DTYPE = "kvbound.scale_dtype"
KEY_CLIP = "kvbound.clip"
KEY_ACT_SCALES = "kvbound.act_scales"
KEY_ROUNDING = "kvbound.rounding"
KEY_SOURCE = "kvbound.source"
KEY_SOURCE_SHA256 = "kvbound.source_sha256"
# Where the calibration tokens of an act_scales or gptq file came from.
KEY_CALIB_DATASET = "kvbound.calib_dataset"
KEY_CALIB_REVISION = "kvbound.calib_revision"
KEY_CALIB_SHA256 = "kvbound.calib_sha256"
KEY_CALIB_WINDOWS = "kvbound.calib_windows"
KEY_CALIB_WINDOW = "kvbound.calib_window"
KEY_CALIB_SEED = "kvbound.calib_seed"
CALIB_KEYS = (
    KEY_CALIB_DATASET,
    KEY_CALIB_REVISION,
    KEY_CALIB_SHA256,
    KEY_CALIB_WINDOWS,
    KEY_CALIB_WINDOW,
    KEY_CALIB_SEED,
)

# Weight variants by directory suffix (see ModelSpec.variant_dir).
VARIANTS = (
    FP16,
    INT8_PC,
    INT4_G,
    f"{INT4_G}.mse",
    f"{INT4_G}.awq",
    f"{INT4_G}.awq-noclip",
    f"{INT4_G}.gptq",
    f"{INT4_G}.awq-gptq",
    f"{INT4_G}.awq-noclip-gptq",
)


@dataclass(frozen=True)
class WeightFormat:
    scheme: str = FP16
    group_size: int = 0
    pack_order: str = ""
    scale_dtype: str = ""
    clip: str = ""
    act_scales: str = ""
    rounding: str = ""

    def __post_init__(self) -> None:
        if self.scheme not in SCHEMES:
            raise ValueError(f"unknown weight scheme {self.scheme!r}; known: {SCHEMES}")
        # The fields each scheme allows: (group_size, pack_order, scale_dtype, clips).
        group_size, pack_order, scale_dtype, clips = {
            FP16: (0, "", "", ("",)),
            INT8_PC: (0, "", SCALE_DTYPE, ("minmax",)),
            INT4_G: (self.group_size, PACK_ORDER, SCALE_DTYPE, CLIPS),
        }[self.scheme]
        fields = (self.group_size, self.pack_order, self.scale_dtype)
        if fields != (group_size, pack_order, scale_dtype) or self.clip not in clips:
            raise ValueError(f"inconsistent weight format: {self}")
        # Activation scales exist for int4_g only, with the min-max or the awq clip; the awq clip
        # search runs on the scaled activations, so it needs them.
        if self.act_scales:
            consistent = (
                self.act_scales in ACT_SCALES and self.scheme == INT4_G and self.clip != "mse"
            )
        else:
            consistent = self.clip != "awq"
        if not consistent:
            raise ValueError(f"inconsistent weight format: {self}")
        # GPTQ chooses int4_g integers against min-max or awq clip scales; the mse clip picks
        # scales by the rounding error of each weight alone, which GPTQ does not keep.
        if self.rounding and (
            self.rounding not in ROUNDINGS or self.scheme != INT4_G or self.clip == "mse"
        ):
            raise ValueError(f"inconsistent weight format: {self}")
        # The kernels read 32 weights per load and find a load's scale with a shift.
        if self.scheme == INT4_G and (
            self.group_size < 32 or self.group_size & (self.group_size - 1)
        ):
            raise ValueError(
                f"int4_g needs a group size that is a power of two, at least 32; "
                f"got {self.group_size}"
            )

    @classmethod
    def make(
        cls,
        scheme: str,
        group_size: int = DEFAULT_GROUP_SIZE,
        clip: str = "minmax",
        act_scales: str = "",
        rounding: str = "",
    ) -> "WeightFormat":
        """The format of a scheme with the fields it does not use filled in."""
        if scheme == INT4_G:
            return cls(INT4_G, group_size, PACK_ORDER, SCALE_DTYPE, clip, act_scales, rounding)
        if scheme == INT8_PC:
            return cls(INT8_PC, 0, "", SCALE_DTYPE, clip, act_scales, rounding)
        return cls(act_scales=act_scales, rounding=rounding)

    @property
    def quantized(self) -> bool:
        return self.scheme != FP16

    @property
    def variant(self) -> str:
        """Directory suffix of checkpoints in this format: the scheme, then ".mse" for the mse
        clip, ".awq" for activation scales with the awq clip and ".awq-noclip" for scales alone;
        GPTQ rounding adds ".gptq" to a bare scheme and "-gptq" to the rest (".awq-gptq")."""
        if self.act_scales:
            variant = f"{self.scheme}.awq" if self.clip == "awq" else f"{self.scheme}.awq-noclip"
        else:
            variant = f"{self.scheme}.mse" if self.clip == "mse" else self.scheme
        if self.rounding:
            variant += f"-{self.rounding}" if "." in variant else f".{self.rounding}"
        return variant

    def to_metadata(self) -> dict[str, str]:
        """The header keys of this format; empty for a plain checkpoint."""
        if not self.quantized:
            return {}
        metadata = {
            KEY_VERSION: FORMAT_VERSION,
            KEY_SCHEME: self.scheme,
            KEY_GROUP_SIZE: str(self.group_size),
            KEY_PACK_ORDER: self.pack_order,
            KEY_SCALE_DTYPE: self.scale_dtype,
            KEY_CLIP: self.clip,
        }
        # Written only when set, so files without activation scales or GPTQ keep their header.
        if self.act_scales:
            metadata[KEY_ACT_SCALES] = self.act_scales
        if self.rounding:
            metadata[KEY_ROUNDING] = self.rounding
        return metadata

    @classmethod
    def from_metadata(cls, metadata: dict[str, str] | None) -> "WeightFormat":
        """Reads a safetensors header's __metadata__; no scheme key means a plain checkpoint."""
        metadata = metadata or {}
        if KEY_SCHEME not in metadata:
            return cls()
        version = metadata.get(KEY_VERSION)
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported {KEY_VERSION} {version!r}; expected {FORMAT_VERSION}")
        try:
            return cls(
                scheme=metadata[KEY_SCHEME],
                group_size=int(metadata[KEY_GROUP_SIZE]),
                pack_order=metadata[KEY_PACK_ORDER],
                scale_dtype=metadata[KEY_SCALE_DTYPE],
                clip=metadata[KEY_CLIP],
                act_scales=metadata.get(KEY_ACT_SCALES, ""),
                rounding=metadata.get(KEY_ROUNDING, ""),
            )
        except KeyError as e:
            raise ValueError(f"quantized checkpoint lacks metadata key {e}") from e

    def check_shape(self, shape: tuple[int, ...] | list[int], name: str = "tensor") -> None:
        """Raises unless a matrix of this shape can be stored: the kernels read int8_pc rows 16
        values at a time, and int4_g rows as whole groups."""
        if not self.quantized:
            return
        if len(shape) != 2:
            raise ValueError(f"{name}: cannot quantize a {len(shape)}-D tensor")
        if self.scheme == INT8_PC and shape[1] % 16:
            raise ValueError(f"{name}: int8_pc needs a row length divisible by 16, got {shape[1]}")
        if self.scheme == INT4_G and shape[1] % self.group_size:
            raise ValueError(
                f"{name}: shape {tuple(shape)} does not split into int4 groups of "
                f"{self.group_size} along the input dimension"
            )

    def bits_per_weight(self, out_features: int, in_features: int) -> float:
        """Stored bits per element of an [out, in] tensor, its scales included."""
        if self.scheme == INT8_PC:
            return 8 + 16 / in_features
        if self.scheme == INT4_G:
            return 4 + 16 / self.group_size
        return 16.0
