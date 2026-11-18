#include <functional>
#include <numeric>
#include <string>

#include "kvbound/cuda/whisper_model.h"

namespace kvb::cuda {
namespace {

// Checks the tensors of one weight-table entry against the rows they must have in total. `cols`
// is the row length of a matrix and 0 for a one-dimensional tensor. A floating-point matrix may
// have trailing dimensions that flatten into its columns, as a convolution weight [out, in, 3]
// does. Only with `may_quantize` may a matrix be stored as integers.
void check_entry(const SafetensorsFile& file, const WeightFormat& format, const Program& program,
                 int index, std::int64_t total_rows, std::int64_t cols, bool may_quantize) {
  if (index < 0 || index >= static_cast<int>(program.weights.size())) {
    throw Error("an op of a Whisper program lacks a weight it reads");
  }
  const WeightParts& parts = program.weights[index];
  const std::vector<std::int64_t>& part_rows = program.part_rows[index];
  if (parts.empty() || (part_rows.empty() ? parts.size() != 1 : part_rows.size() != parts.size())) {
    throw Error("a stacked weight needs the rows of each of its parts");
  }
  std::int64_t stacked_rows = 0;
  for (std::size_t part = 0; part < parts.size(); ++part) {
    const std::string& name = parts[part];
    const TensorView& tensor = file.at(name);
    const std::vector<std::int64_t>& shape = tensor.shape;
    const std::int64_t rows = part_rows.empty() ? total_rows : part_rows[part];
    if (tensor.dtype == DType::I8 || tensor.dtype == DType::U8) {
      if (!may_quantize || cols == 0 || format.scheme == WeightScheme::F16) {
        throw Error(name + ": this tensor must be floating point");
      }
      if (shape != stored_shape(format, rows, cols)) {
        throw Error(name + ": shape does not match config.json");
      }
      if (file.at(name + ".scale").shape != scale_shape(format, rows, cols)) {
        throw Error(name + ".scale: shape does not match the weight format");
      }
    } else {
      const std::int64_t tail = shape.empty()
                                    ? 0
                                    : std::accumulate(shape.begin() + 1, shape.end(),
                                                      std::int64_t{1}, std::multiplies<>());
      const bool fits = !shape.empty() && shape[0] == rows &&
                        (cols == 0 ? shape.size() == 1 : shape.size() >= 2 && tail == cols);
      if (!fits) throw Error(name + ": shape does not match config.json");
    }
    stacked_rows += rows;
  }
  if (stacked_rows != total_rows) {
    throw Error(parts[0] + ": stacked rows do not match config.json");
  }
}

void check_program(const SafetensorsFile& file, const WeightFormat& format, const Program& program,
                   std::int64_t vocab_size, std::int64_t positions) {
  const auto check = [&](int index, std::int64_t rows, std::int64_t cols, bool may_quantize) {
    check_entry(file, format, program, index, rows, cols, may_quantize);
  };
  for (const Op& op : program.ops) {
    const std::int64_t out_cols = op.out >= 0 ? program.values[op.out].cols : 0;
    switch (op.kind) {
      case OpKind::Linear:
        check(op.weight, out_cols, program.values[op.in0].cols, true);
        if (op.bias >= 0) check(op.bias, out_cols, 0, false);
        break;
      case OpKind::LayerNorm:
      case OpKind::AddLayerNorm:
        check(op.weight, out_cols, 0, false);
        check(op.bias, out_cols, 0, false);
        break;
      case OpKind::AddPositions:
        // The kernel reads the table as FP16.
        check(op.weight, positions, out_cols, false);
        break;
      case OpKind::Embed:
        check(op.weight, vocab_size, out_cols, true);
        break;
      default:
        if (op.weight >= 0 || op.bias >= 0) {
          throw Error(std::string(op_name(op.kind)) + " reads no weight in a Whisper program");
        }
    }
  }
}

}  // namespace

DeviceWeights load_whisper_weights(const std::filesystem::path& model_dir,
                                   const WhisperPrograms& programs, const WhisperConfig& cfg,
                                   cudaStream_t stream) {
  const SafetensorsFile file(model_dir / "model.safetensors");
  const WeightFormat format = weight_format_from_metadata(file.metadata());
  check_program(file, format, programs.encoder, cfg.vocab_size, cfg.max_source_positions);
  check_program(file, format, programs.decoder, cfg.vocab_size, cfg.max_target_positions);
  // Both programs index one table, so it is loaded once.
  return DeviceWeights(file, programs.encoder.weights, stream);
}

}  // namespace kvb::cuda
