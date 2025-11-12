#include "node.hpp"
#include <algorithm>

namespace onnx_runner {

OpType stringToOpType(const std::string& op_name) {
    // Basic operations
    if (op_name == "MatMul") return OpType::MATMUL;
    if (op_name == "Relu") return OpType::RELU;
    if (op_name == "Add") return OpType::ADD;
    if (op_name== "Sub") return OpType::SUB;
    if (op_name == "Gemm") return OpType::GEMM;
    if (op_name == "Mul") return OpType::MUL;
    if (op_name == "Div") return OpType::DIV;
    // Tensor manipulation
    if (op_name == "Reshape") return OpType::RESHAPE;
    if (op_name == "Transpose") return OpType::TRANSPOSE;
    if (op_name == "Unsqueeze") return OpType::UNSQUEEZE;
    if (op_name == "Slice") return OpType::SLICE;
    if (op_name == "Concat") return OpType::CONCAT;
    if (op_name == "Gather") return OpType::GATHER;
    if (op_name == "Expand") return OpType::EXPAND;
    if (op_name == "Shape") return OpType::SHAPE;
    // Activations
    if (op_name == "Softmax") return OpType::SOFTMAX;
    if (op_name == "Sigmoid") return OpType::SIGMOID;
    // Reductions
    if (op_name == "ReduceMean") return OpType::REDUCEMEAN;
    // Math operations
    if (op_name == "Pow") return OpType::POW;
    if (op_name == "Sqrt") return OpType::SQRT;
    if (op_name == "Neg") return OpType::NEG;
    if (op_name == "Cos") return OpType::COS;
    if (op_name == "Sin") return OpType::SIN;
    // Comparison & logic
    if (op_name == "Equal") return OpType::EQUAL;
    if (op_name == "Greater") return OpType::GREATER;
    if (op_name == "Where") return OpType::WHERE;
    // Type operations
    if (op_name == "Cast") return OpType::CAST;
    // Tensor creation
    if (op_name == "Range") return OpType::RANGE;
    if (op_name == "ConstantOfShape") return OpType::CONSTANTOFSHAPE;
    // Advanced operations
    if (op_name == "RotaryEmbedding") return OpType::ROTARYEMBEDDING;
    if (op_name == "GroupQueryAttention") return OpType::GROUPQUERYATTENTION;
    if (op_name == "Trilu") return OpType::TRILU;
    if (op_name == "ScatterND") return OpType::SCATTERND;
    // Legacy/less common
    if (op_name == "Conv") return OpType::CONV;
    if (op_name == "MaxPool") return OpType::MAXPOOL;
    if (op_name == "Flatten") return OpType::FLATTEN;
    if (op_name == "BatchNormalization") return OpType::BATCHNORM;
    if (op_name == "ReduceSum") return OpType::REDUCESUM;
    if (op_name == "SimplifiedLayerNormalization") return OpType::SIMPLIFIEDLAYERNORM;
    if (op_name == "SkipSimplifiedLayerNormalization") return OpType::SKIPSIMPLIFIEDLAYERNORM;
    return OpType::UNKNOWN;
}

std::string opTypeToString(OpType op_type) {
    switch (op_type) {
        // Basic operations
        case OpType::MATMUL: return "MatMul";
        case OpType::RELU: return "Relu";
        case OpType::ADD: return "Add";
        case OpType::SUB: return "Sub";
        case OpType::GEMM: return "Gemm";
        case OpType::MUL: return "Mul";
        case OpType::DIV: return "Div";
        // Tensor manipulation
        case OpType::RESHAPE: return "Reshape";
        case OpType::TRANSPOSE: return "Transpose";
        case OpType::UNSQUEEZE: return "Unsqueeze";
        case OpType::SLICE: return "Slice";
        case OpType::CONCAT: return "Concat";
        case OpType::GATHER: return "Gather";
        case OpType::EXPAND: return "Expand";
        case OpType::SHAPE: return "Shape";
        // Activations
        case OpType::SOFTMAX: return "Softmax";
        case OpType::SIGMOID: return "Sigmoid";
        // Reductions
        case OpType::REDUCEMEAN: return "ReduceMean";
        // Math operations
        case OpType::POW: return "Pow";
        case OpType::SQRT: return "Sqrt";
        case OpType::NEG: return "Neg";
        case OpType::COS: return "Cos";
        case OpType::SIN: return "Sin";
        // Comparison & logic
        case OpType::EQUAL: return "Equal";
        case OpType::GREATER: return "Greater";
        case OpType::WHERE: return "Where";
        // Type operations
        case OpType::CAST: return "Cast";
        // Tensor creation
        case OpType::RANGE: return "Range";
        case OpType::CONSTANTOFSHAPE: return "ConstantOfShape";
        // Advanced operations
        case OpType::ROTARYEMBEDDING: return "RotaryEmbedding";
        case OpType::GROUPQUERYATTENTION: return "GroupQueryAttention";
        case OpType::TRILU: return "Trilu";
        case OpType::SCATTERND: return "ScatterND";
        // Legacy/less common
        case OpType::CONV: return "Conv";
        case OpType::MAXPOOL: return "MaxPool";
        case OpType::FLATTEN: return "Flatten";
        case OpType::BATCHNORM: return "BatchNormalization";
        case OpType::REDUCESUM: return "ReduceSum";
        case OpType::SIMPLIFIEDLAYERNORM: return "SimplifiedLayerNormalization";
        case OpType::SKIPSIMPLIFIEDLAYERNORM: return "SkipSimplifiedLayerNormalization";
        default: return "Unknown";
    }
}

} // namespace onnx_runner
