#include "r3dh_allgather_private.h"
#include <climits>
#include <cstring>

int r3dh_allgather_geometry(const RadParam* params, int count, R3dhAllgatherGeometry* result) {
    if (!result || !params || count < 3 || count > 4) return RAD_E_INVAL;
    for (int index = 0; index < count; ++index) {
        if (!params[index].key) return RAD_E_INVAL;
        const char* key = params[index].key;
        if (std::strcmp(key,"world_size") && std::strcmp(key,"numel") &&
            std::strcmp(key,"dtype") && std::strcmp(key,"row")) return RAD_E_INVAL;
        for (int previous = 0; previous < index; ++previous)
            if (!std::strcmp(key,params[previous].key)) return RAD_E_INVAL;
    }
    const RadParam* world = rad_param_find(params,count,"world_size");
    const RadParam* numel = rad_param_find(params,count,"numel");
    const RadParam* type = rad_param_find(params,count,"dtype");
    const RadParam* row = rad_param_find(params,count,"row");
    if (!world || world->kind != RAD_P_INT || world->ival != 2) return RAD_E_SHAPE;
    if (!type || type->kind != RAD_P_STR || !type->sval) return RAD_E_DTYPE;
    R3dhAllgatherGeometry geometry;
    if (!std::strcmp(type->sval,"f32") || !std::strcmp(type->sval,"fp32")) {
        geometry.dtype = RAD_F32; geometry.bytes = 4;
    } else if (!std::strcmp(type->sval,"bf16")) {
        geometry.dtype = RAD_BF16; geometry.bytes = 2;
    } else if (!std::strcmp(type->sval,"f16") || !std::strcmp(type->sval,"fp16")) {
        geometry.dtype = RAD_F16; geometry.bytes = 2;
    } else return RAD_E_DTYPE;
    if (!numel || (numel->kind != RAD_P_INT && numel->kind != RAD_P_RANGE)) return RAD_E_SHAPE;
    geometry.low = numel->ival;
    geometry.high = numel->kind == RAD_P_RANGE ? numel->ihi : numel->ival;
    if (geometry.low < 1 || geometry.high < geometry.low ||
        geometry.high > r3dh_allgather_max_numel) return RAD_E_SHAPE;
    if (row) {
        if (row->kind != RAD_P_INT || row->ival < 0) return RAD_E_SHAPE;
        geometry.row = row->ival;
        if (geometry.row > geometry.high || (geometry.row &&
            (geometry.low % geometry.row || geometry.high % geometry.row))) return RAD_E_SHAPE;
    }
    if ((geometry.low * geometry.bytes) % 16 || (geometry.high * geometry.bytes) % 16 ||
        (geometry.row * geometry.bytes) % 16) return RAD_E_ALIGN;
    *result = geometry;
    return RAD_OK;
}
int r3dh_allgather_init_geometry(const RadParam* params, int count, R3dhAllgatherGeometry* result) {
    if (!result || !params || count < 3 || count > 5) return RAD_E_INVAL;
    RadParam frozen[4]{};
    int frozen_count = 0;
    bool group_seen = false;
    for (int index = 0; index < count; ++index) {
        if (!params[index].key) return RAD_E_INVAL;
        if (!std::strcmp(params[index].key,"tp_group")) {
            if (group_seen || params[index].kind != RAD_P_INT || params[index].ival < 0)
                return RAD_E_INVAL;
            group_seen = true;
        } else {
            if (frozen_count == 4) return RAD_E_INVAL;
            frozen[frozen_count++] = params[index];
        }
    }
    return r3dh_allgather_geometry(frozen,frozen_count,result);
}
namespace {
int tensor_extent(const RadTensor& tensor, int64_t row, int bytes, int64_t* count,
                  int64_t* pitch, uintptr_t* span) {
    if (!tensor.data || !tensor.rank || tensor.rank > RAD_MAX_RANK) return RAD_E_SHAPE;
    int64_t elements = 1;
    for (unsigned axis = 0; axis < tensor.rank; ++axis) {
        if (tensor.shape[axis] < 1 || tensor.shape[axis] > r3dh_allgather_max_numel * 2 / elements)
            return RAD_E_SHAPE;
        elements *= tensor.shape[axis];
    }
    if (elements % row) return RAD_E_SHAPE;
    *pitch = row;
    if (tensor.rank == 2 && tensor.shape[1] == row) {
        if (tensor.stride[1] != 1 || tensor.stride[0] < row) return RAD_E_STRIDE;
        *pitch = tensor.stride[0];
    } else {
        int64_t expected = 1;
        for (unsigned axis = tensor.rank; axis-- > 0;) {
            if (tensor.shape[axis] > 1 && tensor.stride[axis] != expected) return RAD_E_STRIDE;
            expected *= tensor.shape[axis];
        }
    }
    if (*pitch > INT_MAX / bytes || (*pitch * bytes) % 16) return RAD_E_STRIDE;
    const uint64_t extent = (uint64_t)(elements / row - 1) * (uint64_t)*pitch + row;
    if (extent > UINTPTR_MAX / bytes || (uintptr_t)tensor.data > UINTPTR_MAX - extent * bytes)
        return RAD_E_SHAPE;
    *span = extent * bytes;
    *count = elements;
    return RAD_OK;
}
}
int r3dh_allgather_view(const RadArgs* args, R3dhAllgatherGeometry* geometry,
                       R3dhAllgatherView* view) {
    if (!args || !geometry || !view || !args->t || args->n_t != 2) return RAD_E_INVAL;
    if (args->world_size != 2 || args->rank < 0 || args->rank > 1) return RAD_E_SHAPE;
    int status = r3dh_allgather_geometry(args->p,args->n_p,geometry);
    if (status != RAD_OK) return status;
    const unsigned dtype = args->t[0].dtype;
    if (args->t[1].dtype != dtype || (dtype != geometry->dtype &&
        !(geometry->dtype == RAD_F32 && (dtype == RAD_U32 || dtype == RAD_I32))))
        return RAD_E_DTYPE;
    int64_t elements = 1;
    if (!args->t[0].rank || args->t[0].rank > RAD_MAX_RANK) return RAD_E_SHAPE;
    for (unsigned axis = 0; axis < args->t[0].rank; ++axis) {
        const int64_t dimension = args->t[0].shape[axis];
        if (dimension < 1 || dimension > geometry->high / elements) return RAD_E_SHAPE;
        elements *= dimension;
    }
    if (elements > geometry->high) return RAD_E_SHAPE;
    view->numel = elements;
    view->row = geometry->row ? geometry->row : elements;
    if (elements % view->row || (elements * geometry->bytes) % 16) return RAD_E_SHAPE;
    view->rows = elements / view->row;
    int64_t input_count = 0, output_count = 0;
    uintptr_t input_span = 0, output_span = 0;
    status = tensor_extent(args->t[0],view->row,geometry->bytes,&input_count,&view->input_pitch,&input_span);
    if (status != RAD_OK) return status;
    status = tensor_extent(args->t[1],view->row * 2,geometry->bytes,&output_count,&view->output_pitch,&output_span);
    if (status != RAD_OK) return status;
    if (input_count != elements || output_count != elements * 2) return RAD_E_SHAPE;
    const uintptr_t input = (uintptr_t)args->t[0].data, output = (uintptr_t)args->t[1].data;
    if ((input | output) & 15u) return RAD_E_ALIGN;
    if (input < output + output_span && output < input + input_span) return RAD_E_INVAL;
    return RAD_OK;
}
extern "C" int r3dh_allgather_validate(const RadArgs* args) {
    R3dhAllgatherGeometry geometry;
    R3dhAllgatherView view;
    return r3dh_allgather_view(args,&geometry,&view);
}
extern "C" int r3dh_allgather_shape(const RadParam* params, int count, int operand, RadOpdDesc* out) {
    if (!out) return RAD_E_INVAL;
    if (operand < 0 || operand > 1) return RAD_E_SHAPE;
    R3dhAllgatherGeometry geometry;
    const int status = r3dh_allgather_geometry(params,count,&geometry);
    if (status != RAD_OK) return status;
    RadOpdDesc result{};
    result.dtype = geometry.dtype; result.rank = 1;
    result.shape[0] = geometry.high * (operand ? 2 : 1);
    result.fill = RAD_FILL_NORMAL; result.idx_const = -1;
    *out = result;
    return RAD_OK;
}
namespace {
const RadParamSpec params[]{{"world_size",RAD_P_INT,1,0},{"numel",RAD_P_INT,1,0},
    {"dtype",RAD_P_STR,1,0},{"row",RAD_P_INT,0,0}};
const RadOperandSpec operands[]{{"x",RAD_OPD_IN,0},{"y",RAD_OPD_OUT,0}};
const RadOpSchema schemas[]{{"all_gather",params,4,operands,2,
    "Opaque byte gather: y[rows][world_size][row], rank ordered; absent row concatenates."}};
const RadConstraint constraints[]{{"world_size",RAD_C_EQ,2,nullptr},
    {"numel",RAD_C_GE,1,nullptr},{"numel",RAD_C_LE,r3dh_allgather_max_numel,nullptr},
    {"dtype",RAD_C_IN,0,"f32 fp32 bf16 f16 fp16"}};
const RadKernelInfo kernel = [] {
    RadKernelInfo result{};
    result.name = "r3dh_allgather_tp2_peer_pull"; result.op = "all_gather"; result.family = "ar";
    result.computes = "Native gfx1100 fine-grained VRAM peer-pull byte gather";
    result.shape = "TP2; aligned rows; contiguous or pitched 2-D; separate output";
    result.dtypes = "f32,bf16,f16 opaque bits"; result.domain = RAD_DOMAIN_DEVICE; result.priority = 200;
    result.constraints = constraints; result.n_constraints = 4;
    result.init = r3dh_allgather_init; result.fini = r3dh_allgather_fini;
    result.launch = r3dh_allgather_launch; result.opd_shape = r3dh_allgather_shape;
    return result;
}();
}
extern "C" int r3dh_allgather_kernel_count() { return 1; }
extern "C" int r3dh_allgather_schema_count() { return 1; }
extern "C" const RadKernelInfo* r3dh_allgather_kernels(int* count) {
    if (count) *count = 1;
    return &kernel;
}
extern "C" const RadKernelInfo* r3dh_allgather_kernel_at(int index) { return index == 0 ? &kernel : nullptr; }
extern "C" const RadOpSchema* r3dh_allgather_schemas(int* count) {
    if (count) *count = 1;
    return schemas;
}
extern "C" const RadOpSchema* r3dh_allgather_schema_at(int index) { return index == 0 ? schemas : nullptr; }
