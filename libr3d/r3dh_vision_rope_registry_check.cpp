#include "r3dh_vision_rope.h"
#include "../core/rad_internal.h"

extern "C" int r3dh_vision_rope_registry_matches(const RadParam* params, int count) {
    rad::Geometry geometry;
    for (int index = 0; index < count; ++index) {
        const auto& param = params[index];
        if (param.kind == RAD_P_INT) geometry.set_i(param.key, param.ival);
        else if (param.kind == RAD_P_STR) geometry.set_s(param.key, param.sval);
        else if (param.kind == RAD_P_F64) geometry.set_f(param.key, param.dval);
        else return -1;
    }
    const RadConstraint broken = RAD_CIN("sections", "18 18");
    if (rad::constraint_holds(broken, geometry)) return -1;
    const auto* row = r3dh_vision_rope_kernel();
    for (int index = 0; index < row->n_constraints; ++index)
        if (!rad::constraint_holds(row->constraints[index], geometry)) return 0;
    return 1;
}
