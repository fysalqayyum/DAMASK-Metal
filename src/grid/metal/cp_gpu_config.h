// SPDX-License-Identifier: AGPL-3.0-or-later
//> @brief Compile-time configuration of the GPU constitutive kernel (embedded before cp_kernel.h).
#define CP_NO_TANGENT                                                                               // drop the tangent of the reference cp_integrate (cp_kernel.h,
                                                                                                    // tests only); production uses cp_core_tangent
