/**
 * @file nvrtc/options.hpp
 *
 * @brief Definitions and utility functions relating to run-time compilation (RTC)
 * of CUDA code using the NVRTC library
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_NVRTC_COMPILATION_OPTIONS_HPP_
#define CUDA_API_WRAPPERS_NVRTC_COMPILATION_OPTIONS_HPP_

#include <cuda/api/device_properties.hpp>
#include "detail/marshalled_options.hpp"

#include <unordered_map>
#include <unordered_set>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>
#include <limits>

namespace cuda {



namespace detail {


template <class F, class... Args>
void for_each_argument(F f, Args&&... args) {
    using arrT = int[];
    static_cast<void>(arrT{(f(std::forward<Args>(args)), 0)...});
// This:
//	[](...){}((f(std::forward<Args>(args)), 0)...);
// doesn't guarantee execution order
}

/*
// Allocator adapter which should hopefully let
// vectors skip setting character values to 0 when
// resizing.
template <typename T, typename Allocator=std::allocator<T>>
class default_init_allocator : public Allocator {
  typedef std::allocator_traits<Allocator> a_t;
public:
  template <typename U> struct rebind {
    using other =
      default_init_allocator<
        U, typename a_t::template rebind_alloc<U>
      >;
  };

  using Allocator::A;

  template <typename U>
  void construct(U* ptr)
    noexcept(std::is_nothrow_default_constructible<U>::value) {
    ::new(static_cast<void*>(ptr)) U;
  }
  template <typename U, typename...Args>
  void construct(U* ptr, Args&&... args) {
    a_t::construct(static_cast<Allocator&>(*this),
                   ptr, std::forward<Args>(args)...);
  }
};
*/

} // namespaced detail

namespace rtc {

enum class cpp_dialect_t {
	cpp03 = 0,
	cpp11 = 1,
	cpp14 = 2,
	cpp17 = 3,
};

namespace detail {

constexpr const char* cpp_dialect_names[] =  {
	"c++03",
	"c++11",
	"c++14",
	"c++17",
};

} // namespace detail

struct compilation_options_t {

	static constexpr const size_t do_not_set_register_count { 0 };

	/**
	 * Target devices in terms of CUDA compute capability.
	 *
	 * @note Not all computcapabilities are supported! As of CUDA 11.0,
	 * the minimum supported compute capability is 3.5
	 *
	 * @note As of CUDA 11.0, the default is compute_52.
	 *
	 * @todo Use something simpler than set,
	 * e.g. a vector-backed ordered set or a dynamic bit-vector for membership.
	 */
    std::set<cuda::device::compute_capability_t> targets;

    /**
     * Generate relocatable code that can be linked with other relocatable device code. It is equivalent to
     *
     * @note equivalent to "--relocatable-device-code" or "-rdc" for NVCC.
     */
    bool generate_relocatable_code { false };

    /**
     * Do extensible whole program compilation of device code.
     *
     * @todo explain what that is.
     */
    bool compile_extensible_whole_program { false };

    /**
     *  Generate debugging information (and perhaps limit optimizations?)
     */
    bool debug { false };

    bool generate_line_info { false };

    /**
     * Specify the maximum amount of registers that GPU functions can use. Until a function-specific limit, a
     * higher value will generally increase the performance of individual GPU threads that execute this
     * function. However, because thread registers are allocated from a global register pool on each GPU,
     * a higher value of this option will also reduce the maximum thread block size, thereby reducing the
     * amount of thread parallelism. Hence, a good maxrregcount value is the result of a trade-off.
     * If this option is not specified, then no maximum is assumed. Value less than the minimum registers
     * required by ABI will be bumped up by the compiler to ABI minimum limit.
     *
     * @note Set this to @ref do_not_set_register_count to not pass this as a compilation option.
     *
     * @todo Use std::optional
     */
    size_t maximum_register_count { do_not_set_register_count };

    /**
     * When performing single-precision floating-point operations, flush denormal values to zero.
     *
     * @Setting @ref use_fast_math implies setting this to true.
     */
    bool flush_denormal_floats_to_zero { false };

    /**
     * For single-precision floating-point square root, use IEEE round-to-nearest mode or use a faster approximation.
     *
     * @Setting @ref use_fast_math implies setting this to false.
     */
    bool use_precise_square_root { true };

    /**
     * For single-precision floating-point division and reciprocals, use IEEE round-to-nearest mode or use a faster approximation.
     *
     * @Setting @ref use_fast_math implies setting this to false.
     */
    bool use_precise_division { true };

    /**
     * Enables (disables) the contraction of floating-point multiplies and adds/subtracts into floating-point multiply-add operations (FMAD, FFMA, or DFMA).
     *
     * @Setting @ref use_fast_math implies setting this to false.
     */
    bool use_fused_multiply_add { true };

    /**
     * Make use of fast math operations. Implies `--ftz=true--prec-div=false--prec-sqrt=false--fmad=true`.
     */
    bool use_fast_math { false };

    /**
     * Enables more aggressive device code vectorization in the NVVM optimizer.
     */
    bool extra_device_vectorization { false };

    bool specify_language_dialect { false };
    /**
     * Set language dialect to C++03, C++11, C++14 or C++17.
     *
     */
    cpp_dialect_t language_dialect { cpp_dialect_t::cpp03 };


    std::unordered_set<std::string> no_value_defines;

    std::unordered_map<std::string,std::string> valued_defines;

    bool disable_warnings { false };

    /**
     * Treat all kernel pointer parameters as if they had the `restrict` (or `__restrict`) qualifier.
     */
    bool assume_restrict { false };

    /**
     * Assume functions without an explicit specification of their execution space are `__device__`
     * rather than `__host__` functions.
     */
    bool default_execution_space_is_device { false };

    /**
     * A sequence of directories to be searched for headers. These paths are searched _after_ the
     * list of headers given to nvrtcCreateProgram.
     */
    std::vector<std::string> additional_include_paths;

    /**
     * Header files to preinclude during preprocessing of the source.
     *
     * @todo Check how these strings are interpreted. Do they need quotation marks? brackets? full paths?
     */
    std::vector<std::string> preinclude_files;

    /**
     * Provide builtin definitions of @ref std::move and @ref std::forward.
     *
     * @note Only relevant when the dialect is C++11 or later.
     */
    bool builtin_move_and_forward { true };

    /**
     * Provide builtin definitions of std::initializer_list class and member functions.
     *
     * @note Only relevant when the dialect is C++11 or later.
     */
    bool builtin_initializer_list { true };

protected:
    template <typename T>
    void process(T& opts) const;

public:
	marshalled_options_t marshal() const;
};

namespace detail {

const char* true_or_false(bool b) { return b ? "true" : "false"; }

}

template <typename T>
void compilation_options_t::process(T& opt_struct) const
{
	if (generate_relocatable_code)         { opt_struct.push_back("--relocatable-device-code=true");      }
	if (compile_extensible_whole_program)  { opt_struct.push_back("--extensible-whole-program=true");     }
	if (debug)                             { opt_struct.push_back("--debug");                             }
	if (generate_line_info)                { opt_struct.push_back("--generate-line-info");                }
	if (extra_device_vectorization)        { opt_struct.push_back("--extra-device-vectorization");        }
	if (disable_warnings)                  { opt_struct.push_back("--disable-warnings");                  }
	if (assume_restrict)                   { opt_struct.push_back("--restrict");                          }
	if (default_execution_space_is_device) { opt_struct.push_back("--device-as-default-execution-space"); }
	if (not builtin_move_and_forward )     { opt_struct.push_back("--builtin-move-forward=false");        }
	if (use_fast_math)                     { opt_struct.push_back("--use_fast_math");                     }
	else {
		if (flush_denormal_floats_to_zero) { opt_struct.push_back("--ftz=true");                          }
		if (not use_precise_square_root)   { opt_struct.push_back("--prec-sqrt=false");                   }
		if (not use_precise_division)      { opt_struct.push_back("--prec-div=false");                    }
		if (not use_fused_multiply_add)    { opt_struct.push_back("--fmad=false");                        }
	}

	if (specify_language_dialect) {
		opt_struct.push_back("--std=", detail::cpp_dialect_names[(unsigned) language_dialect]);
	}

	if (maximum_register_count != do_not_set_register_count) {
		opt_struct.push_back("--maxrregcount", maximum_register_count);
	}

	// Multi-value options

	for(const auto& target : targets) {
		opt_struct.push_back("--gpu-architecture=compute_", target.as_combined_number());
	}

	for(const auto& def : no_value_defines) {
		opt_struct.push_back("-D", def);
	}

	for(const auto& def : valued_defines) {
		opt_struct.push_back("-D",def.first, '=', def.second);
	}

	for(const auto& path : additional_include_paths) {
		opt_struct.push_back("--include-path=", path);
	}

	for(const auto& preinclude_file : preinclude_files) {
		opt_struct.push_back("--pre-include=", preinclude_file);
	}
}

marshalled_options_t compilation_options_t::marshal() const
{
	detail::marshalled_options_size_computer_t size_computer;
	process(size_computer);
	marshalled_options_t marshalled(size_computer.num_options(), size_computer.buffer_size());
	process(marshalled);
	return marshalled;
}


} // namespace rtc

} // namespace cuda

#endif // CUDA_API_WRAPPERS_NVRTC_COMPILATION_OPTIONS_HPP_
