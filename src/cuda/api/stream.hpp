/**
 * @file stream.hpp
 *
 * @brief A proxy class for CUDA streams, providing access to
 * all Runtime API calls involving their use and management.
 *
 * @note : Missing functionality: Stream attributes; stream capturing.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_STREAM_HPP_
#define CUDA_API_WRAPPERS_STREAM_HPP_

#include <cuda/api/current_context.hpp>
#include <cuda/api/current_device.hpp>
#include <cuda/api/error.hpp>
#include <cuda/api/kernel_launch.hpp>
#include <cuda/api/memory.hpp>
#include <cuda/api/miscellany.hpp>
#include <cuda/api/types.hpp>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <string>
#include <memory>
#include <utility>
#include <algorithm>

namespace cuda {

class device_t;
class event_t;
class stream_t;

namespace stream {

// Use this for the second argument to create_on_current_device()
enum : bool {
	implicitly_synchronizes_with_default_stream = true,
	no_implicit_synchronization_with_default_stream = false,
	sync = implicitly_synchronizes_with_default_stream,
	async = no_implicit_synchronization_with_default_stream,
};

enum wait_condition_t : unsigned {
    greater_or_equal_to            = CU_STREAM_WAIT_VALUE_GEQ,
    geq                            = CU_STREAM_WAIT_VALUE_GEQ,

    equality                       = CU_STREAM_WAIT_VALUE_EQ,
    equals                         = CU_STREAM_WAIT_VALUE_EQ,

    nonzero_after_applying_bitmask = CU_STREAM_WAIT_VALUE_AND,
    one_bits_overlap               = CU_STREAM_WAIT_VALUE_AND,
    bitwise_and                    = CU_STREAM_WAIT_VALUE_AND,

    zero_bits_overlap              = CU_STREAM_WAIT_VALUE_NOR,
    bitwise_nor                    = CU_STREAM_WAIT_VALUE_NOR,
} ;

namespace detail {

inline id_t create_in_current_context(
	bool          synchronizes_with_default_stream,
	priority_t    priority = stream::default_priority
)
{
	unsigned int flags = (synchronizes_with_default_stream == sync) ?
		cudaStreamDefault : cudaStreamNonBlocking;
	id_t new_stream_id;
	auto status = cudaStreamCreateWithPriority(&new_stream_id, flags, priority);
	    // We could instead have used an equivalent Driver API call:
	    // cuStreamCreateWithPriority(cudaStreamCreateWithPriority(&new_stream_id, flags, priority);
	cuda::throw_if_error(status,
		std::string("Failed creating a new stream in CUDA context ")
		+ cuda::detail::ptr_as_hex(context::current::detail::get_handle()));
	return new_stream_id;
}

inline context::handle_t context_handle_of(stream::id_t stream_id)
{
	context::handle_t handle;
	auto result = cuStreamGetCtx(stream_id, &handle);
	throw_if_error(result,
        "Failed obtaining the context of stream " + cuda::detail::ptr_as_hex(stream_id));
	return handle;
}

/**
 * @brief Obtains the device ID with which a stream with a given ID is associated
 *
 * @note No guarantees are made if the input stream id is the default stream's.
 *
 * @param stream_id a stream identifier, other than the default stream for any
 * device or context
 * @return the identifier of the device for which the stream was created.
 */
inline device::id_t device_id_of(stream::id_t stream_id);

inline void record_event_in_current_context(
	device::id_t       current_device_id,
	context::handle_t  current_context_handle_,
	stream::id_t       stream_id,
	event::id_t        event_id);

/**
 * Wraps a CUDA stream ID in a stream_t proxy instance,
 * possibly also taking on the responsibility of eventually
 * destroying the stream
 *
 * @return a stream_t proxy for the CUDA stream
 */
stream_t wrap(
	device::id_t       device_id,
	context::handle_t  context_handle,
	id_t               stream_id,
	bool               take_ownership = false) noexcept;

template<typename T>
CUresult wait_on_value(CUstream stream_id, CUresult address, T value, unsigned int flags);

template<typename T>
CUresult write_value(CUstream stream_id, CUresult address, T value, unsigned int flags);

} // namespace detail

} // namespace stream

inline void synchronize(const stream_t& stream);

/**
 * @brief Wrapper class for a CUDA stream
 *
 * @note a stream is specific to a context, and thus also specific to a device.
 *
 * @note This class is a "reference type", not a "value type". Therefore, making changes
 * to properties of the stream is a const-respecting operation on this class.
 */
class stream_t {

public: // type definitions
	using priority_t = stream::priority_t;

	enum : bool {
		doesnt_synchronizes_with_default_stream  = false,
		does_synchronize_with_default_stream     = true,
	};

public: // const getters
	stream::id_t id() const noexcept { return id_; }
	device_t device() const noexcept;
	context_t context() const noexcept;
	bool is_owning() const noexcept { return owning; }

public: // other non-mutators

	/**
	 * When true, work running in the created stream may run concurrently with
	 * work in stream 0 (the NULL stream), and there is no implicit
	 * synchronization performed between it and stream 0.
	 */
	bool synchronizes_with_default_stream() const
	{
		unsigned int flags;
		auto status = cudaStreamGetFlags(id_, &flags);
		    // Could have used the equivalent Driver API call,
		    // cuStreamGetFlags(id_, &flags);
		throw_if_error(status,
			std::string("Failed obtaining flags for a stream")
			+ " on CUDA device " + std::to_string(device_id_));
		return flags & CU_STREAM_NON_BLOCKING; // == cudaStreamNonBlocking;
	}

	priority_t priority() const
	{
		int the_priority;
		auto status = cudaStreamGetPriority(id_, &the_priority);
		    // Could have used the equivalent Driver API call:
		    // cuStreamGetPriority(id_, &the_priority);
		throw_if_error(status,
			std::string("Failure obtaining priority for a stream")
			+ " on CUDA device " + std::to_string(device_id_));
		return the_priority;
	}

	/**
	 * Determines whether all work on this stream has been completed
	 *
	 * @note having work is _not_ the same as being busy executing that work!
	 *
	 * @todo What if there are incomplete operations, but they're all waiting on
	 * something on another queue? Should the queue count as "busy" then?
	 *
	 * @return true if there is still work pending, false otherwise
	 */
	bool has_work_remaining() const
	{
		context::current::detail::scoped_override_t set_context_for_this_scope(context_handle_);
		auto status = cudaStreamQuery(id_);
		    // Could have used the equivalent Driver API call:
		    // cuStreamQuery(id_);
		switch(status) {
		case cudaSuccess:
			return false;
		case cudaErrorNotReady:
			return true;
		default:
			throw(cuda::runtime_error(status,
				"unexpected status returned from cudaStreamQuery() for stream "
				+ detail::ptr_as_hex(id_)));
		}
	}

	/**
	 * The opposite of @ref has_work()
	 *
	 * @return true if there is no work pending, false if all
	 * previously-scheduled work has been completed
	 */
	bool is_clear() const { return !has_work_remaining(); }

	/**
	 * An alias for @ref is_clear() - to conform to how the CUDA runtime
	 * API names this functionality
	 */
	bool query() const { return is_clear(); }


protected: // static methods

	/**
	 * A function used internally by this class as the host function to call directly; see
	 * @ref enqueue_t::host_function_call - but only with CUDA version 10.0 and later.
	 *
	 * @param stream_id the ID of the stream for which a host function call was triggered - this
	 * will be passed by the CUDA runtime
	 * @param stream_wrapper_members_and_callable a tuple, containing the information necessary to
	 * recreate the wrapper with which the callback is associated, without any additional CUDA API calls -
	 * plus the callable which was passed to @ref enqueue_t::host_function_call, and which the programmer
	 * actually wants to be called.
	 */
	template <typename Callable>
	static void stream_launched_host_function_adapter(void * stream_wrapper_members_and_callable)
	{
		using tuple_type = std::tuple<device::id_t, context::handle_t , stream::id_t, Callable>;
		auto* tuple_ptr = reinterpret_cast<tuple_type *>(stream_wrapper_members_and_callable);
		auto unique_ptr_to_tuple = std::unique_ptr<tuple_type>{tuple_ptr}; // Ensures deletion when we leave this function.
		auto device_id        = std::get<0>(*unique_ptr_to_tuple.get());
		auto context_handle   = std::get<1>(*unique_ptr_to_tuple.get());
		auto stream_id        = std::get<2>(*unique_ptr_to_tuple.get());
		const auto& callable  = std::get<3>(*unique_ptr_to_tuple.get());
		callable( stream_t{device_id, context_handle, stream_id, do_not_take_ownership} );
	}

	/**
	 * @brief A function to @ref `host_function_launch_adapter`, for use with the old-style CUDA Runtime API call,
	 * which passes more arguments to the callable - and calls the host function even on device failures.
	 *
	 * @param stream_id the ID of the stream for which a host function call was triggered - this
	 * will be passed by the CUDA runtime
	 * @note status indicates the status the CUDA status when the host function call is triggered; anything
	 * other than @ref `cuda::status::success` means there's been a device error previously - but
	 * in that case, we won't invoke the callable, as such execution is deprecated; see:
	 * https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM
	 * @param device_id_and_callable a pair-value, containing the ID of the device to which the stream launching
	 * the host function call is associated, as well as the callable callback which was passed to
	 * @ref enqueue_t::host_function_call, and which the programmer actually wants to be called.
	 */
	template <typename Callable>
	static void callback_launch_adapter(
		stream::id_t,
		status_t      status,
		void *        stream_wrapper_members_and_callable)
	{
		if (status != cuda::status::success) {
			using tuple_type = std::tuple<device::id_t, context::handle_t , stream::id_t, Callable>;
			delete reinterpret_cast<tuple_type*>(stream_wrapper_members_and_callable);
			return;
		}
		stream_launched_host_function_adapter<Callable>(stream_wrapper_members_and_callable);
	}

public: // mutators

	/**
	 * @brief A gadget through which commands are enqueued on the stream.
	 *
	 * @note this class exists solely as a form of "syntactic sugar", allowing for code such as
	 *
	 *   my_stream.enqueue.copy(foo, bar, my_size)
	 */
	class enqueue_t {
	protected:
		const stream_t& associated_stream;

	public:
		enqueue_t(const stream_t& stream) : associated_stream(stream) {}

		// Note: It's important to forward the parameters rather than
		// pass anything by value, e.g. incase Iw ont
		template<typename KernelFunction, typename... KernelParameters>
		void kernel_launch(
			bool                        thread_block_cooperativity,
			const KernelFunction&       kernel_function,
			launch_configuration_t      launch_configuration,
			KernelParameters &&...      parameters)
		{
			// Kernel executions cannot be enqueued in streams associated
			// with devices other than the current one, see:
			// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-and-event-behavior
			context::current::detail::scoped_override_t set_context_for_this_scope(associated_stream.context_handle_);
			return cuda::enqueue_launch(
				thread_block_cooperativity,
				kernel_function,
				associated_stream,
				launch_configuration,
				std::forward<KernelParameters>(parameters)...);
		}

		template<typename KernelFunction, typename... KernelParameters>
		void kernel_launch(
			const KernelFunction&       kernel_function,
			launch_configuration_t      launch_configuration,
			KernelParameters&&...       parameters)
		{
			// TODO: Somehow I can't avoid code duplication with the previous variant of kernel_launch;
			// why is that?
			//
			// return kernel_launch(cuda::thread_blocks_cant_cooperate,
			// 	kernel_function, stream_id_, launch_configuration, parameters...);
			//

            context::current::detail::scoped_override_t set_context_for_this_scope(associated_stream.context_handle_);
			return cuda::enqueue_launch(
				kernel_function, associated_stream, launch_configuration, std::forward<KernelParameters>(parameters)...);
		}

		/**
		 * Have the CUDA device perform an I/O operation between two specified
		 * memory regions (on or off the actual device)
		 *
		 */

		///@{
		/**
		 * @param destination destination region into which to copy. May be
		 * anywhere in which memory can be mapped to the device's memory space (e.g.
		 * the device's global memory, host memory or the global memory of another device)
		 * @param source destination region from which to copy. May be
		 * anywhere in which memory can be mapped to the device's memory space (e.g.
		 * the device's global memory, host memory or the global memory of another device)
		 * @param num_bytes size of the region to copy
		 **/
		void copy(void *destination, const void *source, size_t num_bytes)
		{
			// It is not necessary to make the device current, according to:
			// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-and-event-behavior
			memory::async::detail::copy(destination, source, num_bytes, associated_stream.id_);
		}

		void copy(memory::region_t destination, memory::region_t source)
		{
			memory::async::detail::copy(destination, source, associated_stream.id_);
		}
		///@}

		/**
		 * Set all bytes of a certain region in device memory (or unified memory,
		 * but using the CUDA device to do it) to a single fixed value.
		 *
		 * @param destination Beginning of the region to fill
		 * @param byte_value the value with which to fill the memory region bytes
		 * @param num_bytes size of the region to fill
		 */
		void memset(void *destination, int byte_value, size_t num_bytes)
		{
			// Is it necessary to set the device? I wonder.
            context::current::detail::scoped_override_t set_context_for_this_scope(associated_stream.context_handle_);
			memory::device::async::detail::set(destination, byte_value, num_bytes, associated_stream.id_);
		}

		/**
		 * Set all bytes of a certain region in device memory (or unified memory,
		 * but using the CUDA device to do it) to zero.
		 *
		 * @note this is a separate method, since the CUDA runtime has a separate
		 * API call for setting to zero; does that mean there are special facilities
		 * for zero'ing memory faster? Who knows.
		 *
		 * @param destination Beginning of the region to fill
		 * @param num_bytes size of the region to fill
		 */
		void memzero(void *destination, size_t num_bytes)
		{
            context::current::detail::scoped_override_t set_context_for_this_scope(associated_stream.context_handle_);
			memory::device::async::detail::zero(destination, num_bytes, associated_stream.id_);
		}

		/**
		 * Have an event 'fire', i.e. marked as having occurred,
		 * after all hereto-scheduled work on this stream has been completed.
		 * Threads which are @ref stream_t::wait_on() 'ing the event will become available
		 * for continued execution.
		 *
		 * @param existing_event A pre-created CUDA event (for the stream's device); any existing
		 * "registration" of the event to occur elsewhere is overwritten.
		 *
		 * @note It is possible to wait for events across devices, but it is _not_ possible to
		 * trigger events across devices.
		 **/
		event_t& event(event_t& existing_event);

		/**
		 * Have an event 'fire', i.e. marked as having occurred,
		 * after all hereto-scheduled work on this stream has been completed.
		 * Threads which are @ref stream_t::wait_on() 'ing the event will become available
		 * for continued execution.
		 *
		 * @note the parameters are the same as for @ref event::create()
		 *
		 * @note It is possible to wait for events across devices, but it is _not_ possible to
		 * trigger events across devices.
		 *
		 **/
		event_t event(
			bool          uses_blocking_sync = event::sync_by_busy_waiting,
			bool          records_timing     = event::do_record_timings,
			bool          interprocess       = event::not_interprocess);

		/**
		 * Execute the specified function on the calling host thread once all
		 * hereto-scheduled work on this stream has been completed.
		 *
		 * @param callable_ a function to execute on the host. It must be callable
		 * with two parameters: `cuda::stream::id_t stream_id, cuda::event::id_t event_id`
		 */
		template <typename Callable>
		void host_function_call(Callable callable_)
		{
            context::current::detail::scoped_override_t set_context_for_this_scope(associated_stream.context_handle_);

			// Since callable_ will be going out of scope after the enqueueing,
			// and we don't know anything about the scope of the original argument with
			// which we were called, we must make a copy of `callable_` on the heap
			// and pass that as the user-defined data. We also add information about
			// the enqueueing stream.
			auto raw_callable_extra_argument = new
				std::tuple<device::id_t, context::handle_t, stream::id_t, Callable>(
                    associated_stream.device_id_,
                    associated_stream.context_handle_,
					associated_stream.id(),
					Callable(std::move(callable_))
				);

			// While we always register the same static function, `callback_adapter` as the
			// callback - what it will actually _do_ is invoke the callback we were passed.

#if CUDART_VERSION >= 10000
			auto status = cudaLaunchHostFunc(
				associated_stream.id_, &stream_launched_host_function_adapter<Callable>, raw_callable_extra_argument);
			    // Could have used the equivalent Driver API call: cuLaunchHostFunc()
#else
			// The nVIDIA runtime API (at least up to v10.2) requires passing 0 as the flags
			// variable, see:
			// http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
			constexpr const unsigned fixed_flags { 0u };
			auto status = cudaStreamAddCallback(
				associated_stream.id_, &callback_launch_adapter<Callable>, raw_callable_extra_argument, fixed_flags);
			    // Could have used the equivalent Driver API call: cuAddStreamCallback()
#endif

			throw_if_error(status,
				std::string("Failed scheduling a callback to be launched")
				+ " on stream " + cuda::detail::ptr_as_hex(associated_stream.id_)
				+ " in CUDA context " + cuda::detail::ptr_as_hex(associated_stream.context_handle_) +
				" on device " + std::to_string(associated_stream.device_id_));
		}

		/**
		 * Sets the attachment of a region of managed memory (i.e. in the address space visible
		 * on all CUDA devices and the host) in one of several supported attachment modes.
		 *
		 * The attachmentis actually a commitment vis-a-vis the CUDA driver and the GPU itself
		 * that it doesn't need to worry about accesses to this memory from devices other than
		 * its object of attachment, so that the driver can optimize scheduling accordingly.
		 *
		 * @note by default, the memory region is attached to this specific stream on its
		 * specific device. In this case, the host will be allowed to read from this memory
		 * region whenever no kernels are pending on this stream.
		 *
		 * @note Attachment happens asynchronously, as an operation on this stream, i.e.
		 * the attachment goes into effect (some time after) previous scheduled actions have
		 * concluded.
		 */
		///@{
		/**
		 * @param managed_region_start a pointer to the beginning of the managed memory region.
		 * This cannot be a pointer to anywhere in the middle of an allocated region - you must
		 * pass whatever @ref cuda::memory::managed::allocate() (or `cudaMallocManaged()`)
		 * returned.
		 */
		void memory_attachment(
			const void* managed_region_start,
			memory::managed::attachment_t attachment = memory::managed::attachment_t::single_stream)
		{
            context::current::detail::scoped_override_t set_context_for_this_scope(associated_stream.context_handle_);
			// This fixed value is required by the CUDA Runtime API,
			// to indicate that the entire memory region, rather than a part of it, will be
			// attached to this stream
			constexpr const size_t length = 0;
			auto flags = static_cast<unsigned>(attachment);
			auto status =  cudaStreamAttachMemAsync(
				associated_stream.id_, managed_region_start, length, flags);
			    // Could have used the equivalent Driver API call cuStreamAttachMemAsync
			throw_if_error(status,
				"Failed scheduling an attachment of a managed memory region"
				" on stream " + cuda::detail::ptr_as_hex(associated_stream.id_)
				+ " in CUDA context" + cuda::detail::ptr_as_hex(associated_stream.context_handle_)
				+ " on device " + std::to_string(associated_stream.device_id_));
		}

		/**
		 * @param region the managed memory region to attach; it cannot be a sub-region -
		 * you must pass whatever @ref cuda::memory::managed::allocate() returned.
		 */
		void memory_attachment(
			memory::managed::region_t region,
			memory::managed::attachment_t attachment = memory::managed::attachment_t::single_stream)
		{
			memory_attachment(region.start, attachment);
		}
		///@}


		/**
		 * Will pause all further activity on the stream until the specified event has
		 * occurred  (i.e. has fired, i.e. has had all preceding scheduled work
		 * on the stream on which it was recorded completed).
		 *
		 * @note this call will not delay any already-enqueued work on the stream,
		 * only work enqueued _after_ the call.
		 *
		 * @param event_ the event for whose occurrence to wait; the event
		 * would typically be recorded on another stream.
		 *
		 */
		void wait(const event_t& event_);

		/**
		 * Schedule writing a single value to global device memory after all
		 * previous work has concluded.
		 *
		 * @tparam T the value to schedule a setting of. Can only be a raw
		 * uint32_t or uint64_t !
		 * @param address location in global device memory to set at the appropriate time.
		 * @param value the value to write to @p address.
		 * @param with_memory_barrier if false, allows reordering of this write operation
		 * with writes scheduled before it.
		 */
		template <typename T>
		void set_single_value(T* __restrict__ address, T value, bool with_memory_barrier = true)
        {
            static_assert(
                std::is_same<T,int32_t>::value or std::is_same<T,int64_t>::value,
                "Unsupported type for stream value wait."
            );
            unsigned flags = with_memory_barrier ?
                CU_STREAM_WRITE_VALUE_DEFAULT :
                CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER;
		    auto result = static_cast<status_t>(
		        stream::detail::write_value(associated_stream.id_, address, value, flags));
		    throw_if_error(result, "Failed scheduling a write to global memory on stream "
		        + cuda::detail::ptr_as_hex(associated_stream.id_)
                + " in CUDA context" + cuda::detail::ptr_as_hex(associated_stream.context_handle_)
                + " on device " + std::to_string(associated_stream.device_id_));
        }

        /**
         * Wait for a value in device global memory to change so as to meet some condition
         *
         * @tparam T the value to schedule a setting of. Can only be a raw
		 * uint32_t or uint64_t !
		 * @param address location in global device memory to set at the appropriate time.
		 * @param condition the kind of condition to check against the reference value. Examples:
         * equal to 5, greater-or-equal to 5, non-zero bitwise-and with 5 etc.
		 * @param value the condition is checked against this reference value. Example: waiting on
         * the value at address to be greater-or-equal to this value.
		 * @param with_memory_barrier If true, all remote writes guaranteed to have reached the device
         * before the wait is performed will be visible to all operations on this stream/queue scheduled
         * after the wait.
         */
        template <typename T>
        void wait(const T* address, stream::wait_condition_t condition, T value, bool with_memory_barrier = false)
        {
            static_assert(
                std::is_same<T,int32_t>::value or std::is_same<T,int64_t>::value,
                "Unsupported type for stream value wait."
            );
            unsigned flags = static_cast<unsigned>(condition) |
                (with_memory_barrier ? CU_STREAM_WAIT_VALUE_FLUSH : 0);
            auto result = static_cast<status_t>(
                stream::detail::wait_on_value(associated_stream.id_, address, value, flags));
            throw_if_error(result,
                "Failed scheduling a wait  to global memory on stream "
                + cuda::detail::ptr_as_hex(associated_stream.id_)
                + " in CUDA context" + cuda::detail::ptr_as_hex(associated_stream.context_handle_)
                + " on device " + std::to_string(associated_stream.device_id_));
        }

        /**
         * Guarantee all remote writes to the specified address are visible to subsequent operations
         * scheduled on this stream.
         *
         * @param address location the previous remote writes to which need to be visible to
         * subsequent operations.
         */
        void flush_remote_writes()
        {
            CUstreamBatchMemOpParams flush_op;
            flush_op.operation = CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES;
            unsigned count = 1;
            unsigned flags = 0;
            // Let's cross our fingers and assume nothing else needs to be set here...
            cuStreamBatchMemOp(associated_stream.id_, count, &flush_op, flags);
        }

        /**
         * Enqueue multiple single-value write, wait and flush operations to the device
         * (avoiding the overhead of multiple enqueue calls).
         *
         * @note see @ref wait(), @ref set_single_value and @ref flush_remote_writes.
         *
         * @{
         */

        /**
         * @param ops_begin beginning of a sequence of single-value operation specifications
         * @param ops_end end of a sequence of single-value operation specifications
         */
        template <typename Iterator>
        void single_value_operations_batch(Iterator ops_begin, Iterator ops_end)
        {
            static_assert(std::is_same<typename std::iterator_traits<Iterator>::value_type, CUstreamBatchMemOpParams>::value,
            "Only accepting iterator pairs for the CUDA-driver-API memory operation descriptor,"
                " CUstreamBatchMemOpParams, as the value type");
            auto num_ops = std::distance(ops_begin, ops_end);
            if (std::is_same<typename std::remove_const<decltype(ops_begin)>::type, CUstreamBatchMemOpParams* >::value,
                "Only accepting containers of the CUDA-driver-API memory operation descriptor, CUstreamBatchMemOpParams")
            {
                auto ops_ptr = reinterpret_cast<const CUstreamBatchMemOpParams*>(ops_begin);
                cuStreamBatchMemOp(associated_stream.id_, num_ops, ops_ptr);
            }
            else {
                auto ops_uptr = std::unique_ptr<CUstreamBatchMemOpParams[]>(new CUstreamBatchMemOpParams[num_ops]);
                std::copy(ops_begin, ops_end, ops_uptr.get());
                cuStreamBatchMemOp(associated_stream.id_, num_ops, ops_uptr.get());
            }
        }

        /**
         * @param single_value_ops A sequence of single-value operation specifiers to enqueue together.
         */
        template <typename Container>
        void single_value_operations_batch(const Container& single_value_ops)
        {
            return single_value_operations_batch(single_value_ops.begin(), single_value_ops.end());
        }

    }; // class enqueue_t

	friend class enqueue_t;

	/**
	 * Block or busy-wait until all previously-scheduled work
	 * on this stream has been completed
	 */
	void synchronize() const
	{
		cuda::synchronize(*this);
	}

protected: // constructor

    stream_t(
        device::id_t       device_id,
        context::handle_t  context_handle,
        stream::id_t       stream_id,
        bool               take_ownership = false) noexcept
	: device_id_(device_id), context_handle_(context_handle), id_(stream_id), owning(take_ownership) { }

public: // constructors and destructor

	stream_t(const stream_t& other) noexcept : 
		stream_t(other.device_id_, other.context_handle_, other.id_, false) 
	{ }

	stream_t(stream_t&& other) noexcept : 
		stream_t(other.device_id_, other.context_handle_, other.id_, other.owning)
	{
		other.owning = false;
	}

	~stream_t()
	{
		if (owning) {
			context::current::detail::scoped_override_t set_context_for_this_scope(context_handle_);
			cudaStreamDestroy(id_);
		}
	}

public: // operators

	// TODO: Do we really want to allow assignments? Hmm... probably not, it's
	// too risky - someone might destroy one of the streams and use the others
	stream_t& operator=(const stream_t& other) = delete;
	stream_t& operator=(stream_t& other) = delete;

public: // friendship

	friend stream_t stream::detail::wrap(
		device::id_t       device_id,
		context::handle_t  context_handle,
		stream::id_t       stream_id,
		bool               take_ownership) noexcept;

	friend inline bool operator==(const stream_t& lhs, const stream_t& rhs) noexcept
	{
		return
			lhs.context_handle_ == rhs.context_handle_
#ifndef NDEBUG
			and lhs.device_id_ == rhs.device_id_
#endif
			and lhs.id_ == rhs.id_;
	}

protected: // data members
	const device::id_t       device_id_;
	const context::handle_t  context_handle_;
	const stream::id_t       id_;
	bool                     owning;

public: // data members - which only exist in lieu of namespaces
	enqueue_t     enqueue { *this };
		// The use of *this here is safe, since enqueue_t doesn't do anything with it
		// on its own. Any use of enqueue only happens through, well, *this - and
		// after construction.
};

inline bool operator!=(const stream_t& lhs, const stream_t& rhs) noexcept
{
	return not (lhs == rhs);
}

namespace stream {

namespace detail {
/**
 * @brief Wrap an existing stream in a @ref stream_t instance.
 *
 * @param device_id ID of the device for which the stream is defined
 * @param stream_id ID of the pre-existing stream
 * @param take_ownership When set to `false`, the stream
 * will not be destroyed along with the wrapper; use this setting
 * when temporarily working with a stream existing irrespective of
 * the current context and outlasting it. When set to `true`,
 * the proxy class will act as it does usually, destroying the stream
 * when being destructed itself.
 * @return an instance of the stream proxy class, with the specified
 * device-stream combination.
 */
inline stream_t wrap(
	device::id_t       device_id,
	context::handle_t  context_handle,
	stream::id_t       stream_id,
	bool               take_ownership) noexcept
{
	return stream_t(device_id, context_handle, stream_id, take_ownership);
}

inline stream_t create(
	device::id_t       device_id,
	context::handle_t  context_handle,
	bool               synchronizes_with_default_stream,
	priority_t         priority = stream::default_priority)
{
	context::current::detail::scoped_override_t set_context_for_this_scope(context_handle);
	auto new_stream_id = cuda::stream::detail::create_in_current_context(
		synchronizes_with_default_stream, priority);
	return wrap(device_id, context_handle, new_stream_id, do_take_ownership);
}

template<>
inline CUresult wait_on_value<uint32_t>(CUstream stream_id, CUresult address, uint32_t value, unsigned int flags)
{
    return cuStreamWaitValue32(stream_id, address, value, flags);
}

template<>
inline CUresult wait_on_value<uint64_t>(CUstream stream_id, CUresult address, uint64_t value, unsigned int flags)
{
    return cuStreamWaitValue64(stream_id, address, value, flags);
}


template<>
inline CUresult write_value<uint32_t>(CUstream stream_id, CUresult address, uint32_t value, unsigned int flags)
{
    return cuStreamWriteValue32(stream_id, address, value, flags);
}

template<>
inline CUresult write_value<uint64_t>(CUstream stream_id, CUresult address, uint64_t value, unsigned int flags)
{
    return cuStreamWriteValue64(stream_id, address, value, flags);
}

} // namespace detail

/**
 * @brief Create a new stream (= queue) on a CUDA device.
 *
 * @param device the device on which a stream is to be created
 * @param synchronizes_with_default_stream if true, no work on this stream
 * will execute concurrently with work from the default stream (stream 0)
 * @param priority priority of tasks on the stream, relative to other streams,
 * for execution scheduling; lower numbers represent higher properties. Each
 * device has a range of priorities, which can be obtained using
 * @ref device_t::stream_priority_range() .
 * @return The newly-created stream
 */
inline stream_t create(
	device_t     device,
	bool         synchronizes_with_default_stream,
	priority_t   priority = stream::default_priority);

inline stream_t create(
	context_t    context,
	bool         synchronizes_with_default_stream,
	priority_t   priority = stream::default_priority);

} // namespace stream

using queue_t = stream_t;
using queue_id_t = stream::id_t;

inline void synchronize(const stream_t& stream)
{
	auto status = cudaStreamSynchronize(stream.id());
	throw_if_error(status,
		std::string("Failed synchronizing a stream")
		+ " on CUDA device " + std::to_string(stream.device().id()));
}


} // namespace cuda

#endif // CUDA_API_WRAPPERS_STREAM_HPP_
