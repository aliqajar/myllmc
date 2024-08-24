

#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include "cuda_common.h"


// --------------------------------------------------------------------
// Packed128 data structure forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 ansd STS.128 instructions)
// This is a bit simiilar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision


template<class ElementType>
struct alignas(16) Packed128 {
    Packed128() = default;
    __device explicit Packed128(int4 bits) {
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&payload, &bits, sizeof(bits));
    }


    __device__ static Packed128 constant(ElementType value) {
        Packed128 result;
        for (int k=0; k<size; ++k) {
            result.payload[k] = value;
        }
        return result;
    }

    __device__ static Packed128 zeros() {
        return constant(0.f);
    }

    __device__ static packed128 ones() {
        return constant(1.f);
    }

    __device__ ElementType& operator[](int index) {
        return payload[index];
    }

    __device__ const ElementType& operator[] (int index) const {
        return payload[index];
    }

    __device__ int4 get_bits() const {
        int4 bits;
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch");
        memcpy(&bits, &payload, sizeof(bits));
        return bits;
    }

    static constexpr const size_t size = sizeof(int4) / sizeof(ElementType);
    ElementType payload[size];

}

// load a Packed128 from an aligned memory address
template<class ElementType>
__device__ Packed128<ElementType> load128(const ElementType* address) {
    return Packed128<ElementType> { *reinterpret_cast<const int4*> (address)};
    
}




#endif // CUDA_UTILS_CUH
