#pragma once
#include <ap_int.h>
#include <hls_stream.h>


template<
		unsigned int FMChannels,		// number of input feature maps
		unsigned int FMDim,			// width of input feature map (assumed square)
		unsigned int SIMD, 				// number of SIMD lanes
		unsigned int PadSize,
		typename TI      // redefine I/O interpretation as needed for input activations
>
void PaddingLayer_Batch(hls::stream<ap_uint<SIMD * TI::width>>  &in,
			    hls::stream<ap_uint<SIMD * TI::width>> &out,
			    unsigned const   reps) {
#pragma HLS INLINE
  constexpr unsigned int factor = FMChannels/SIMD;
	
	for(int rep = 0; rep < reps; rep++){
		for(int i=0; i<FMDim + 2 * PadSize;i++)
			for(int j=0; j<FMDim + 2 * PadSize;j++){
#pragma HLS PIPELINE
				bool pad = (i < PadSize) || (i >= FMDim + PadSize) || (j < PadSize) || (j >= FMDim + PadSize);
				for(int f=0; f<factor; f++){
					if(pad)
						out.write((ap_uint<SIMD * TI::width>)0);
					else
						out.write(in.read());
				}
			}
	}
}

