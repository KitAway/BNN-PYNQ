#pragma once
#define CASSERT_DATAFLOW(x) ;
#include <ap_int.h>
#include <hls_stream.h>
#include "padding.h"
#include "slidingwindow.h"
#include "streamtools.h"
//#include <iostream>
//using namespace std;


template<
		unsigned int FMChannels,		// number of input feature maps
		unsigned int FMDim,			// width of input feature map (assumed square)
		unsigned int SIMD, 				// number of SIMD lanes
		typename TI,      // redefine I/O interpretation as needed for input activations
		int StreamW  // safely deducible (stream width must be int though!)
>
void ShiftLayer_Batch(hls::stream<ap_uint<StreamW>>  &in,
			    hls::stream<ap_uint<StreamW>> &out,
			    unsigned const   reps) {
#pragma HLS INLINE

	constexpr unsigned int FilterSize = 3;
	constexpr unsigned int FilterArea = FilterSize * FilterSize;
	constexpr unsigned int FilterMiddle = (FilterArea - 1) /2;
	constexpr unsigned int PadSize = (FilterSize - 1 ) / 2;
	constexpr unsigned int ShiftChan = ( FMChannels / 9 ) * 9;

  constexpr unsigned int InpPerImage = FMDim*FMDim*FMChannels/StreamW;
  constexpr unsigned int factor = FMChannels/SIMD;
  WidthAdjustedInputStream <StreamW, SIMD*TI::width, InpPerImage>  w_in (in,  reps);
  WidthAdjustedOutputStream <FMChannels*TI::width, StreamW, 
														FMDim * FMDim * FMChannels / (FMChannels*TI::width)>  Out (out,  reps);
  
	hls::stream<ap_uint<SIMD * TI::width> > pad("StreamingPadLayer_Batch");

	PaddingLayer_Batch<FMChannels, FMDim, SIMD, 1, TI>(w_in, pad, reps);
  hls::stream<ap_uint<SIMD * TI::width> > shiftInp("StreamingShiftLayer_Batch.convInp");
	ConvolutionInputGenerator<FilterSize, FMChannels, TI::width, FMDim + 2 * PadSize, FMDim, SIMD, 1>(pad, shiftInp, reps);
	ap_uint<FMChannels * TI::width> buffer[FilterArea];
	ap_uint<FMChannels * TI::width> outV;
	for(int rep = 0; rep < reps; rep++)
		for(int i = 0; i < FMDim; i ++)
			for(int j = 0; j < FMDim; j ++){
				for(int buf =0; buf < FilterArea; buf ++){
					for(int f=0; f < factor; f++){
						buffer[buf]((f+1)*SIMD*TI::width - 1, f*SIMD*TI::width)= shiftInp.read();
					}
//					cout << hex <<"buffer["<<buf<<"]=" << buffer[buf] << endl;
				}
				for(int s=0; s < FMChannels; s++){
#pragma HLS UNROLL
					int index = s % FilterArea;
					if(s >= ShiftChan)
						index = FilterMiddle ;
					outV((1 + s) * TI::width - 1, s * TI::width) = buffer[index]((1 + s) * TI::width - 1, s * TI::width);
				}
//				cout <<hex << "OutV="<<outV << endl;
				static_cast<hls::stream<ap_uint<FMChannels*TI::width>>&>(Out).write(outV);
			}
}


