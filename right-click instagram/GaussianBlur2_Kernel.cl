/* ============================================================

Copyright (c) 2009-2010 Advanced Micro Devices, Inc.  All rights reserved.
 
Redistribution and use of this material is permitted under the following 
conditions:
 
Redistributions must retain the above copyright notice and all terms of this 
license.
 
In no event shall anyone redistributing or accessing or using this material 
commence or participate in any arbitration or legal action relating to this 
material against Advanced Micro Devices, Inc. or any copyright holders or 
contributors. The foregoing shall survive any expiration or termination of 
this license or any agreement or access or use related to this material. 

ANY BREACH OF ANY TERM OF THIS LICENSE SHALL RESULT IN THE IMMEDIATE REVOCATION 
OF ALL RIGHTS TO REDISTRIBUTE, ACCESS OR USE THIS MATERIAL.

THIS MATERIAL IS PROVIDED BY ADVANCED MICRO DEVICES, INC. AND ANY COPYRIGHT 
HOLDERS AND CONTRIBUTORS "AS IS" IN ITS CURRENT CONDITION AND WITHOUT ANY 
REPRESENTATIONS, GUARANTEE, OR WARRANTY OF ANY KIND OR IN ANY WAY RELATED TO 
SUPPORT, INDEMNITY, ERROR FREE OR UNINTERRUPTED OPERA TION, OR THAT IT IS FREE 
FROM DEFECTS OR VIRUSES.  ALL OBLIGATIONS ARE HEREBY DISCLAIMED - WHETHER 
EXPRESS, IMPLIED, OR STATUTORY - INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED 
WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, 
ACCURACY, COMPLETENESS, OPERABILITY, QUALITY OF SERVICE, OR NON-INFRINGEMENT. 
IN NO EVENT SHALL ADVANCED MICRO DEVICES, INC. OR ANY COPYRIGHT HOLDERS OR 
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, PUNITIVE,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, REVENUE, DATA, OR PROFITS; OR 
BUSINESS INTERRUPTION) HOWEVER CAUSED OR BASED ON ANY THEORY OF LIABILITY 
ARISING IN ANY WAY RELATED TO THIS MATERIAL, EVEN IF ADVISED OF THE POSSIBILITY 
OF SUCH DAMAGE. THE ENTIRE AND AGGREGATE LIABILITY OF ADVANCED MICRO DEVICES, 
INC. AND ANY COPYRIGHT HOLDERS AND CONTRIBUTORS SHALL NOT EXCEED TEN DOLLARS 
(US $10.00). ANYONE REDISTRIBUTING OR ACCESSING OR USING THIS MATERIAL ACCEPTS 
THIS ALLOCATION OF RISK AND AGREES TO RELEASE ADVANCED MICRO DEVICES, INC. AND 
ANY COPYRIGHT HOLDERS AND CONTRIBUTORS FROM ANY AND ALL LIABILITIES, 
OBLIGATIONS, CLAIMS, OR DEMANDS IN EXCESS OF TEN DOLLARS (US $10.00). THE 
FOREGOING ARE ESSENTIAL TERMS OF THIS LICENSE AND, IF ANY OF THESE TERMS ARE 
CONSTRUED AS UNENFORCEABLE, FAIL IN ESSENTIAL PURPOSE, OR BECOME VOID OR 
DETRIMENTAL TO ADVANCED MICRO DEVICES, INC. OR ANY COPYRIGHT HOLDERS OR 
CONTRIBUTORS FOR ANY REASON, THEN ALL RIGHTS TO REDISTRIBUTE, ACCESS OR USE 
THIS MATERIAL SHALL TERMINATE IMMEDIATELY. MOREOVER, THE FOREGOING SHALL 
SURVIVE ANY EXPIRATION OR TERMINATION OF THIS LICENSE OR ANY AGREEMENT OR 
ACCESS OR USE RELATED TO THIS MATERIAL.

NOTICE IS HEREBY PROVIDED, AND BY REDISTRIBUTING OR ACCESSING OR USING THIS 
MATERIAL SUCH NOTICE IS ACKNOWLEDGED, THAT THIS MATERIAL MAY BE SUBJECT TO 
RESTRICTIONS UNDER THE LAWS AND REGULATIONS OF THE UNITED STATES OR OTHER 
COUNTRIES, WHICH INCLUDE BUT ARE NOT LIMITED TO, U.S. EXPORT CONTROL LAWS SUCH 
AS THE EXPORT ADMINISTRATION REGULATIONS AND NATIONAL SECURITY CONTROLS AS 
DEFINED THEREUNDER, AS WELL AS STATE DEPARTMENT CONTROLS UNDER THE U.S. 
MUNITIONS LIST. THIS MATERIAL MAY NOT BE USED, RELEASED, TRANSFERRED, IMPORTED,
EXPORTED AND/OR RE-EXPORTED IN ANY MANNER PROHIBITED UNDER ANY APPLICABLE LAWS, 
INCLUDING U.S. EXPORT CONTROL LAWS REGARDING SPECIFICALLY DESIGNATED PERSONS, 
COUNTRIES AND NATIONALS OF COUNTRIES SUBJECT TO NATIONAL SECURITY CONTROLS. 
MOREOVER, THE FOREGOING SHALL SURVIVE ANY EXPIRATION OR TERMINATION OF ANY 
LICENSE OR AGREEMENT OR ACCESS OR USE RELATED TO THIS MATERIAL.

NOTICE REGARDING THE U.S. GOVERNMENT AND DOD AGENCIES: This material is 
provided with "RESTRICTED RIGHTS" and/or "LIMITED RIGHTS" as applicable to 
computer software and technical data, respectively. Use, duplication, 
distribution or disclosure by the U.S. Government and/or DOD agencies is 
subject to the full extent of restrictions in all applicable regulations, 
including those found at FAR52.227 and DFARS252.227 et seq. and any successor 
regulations thereof. Use of this material by the U.S. Government and/or DOD 
agencies is acknowledgment of the proprietary rights of any copyright holders 
and contributors, including those of Advanced Micro Devices, Inc., as well as 
the provisions of FAR52.227-14 through 23 regarding privately developed and/or 
commercial computer software.

This license forms the entire agreement regarding the subject matter hereof and 
supersedes all proposals and prior discussions and writings between the parties 
with respect thereto. This license does not affect any ownership, rights, title,
or interest in, or relating to, this material. No terms of this license can be 
modified or waived, and no breach of this license can be excused, unless done 
so in a writing signed by all affected parties. Each term of this license is 
separately enforceable. If any term of this license is determined to be or 
becomes unenforceable or illegal, such term shall be reformed to the minimum 
extent necessary in order for this license to remain in effect in accordance 
with its terms as modified by such reformation. This license shall be governed 
by and construed in accordance with the laws of the State of Texas without 
regard to rules on conflicts of law of any state or jurisdiction or the United 
Nations Convention on the International Sale of Goods. All disputes arising out 
of this license shall be subject to the jurisdiction of the federal and state 
courts in Austin, Texas, and all defenses are hereby waived concerning personal 
jurisdiction and venue of these courts.

============================================================ */

/*
 * For a description of the algorithm and the terms used, please see the
 * documentation for this sample.
 *
 * Each thread calculates a pixel component(rgba), by applying a filter 
 * on group of 8 neighbouring pixels in both x and y directions. 
 * Both filters are summed (vector sum) to form the final result.
 */


__kernel void sobel_filter(__global uchar4* inputImage, __global uchar4* outputImage)
{
	uint x = get_global_id(0);
    uint y = get_global_id(1);

	uint width = get_global_size(0);
	uint height = get_global_size(1);

	float4 Gx = (float4)(0);
	float4 Gy = Gx;
	
	int c = x + y * width;


	/* Read each texel component and calculate the filtered value using neighbouring texel components */
	if( x >= 3 && x < (width-3) && y >= 3 && y < height - 3)
	{
		float4 i00 = convert_float4(inputImage[c - 3 - 3*width ]);
		float4 i10 = convert_float4(inputImage[c - 2 - 3*width]);
		float4 i20 = convert_float4(inputImage[c - 1 - 3*width]);
		float4 i30 = convert_float4(inputImage[c - 3*width ]);
		float4 i40 = convert_float4(inputImage[c + 1 - 3*width]);
		float4 i50 = convert_float4(inputImage[c + 2 - 3*width ]);
		float4 i60 = convert_float4(inputImage[c + 3 - 3*width]);
		float4 i01 = convert_float4(inputImage[c - 3 - 2*width]);
		float4 i11 = convert_float4(inputImage[c - 2 - 2*width]);
		float4 i21 = convert_float4(inputImage[c - 1 - 2*width]);
		float4 i31 = convert_float4(inputImage[c - 2*width]);
		float4 i41 = convert_float4(inputImage[c + 1 - 2*width]);
		float4 i51 = convert_float4(inputImage[c + 2 - 2*width]);
		float4 i61 = convert_float4(inputImage[c + 3 - 2*width]);
		float4 i02 = convert_float4(inputImage[c - 3 - width]);
		float4 i12 = convert_float4(inputImage[c - 2 - width]);
		float4 i22 = convert_float4(inputImage[c - 1 - width]);
		float4 i32 = convert_float4(inputImage[c - width]);
		float4 i42 = convert_float4(inputImage[c + 1 - width]);
		float4 i52 = convert_float4(inputImage[c + 2 - width]);
		float4 i62 = convert_float4(inputImage[c + 3 - width]);
		float4 i03 = convert_float4(inputImage[c-3]);
		float4 i13 = convert_float4(inputImage[c-2]);
	    float4 i23 = convert_float4(inputImage[c-1]);
		float4 i33 = convert_float4(inputImage[c]);
		float4 i43 = convert_float4(inputImage[c+1]);
		float4 i53 = convert_float4(inputImage[c+2]);
		float4 i63 = convert_float4(inputImage[c+3]);
		float4 i04 = convert_float4(inputImage[c - 3 + width]);
		float4 i14 = convert_float4(inputImage[c - 2 + width]);
		float4 i24 = convert_float4(inputImage[c - 1 + width]);
		float4 i34 = convert_float4(inputImage[c + width]);
		float4 i44 = convert_float4(inputImage[c + 1 + width]);
		float4 i54 = convert_float4(inputImage[c + 2 + width]);
		float4 i64 = convert_float4(inputImage[c + 3 + width]);
		float4 i05 = convert_float4(inputImage[c - 3 + 2*width]);
		float4 i15 = convert_float4(inputImage[c - 2 + 2*width]);
		float4 i25 = convert_float4(inputImage[c - 1 + 2*width]);
		float4 i35 = convert_float4(inputImage[c + 2*width]);
		float4 i45 = convert_float4(inputImage[c + 1 + 2*width]);
		float4 i55 = convert_float4(inputImage[c + 2 + 2*width]);
		float4 i65 = convert_float4(inputImage[c + 3 + 2*width]);
		float4 i06 = convert_float4(inputImage[c - 3 + 3*width]);
		float4 i16 = convert_float4(inputImage[c - 2 + 3*width]);
		float4 i26 = convert_float4(inputImage[c - 1 + 3*width]);
		float4 i36 = convert_float4(inputImage[c + 3*width]);
		float4 i46 = convert_float4(inputImage[c + 1 + 3*width]);
		float4 i56 = convert_float4(inputImage[c + 2 + 3*width]);
		float4 i66 = convert_float4(inputImage[c + 3 + 3*width]);
		
		Gx =   (float4)(5)*i03  +(float4)(5)*i11 +(float4)(18)*i12 +(float4)(32)*i13 +(float4)(18)*i14 +(float4)(5)*i15 +(float4)(18)*i21 +(float4)(64)*i22 +(float4)(100)*i23 +(float4)(64)*i24 +(float4)(18)*i25 +(float4)(5)*i30 +(float4)(32)*i31 +(float4)(100)*i32 +(float4)(100)*i33 +(float4)(100)* i34 +(float4)(32)* i35 +(float4)(5)*i36 +(float4)(18)*i41 +(float4)(64)*i42 +(float4)(100)*i43 +(float4)(64)*i44 +(float4)(18)*i45 +(float4)(5)*i51 +(float4)(18)*i52 +(float4)(32)*i53 +(float4)(18)*i54 +(float4)(5)*i55 +(float4)(5)*i63;

		outputImage[c] = convert_uchar4(Gx/(float4)(1048));
		

	}
			
}

	

	 






	

	




	

	

	
	
