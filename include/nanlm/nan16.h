#ifndef NANLM_NAN16_H
#define NANLM_NAN16_H
/* Portable integer phase and NaN0S carrier primitives, usable from C11/C++20. */
#include <stdint.h>
#include <stddef.h>
static const int16_t nan_sine[257]={0,201,402,603,804,1005,1206,1407,1608,1809,2009,2210,2410,2611,2811,3012,3212,3412,3612,3811,4011,4210,4410,4609,4808,5007,5205,5404,5602,5800,5998,6195,6393,6590,6786,6983,7179,7375,7571,7767,7962,8157,8351,8545,8739,8933,9126,9319,9512,9704,9896,10087,10278,10469,10659,10849,11039,11228,11417,11605,11793,11980,12167,12353,12539,12725,12910,13094,13279,13462,13645,13828,14010,14191,14372,14553,14732,14912,15090,15269,15446,15623,15800,15976,16151,16325,16499,16673,16846,17018,17189,17360,17530,17700,17869,18037,18204,18371,18537,18703,18868,19032,19195,19357,19519,19680,19841,20000,20159,20317,20475,20631,20787,20942,21096,21250,21403,21554,21705,21856,22005,22154,22301,22448,22594,22739,22884,23027,23170,23311,23452,23592,23731,23870,24007,24143,24279,24413,24547,24680,24811,24942,25072,25201,25329,25456,25582,25708,25832,25955,26077,26198,26319,26438,26556,26674,26790,26905,27019,27133,27245,27356,27466,27575,27683,27790,27896,28001,28105,28208,28310,28411,28510,28609,28706,28803,28898,28992,29085,29177,29268,29358,29447,29534,29621,29706,29791,29874,29956,30037,30117,30195,30273,30349,30424,30498,30571,30643,30714,30783,30852,30919,30985,31050,31113,31176,31237,31297,31356,31414,31470,31526,31580,31633,31685,31736,31785,31833,31880,31926,31971,32014,32057,32098,32137,32176,32213,32250,32285,32318,32351,32382,32412,32441,32469,32495,32521,32545,32567,32589,32609,32628,32646,32663,32678,32692,32705,32717,32728,32737,32745,32752,32757,32761,32765,32766,32767};
static inline int16_t nan_sin(uint16_t p){
 uint16_t i=p&0x3fff;
 if(p&0x4000)i=(uint16_t)(0x4000-i);
 uint16_t k=i>>6,f=i&63;
 int32_t v=nan_sine[k];
 if(f) v+=((int32_t)(nan_sine[k+1]-nan_sine[k])*f+32)/64;
 return (int16_t)((p&0x8000)?-v:v);
}
static inline int16_t nan_cos(uint16_t p){return nan_sin((uint16_t)(p+0x4000u));}
/* Integer vectoring CORDIC; (0,0) is defined as zero. int64 intermediates
   avoid overflow even for INT32_MIN; division gives portable signed rounding. */
static inline uint16_t nan_atan2(int32_t y,int32_t x){
 static const int16_t angles[16]={8192,4836,2555,1297,651,326,163,81,41,20,10,5,3,1,1,0};
 if(!x&&!y)return 0;
 int64_t xx=(int64_t)x*16777216,yy=(int64_t)y*16777216;
 int32_t p=0;
 if(xx<0){xx=-xx;yy=-yy;p=32768;}
 for(unsigned i=0;i<16&&yy;++i){
  int64_t dx=xx/((int64_t)1<<i),dy=yy/((int64_t)1<<i);
  if(yy>0){xx+=dy;yy-=dx;p+=angles[i];}
  else{xx-=dy;yy+=dx;p-=angles[i];}
 }
 return (uint16_t)p;
}
/* 32-bit fractional accumulator: upper half is A, lower half is C. */
static inline uint32_t nan_phase32(uint16_t a,uint16_t c){return ((uint32_t)a<<16)|c;}
static inline uint32_t nan_bool(unsigned n,uint32_t a,uint32_t b){
 uint32_t r=0;
 if(n&1)r|=~a&~b;
 if(n&2)r|=~a&b;
 if(n&4)r|=a&~b;
 if(n&8)r|=a&b;
 return r;
}
static inline uint16_t nan_head(unsigned c){
 return (uint16_t)((c&1?0x8000:0)|(c&2?0x7ff8:0x7ff1));
}
static inline int nan_class(uint16_t h){
 switch(h){case 0x7ff1:return 0;case 0xfff1:return 1;case 0x7ff8:return 2;case 0xfff8:return 3;default:return -1;}
}
/* Explicit memory bytes, independent of host endian and floating-point NaNs. */
static inline void nan_pack(uint8_t out[8],unsigned n,uint32_t payload){
 uint16_t nearh=nan_head(n>>2),farh=nan_head(n&3);
 out[0]=(uint8_t)(farh>>8);out[1]=(uint8_t)farh;
 for(unsigned i=0;i<4;++i)out[i+2]=(uint8_t)(payload>>(8*i));
 out[6]=(uint8_t)nearh;out[7]=(uint8_t)(nearh>>8);
}
static inline int nan_unpack(const uint8_t in[8],uint32_t *payload){
 int a=nan_class((uint16_t)(in[6]|((uint16_t)in[7]<<8)));
 int b=nan_class((uint16_t)(((uint16_t)in[0]<<8)|in[1]));
 if(a<0||b<0)return -1;
 *payload=0;
 for(unsigned i=0;i<4;++i)*payload|=(uint32_t)in[i+2]<<(8*i);
 return (a<<2)|b;
}
#endif
