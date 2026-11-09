#include "nanlm/nan16.h"
#include <math.h>
#include <stdio.h>
int main(void){
 double worst=0;
 for(unsigned p=0;p<65536;++p){
  double e=fabs(nan_sin((uint16_t)p)/32767.-sin(p*6.2831853071795864769/65536.));
  if(e>worst)worst=e;
 }
 if(worst>3.5e-5)return 1;
 for(unsigned n=0;n<16;++n){uint8_t bytes[8];uint32_t payload=0;
  nan_pack(bytes,n,0xdeadbeef);
  if(nan_unpack(bytes,&payload)!=(int)n||payload!=0xdeadbeef)return 2;
 }
 printf("PASS C11 phase/carrier API; error=%.9g\n",worst);
 return 0;
}
