#pragma once
#include <cstdint>
#include <string>

namespace nanlm {

// Strict UTF-8 prefix automaton. It rejects overlong encodings, UTF-16
// surrogate codepoints, C0/C1, F5..FF, and out-of-range continuation bytes.
// It accepts partial prefixes while sampling, but a final string must end
// with no outstanding continuation bytes.
struct Utf8Prefix {
 unsigned remaining=0;
 uint8_t lowest=0x80;
 uint8_t highest=0xbf;

 bool permits(uint8_t byte,size_t slots=4)const {
  if(remaining){
   return remaining<=slots && byte>=lowest && byte<=highest;
  }
  if(byte<=0x7f)return true;
  if(byte>=0xc2&&byte<=0xdf)return slots>=2;
  if(byte>=0xe0&&byte<=0xef)return slots>=3;
  if(byte>=0xf0&&byte<=0xf4)return slots>=4;
  return false;
 }
 bool push(uint8_t byte,size_t slots=4){
  if(!permits(byte,slots))return false;
  if(remaining){
   --remaining;
   lowest=0x80;highest=0xbf;
   return true;
  }
  if(byte<=0x7f)return true;
  if(byte<=0xdf){remaining=1;return true;}
  if(byte==0xe0){remaining=2;lowest=0xa0;highest=0xbf;return true;}
  if(byte==0xed){remaining=2;lowest=0x80;highest=0x9f;return true;}
  if(byte<=0xef){remaining=2;return true;}
  if(byte==0xf0){remaining=3;lowest=0x90;highest=0xbf;return true;}
  if(byte==0xf4){remaining=3;lowest=0x80;highest=0x8f;return true;}
  remaining=3;
  return true;
 }
 bool complete()const{return remaining==0;}
 bool consume_prefix(const std::string& bytes){
  for(unsigned char b:bytes)if(!push(b))return false;
  return true;
 }
};
inline bool valid_utf8(const std::string& text){
 Utf8Prefix state;
 return state.consume_prefix(text)&&state.complete();
}
} // namespace nanlm
