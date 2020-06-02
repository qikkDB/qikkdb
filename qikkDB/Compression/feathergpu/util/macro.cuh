#pragma once
//TODO: better distinguishing between signed/unsigned versions
#define CWORD_SIZE(T)(T) (sizeof(T) * 8)

#define NBITSTOMASK(n) ((1u<<(n)) - 1)
#define LNBITSTOMASK(n) ((1UL<<(n)) - 1)
#define LLNBITSTOMASK(n) ((1ULL<<(n)) - 1)

#define BITMASK(T, n) ((((T) 1u) <<(n)) - 1)

#define BIT_SET(a,b) ((a) |= (1ULL<<(b)))
#define BIT_CLEAR(a,b) ((a) &= ~(1ULL<<(b)))
#define BIT_FLIP(a,b) ((a) ^= (1ULL<<(b)))
#define BIT_CHECK(a,b) ((a) & (1ULL<<(b)))


// OLD (re)move
/* #define fillto(b,c) (((c + b - 1) / b) * b) */
/* #define _unused(x) x __attribute__((unused)) */
/* #define convert_struct(n, s)  struct sgn {signed int x:n;} __attribute__((unused)) s */
/* #define SGN(a) (int)((unsigned int)((int)a) >> (sizeof(int) * CHAR_BIT - 1)) */
/* #define GETNSGNBITS(a,n,b) ((SGN(a) << (n-1)) | GETNBITS(((a)>>(b-n)), (n-1))) */
//end OLD

// Make a FOREACH macro
#define FE_1(WHAT, X) WHAT(X)
#define FE_2(WHAT, X, ...) WHAT(X)FE_1(WHAT, __VA_ARGS__)
#define FE_3(WHAT, X, ...) WHAT(X)FE_2(WHAT, __VA_ARGS__)
#define FE_4(WHAT, X, ...) WHAT(X)FE_3(WHAT, __VA_ARGS__)
#define FE_5(WHAT, X, ...) WHAT(X)FE_4(WHAT, __VA_ARGS__)
#define FE_6(WHAT, X, ...) WHAT(X)FE_5(WHAT, __VA_ARGS__)
//... repeat as needed

#define GET_MACRO(_1,_2,_3,_4,_5,NAME,...) NAME
#define FOR_EACH(action,...) \
  GET_MACRO(__VA_ARGS__,FE_5,FE_4,FE_3,FE_2,FE_1)(action,__VA_ARGS__)
