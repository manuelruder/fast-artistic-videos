#include "luaT.h"
#include "THC.h"

#include "utils.c"

#include "BilinearSamplerBDHW.cu"

LUA_EXTERNC DLL_EXPORT int luaopen_libcustn(lua_State *L);

int luaopen_libcustn(lua_State *L)
{
  lua_newtable(L);
  cunn_BilinearSamplerBDHW_init(L);

  return 1;
}
