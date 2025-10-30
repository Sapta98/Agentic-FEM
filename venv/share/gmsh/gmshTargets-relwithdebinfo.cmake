#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "gmsh::shared" for configuration "RelWithDebInfo"
set_property(TARGET gmsh::shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(gmsh::shared PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/libgmsh.4.14.1.dylib"
  IMPORTED_SONAME_RELWITHDEBINFO "@rpath/libgmsh.4.14.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS gmsh::shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_gmsh::shared "${_IMPORT_PREFIX}/lib/libgmsh.4.14.1.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
