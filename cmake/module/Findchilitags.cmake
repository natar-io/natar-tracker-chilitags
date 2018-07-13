# - Find CHILITAGS library
# Once done this will define
#
#  CHILITAGS_FOUND - This defines if we found CHILITAGS
#  CHILITAGS_INCLUDE_DIR - CHILITAGS include directory
#  CHILITAGS_LIBS - CHILITAGS libraries
#  CHILITAGS_DEFINITIONS - Compiler switches required for CHILITAGS


# use pkg-config to get the directories and then use these values
# in the FIND_PATH() and FIND_LIBRARY() calls
#FIND_PACKAGE(PkgConfig)
#PKG_SEARCH_MODULE(PC_LIBCHILITAGS libCHILITAGS)

SET(CHILITAGS_DEFINITIONS ${PC_CHILITAGS_CFLAGS_OTHER})

FIND_PATH(
    CHILITAGS_INCLUDE_DIR
    NAMES "chilitags.hpp"
    PATH_SUFFIXES "chilitags"
)

FIND_LIBRARY(CHILITAGS_LIBS NAMES chilitags
   HINTS
   ${PC_CHILITAGS_LIBDIR}
   ${PC_CHILITAGS_LIBRARY_DIRS}
)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CHILITAGS DEFAULT_MSG CHILITAGS_LIBS CHILITAGS_INCLUDE_DIR)