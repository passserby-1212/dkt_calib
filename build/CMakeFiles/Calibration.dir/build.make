# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xutao/桌面/dkt

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xutao/桌面/dkt/build

# Include any dependencies generated for this target.
include CMakeFiles/Calibration.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Calibration.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Calibration.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Calibration.dir/flags.make

CMakeFiles/Calibration.dir/src/camera.cpp.o: CMakeFiles/Calibration.dir/flags.make
CMakeFiles/Calibration.dir/src/camera.cpp.o: /home/xutao/桌面/dkt/src/camera.cpp
CMakeFiles/Calibration.dir/src/camera.cpp.o: CMakeFiles/Calibration.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/xutao/桌面/dkt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Calibration.dir/src/camera.cpp.o"
	/usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Calibration.dir/src/camera.cpp.o -MF CMakeFiles/Calibration.dir/src/camera.cpp.o.d -o CMakeFiles/Calibration.dir/src/camera.cpp.o -c /home/xutao/桌面/dkt/src/camera.cpp

CMakeFiles/Calibration.dir/src/camera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Calibration.dir/src/camera.cpp.i"
	/usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xutao/桌面/dkt/src/camera.cpp > CMakeFiles/Calibration.dir/src/camera.cpp.i

CMakeFiles/Calibration.dir/src/camera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Calibration.dir/src/camera.cpp.s"
	/usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xutao/桌面/dkt/src/camera.cpp -o CMakeFiles/Calibration.dir/src/camera.cpp.s

CMakeFiles/Calibration.dir/src/lidar.cpp.o: CMakeFiles/Calibration.dir/flags.make
CMakeFiles/Calibration.dir/src/lidar.cpp.o: /home/xutao/桌面/dkt/src/lidar.cpp
CMakeFiles/Calibration.dir/src/lidar.cpp.o: CMakeFiles/Calibration.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/xutao/桌面/dkt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/Calibration.dir/src/lidar.cpp.o"
	/usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Calibration.dir/src/lidar.cpp.o -MF CMakeFiles/Calibration.dir/src/lidar.cpp.o.d -o CMakeFiles/Calibration.dir/src/lidar.cpp.o -c /home/xutao/桌面/dkt/src/lidar.cpp

CMakeFiles/Calibration.dir/src/lidar.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Calibration.dir/src/lidar.cpp.i"
	/usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xutao/桌面/dkt/src/lidar.cpp > CMakeFiles/Calibration.dir/src/lidar.cpp.i

CMakeFiles/Calibration.dir/src/lidar.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Calibration.dir/src/lidar.cpp.s"
	/usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xutao/桌面/dkt/src/lidar.cpp -o CMakeFiles/Calibration.dir/src/lidar.cpp.s

CMakeFiles/Calibration.dir/src/registration.cpp.o: CMakeFiles/Calibration.dir/flags.make
CMakeFiles/Calibration.dir/src/registration.cpp.o: /home/xutao/桌面/dkt/src/registration.cpp
CMakeFiles/Calibration.dir/src/registration.cpp.o: CMakeFiles/Calibration.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/xutao/桌面/dkt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/Calibration.dir/src/registration.cpp.o"
	/usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Calibration.dir/src/registration.cpp.o -MF CMakeFiles/Calibration.dir/src/registration.cpp.o.d -o CMakeFiles/Calibration.dir/src/registration.cpp.o -c /home/xutao/桌面/dkt/src/registration.cpp

CMakeFiles/Calibration.dir/src/registration.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Calibration.dir/src/registration.cpp.i"
	/usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xutao/桌面/dkt/src/registration.cpp > CMakeFiles/Calibration.dir/src/registration.cpp.i

CMakeFiles/Calibration.dir/src/registration.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Calibration.dir/src/registration.cpp.s"
	/usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xutao/桌面/dkt/src/registration.cpp -o CMakeFiles/Calibration.dir/src/registration.cpp.s

CMakeFiles/Calibration.dir/src/run_group_extrinsic_calib.cpp.o: CMakeFiles/Calibration.dir/flags.make
CMakeFiles/Calibration.dir/src/run_group_extrinsic_calib.cpp.o: /home/xutao/桌面/dkt/src/run_group_extrinsic_calib.cpp
CMakeFiles/Calibration.dir/src/run_group_extrinsic_calib.cpp.o: CMakeFiles/Calibration.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/xutao/桌面/dkt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/Calibration.dir/src/run_group_extrinsic_calib.cpp.o"
	/usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Calibration.dir/src/run_group_extrinsic_calib.cpp.o -MF CMakeFiles/Calibration.dir/src/run_group_extrinsic_calib.cpp.o.d -o CMakeFiles/Calibration.dir/src/run_group_extrinsic_calib.cpp.o -c /home/xutao/桌面/dkt/src/run_group_extrinsic_calib.cpp

CMakeFiles/Calibration.dir/src/run_group_extrinsic_calib.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Calibration.dir/src/run_group_extrinsic_calib.cpp.i"
	/usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xutao/桌面/dkt/src/run_group_extrinsic_calib.cpp > CMakeFiles/Calibration.dir/src/run_group_extrinsic_calib.cpp.i

CMakeFiles/Calibration.dir/src/run_group_extrinsic_calib.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Calibration.dir/src/run_group_extrinsic_calib.cpp.s"
	/usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xutao/桌面/dkt/src/run_group_extrinsic_calib.cpp -o CMakeFiles/Calibration.dir/src/run_group_extrinsic_calib.cpp.s

CMakeFiles/Calibration.dir/src/run_intrinsic_calib.cpp.o: CMakeFiles/Calibration.dir/flags.make
CMakeFiles/Calibration.dir/src/run_intrinsic_calib.cpp.o: /home/xutao/桌面/dkt/src/run_intrinsic_calib.cpp
CMakeFiles/Calibration.dir/src/run_intrinsic_calib.cpp.o: CMakeFiles/Calibration.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/xutao/桌面/dkt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/Calibration.dir/src/run_intrinsic_calib.cpp.o"
	/usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Calibration.dir/src/run_intrinsic_calib.cpp.o -MF CMakeFiles/Calibration.dir/src/run_intrinsic_calib.cpp.o.d -o CMakeFiles/Calibration.dir/src/run_intrinsic_calib.cpp.o -c /home/xutao/桌面/dkt/src/run_intrinsic_calib.cpp

CMakeFiles/Calibration.dir/src/run_intrinsic_calib.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Calibration.dir/src/run_intrinsic_calib.cpp.i"
	/usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xutao/桌面/dkt/src/run_intrinsic_calib.cpp > CMakeFiles/Calibration.dir/src/run_intrinsic_calib.cpp.i

CMakeFiles/Calibration.dir/src/run_intrinsic_calib.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Calibration.dir/src/run_intrinsic_calib.cpp.s"
	/usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xutao/桌面/dkt/src/run_intrinsic_calib.cpp -o CMakeFiles/Calibration.dir/src/run_intrinsic_calib.cpp.s

CMakeFiles/Calibration.dir/src/run_lidars_calib.cpp.o: CMakeFiles/Calibration.dir/flags.make
CMakeFiles/Calibration.dir/src/run_lidars_calib.cpp.o: /home/xutao/桌面/dkt/src/run_lidars_calib.cpp
CMakeFiles/Calibration.dir/src/run_lidars_calib.cpp.o: CMakeFiles/Calibration.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/xutao/桌面/dkt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/Calibration.dir/src/run_lidars_calib.cpp.o"
	/usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Calibration.dir/src/run_lidars_calib.cpp.o -MF CMakeFiles/Calibration.dir/src/run_lidars_calib.cpp.o.d -o CMakeFiles/Calibration.dir/src/run_lidars_calib.cpp.o -c /home/xutao/桌面/dkt/src/run_lidars_calib.cpp

CMakeFiles/Calibration.dir/src/run_lidars_calib.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Calibration.dir/src/run_lidars_calib.cpp.i"
	/usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xutao/桌面/dkt/src/run_lidars_calib.cpp > CMakeFiles/Calibration.dir/src/run_lidars_calib.cpp.i

CMakeFiles/Calibration.dir/src/run_lidars_calib.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Calibration.dir/src/run_lidars_calib.cpp.s"
	/usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xutao/桌面/dkt/src/run_lidars_calib.cpp -o CMakeFiles/Calibration.dir/src/run_lidars_calib.cpp.s

CMakeFiles/Calibration.dir/src/run_single_extrinsic_calib.cpp.o: CMakeFiles/Calibration.dir/flags.make
CMakeFiles/Calibration.dir/src/run_single_extrinsic_calib.cpp.o: /home/xutao/桌面/dkt/src/run_single_extrinsic_calib.cpp
CMakeFiles/Calibration.dir/src/run_single_extrinsic_calib.cpp.o: CMakeFiles/Calibration.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/xutao/桌面/dkt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/Calibration.dir/src/run_single_extrinsic_calib.cpp.o"
	/usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Calibration.dir/src/run_single_extrinsic_calib.cpp.o -MF CMakeFiles/Calibration.dir/src/run_single_extrinsic_calib.cpp.o.d -o CMakeFiles/Calibration.dir/src/run_single_extrinsic_calib.cpp.o -c /home/xutao/桌面/dkt/src/run_single_extrinsic_calib.cpp

CMakeFiles/Calibration.dir/src/run_single_extrinsic_calib.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Calibration.dir/src/run_single_extrinsic_calib.cpp.i"
	/usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xutao/桌面/dkt/src/run_single_extrinsic_calib.cpp > CMakeFiles/Calibration.dir/src/run_single_extrinsic_calib.cpp.i

CMakeFiles/Calibration.dir/src/run_single_extrinsic_calib.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Calibration.dir/src/run_single_extrinsic_calib.cpp.s"
	/usr/bin/g++-8 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xutao/桌面/dkt/src/run_single_extrinsic_calib.cpp -o CMakeFiles/Calibration.dir/src/run_single_extrinsic_calib.cpp.s

# Object files for target Calibration
Calibration_OBJECTS = \
"CMakeFiles/Calibration.dir/src/camera.cpp.o" \
"CMakeFiles/Calibration.dir/src/lidar.cpp.o" \
"CMakeFiles/Calibration.dir/src/registration.cpp.o" \
"CMakeFiles/Calibration.dir/src/run_group_extrinsic_calib.cpp.o" \
"CMakeFiles/Calibration.dir/src/run_intrinsic_calib.cpp.o" \
"CMakeFiles/Calibration.dir/src/run_lidars_calib.cpp.o" \
"CMakeFiles/Calibration.dir/src/run_single_extrinsic_calib.cpp.o"

# External object files for target Calibration
Calibration_EXTERNAL_OBJECTS =

/home/xutao/桌面/dkt/lib/libCalibration.a: CMakeFiles/Calibration.dir/src/camera.cpp.o
/home/xutao/桌面/dkt/lib/libCalibration.a: CMakeFiles/Calibration.dir/src/lidar.cpp.o
/home/xutao/桌面/dkt/lib/libCalibration.a: CMakeFiles/Calibration.dir/src/registration.cpp.o
/home/xutao/桌面/dkt/lib/libCalibration.a: CMakeFiles/Calibration.dir/src/run_group_extrinsic_calib.cpp.o
/home/xutao/桌面/dkt/lib/libCalibration.a: CMakeFiles/Calibration.dir/src/run_intrinsic_calib.cpp.o
/home/xutao/桌面/dkt/lib/libCalibration.a: CMakeFiles/Calibration.dir/src/run_lidars_calib.cpp.o
/home/xutao/桌面/dkt/lib/libCalibration.a: CMakeFiles/Calibration.dir/src/run_single_extrinsic_calib.cpp.o
/home/xutao/桌面/dkt/lib/libCalibration.a: CMakeFiles/Calibration.dir/build.make
/home/xutao/桌面/dkt/lib/libCalibration.a: CMakeFiles/Calibration.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/xutao/桌面/dkt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX static library /home/xutao/桌面/dkt/lib/libCalibration.a"
	$(CMAKE_COMMAND) -P CMakeFiles/Calibration.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Calibration.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Calibration.dir/build: /home/xutao/桌面/dkt/lib/libCalibration.a
.PHONY : CMakeFiles/Calibration.dir/build

CMakeFiles/Calibration.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Calibration.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Calibration.dir/clean

CMakeFiles/Calibration.dir/depend:
	cd /home/xutao/桌面/dkt/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xutao/桌面/dkt /home/xutao/桌面/dkt /home/xutao/桌面/dkt/build /home/xutao/桌面/dkt/build /home/xutao/桌面/dkt/build/CMakeFiles/Calibration.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/Calibration.dir/depend

