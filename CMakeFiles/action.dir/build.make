# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.11

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/augenepark/HackUVA2018

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/augenepark/HackUVA2018

# Include any dependencies generated for this target.
include CMakeFiles/action.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/action.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/action.dir/flags.make

CMakeFiles/action.dir/action.o: CMakeFiles/action.dir/flags.make
CMakeFiles/action.dir/action.o: action.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/augenepark/HackUVA2018/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/action.dir/action.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/action.dir/action.o -c /Users/augenepark/HackUVA2018/action.cpp

CMakeFiles/action.dir/action.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/action.dir/action.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/augenepark/HackUVA2018/action.cpp > CMakeFiles/action.dir/action.i

CMakeFiles/action.dir/action.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/action.dir/action.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/augenepark/HackUVA2018/action.cpp -o CMakeFiles/action.dir/action.s

# Object files for target action
action_OBJECTS = \
"CMakeFiles/action.dir/action.o"

# External object files for target action
action_EXTERNAL_OBJECTS =

action: CMakeFiles/action.dir/action.o
action: CMakeFiles/action.dir/build.make
action: /usr/local/lib/libopencv_dnn.3.4.1.dylib
action: /usr/local/lib/libopencv_ml.3.4.1.dylib
action: /usr/local/lib/libopencv_objdetect.3.4.1.dylib
action: /usr/local/lib/libopencv_shape.3.4.1.dylib
action: /usr/local/lib/libopencv_stitching.3.4.1.dylib
action: /usr/local/lib/libopencv_superres.3.4.1.dylib
action: /usr/local/lib/libopencv_videostab.3.4.1.dylib
action: /usr/local/lib/libopencv_calib3d.3.4.1.dylib
action: /usr/local/lib/libopencv_features2d.3.4.1.dylib
action: /usr/local/lib/libopencv_flann.3.4.1.dylib
action: /usr/local/lib/libopencv_highgui.3.4.1.dylib
action: /usr/local/lib/libopencv_photo.3.4.1.dylib
action: /usr/local/lib/libopencv_video.3.4.1.dylib
action: /usr/local/lib/libopencv_videoio.3.4.1.dylib
action: /usr/local/lib/libopencv_imgcodecs.3.4.1.dylib
action: /usr/local/lib/libopencv_imgproc.3.4.1.dylib
action: /usr/local/lib/libopencv_core.3.4.1.dylib
action: CMakeFiles/action.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/augenepark/HackUVA2018/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable action"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/action.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/action.dir/build: action

.PHONY : CMakeFiles/action.dir/build

CMakeFiles/action.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/action.dir/cmake_clean.cmake
.PHONY : CMakeFiles/action.dir/clean

CMakeFiles/action.dir/depend:
	cd /Users/augenepark/HackUVA2018 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/augenepark/HackUVA2018 /Users/augenepark/HackUVA2018 /Users/augenepark/HackUVA2018 /Users/augenepark/HackUVA2018 /Users/augenepark/HackUVA2018/CMakeFiles/action.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/action.dir/depend

