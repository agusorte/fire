project(saturation_seg)
find_package( OpenCV REQUIRED )
add_executable(saturation_seg main.cpp)

target_link_libraries(saturation_seg  ${OpenCV_LIBS} )
