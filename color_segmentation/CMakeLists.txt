project(color_seg)
find_package( OpenCV REQUIRED )
add_executable(color_seg main.cpp)

target_link_libraries(color_seg  ${OpenCV_LIBS} )
