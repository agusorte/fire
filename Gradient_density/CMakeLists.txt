project(gradient)
find_package( OpenCV REQUIRED )
add_executable(gradient main.cpp)

target_link_libraries(gradient  ${OpenCV_LIBS} )
