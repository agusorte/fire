project(gmm)
find_package( OpenCV REQUIRED )
add_executable(gmm main.cpp)

target_link_libraries(gmm  ${OpenCV_LIBS} )
