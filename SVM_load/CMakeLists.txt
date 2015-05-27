project(svn_load)
find_package( OpenCV REQUIRED )
add_executable(svn_load main.cpp utilities.cpp)

target_link_libraries(svn_load  ${OpenCV_LIBS})
