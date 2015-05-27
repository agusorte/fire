project(svn_test)
find_package( OpenCV REQUIRED )
add_executable(svn_test main.cpp utilities.cpp)

target_link_libraries(svn_test  ${OpenCV_LIBS})
