project(svn_test_gt)
find_package( OpenCV REQUIRED )
add_executable(svn_test_gt main.cpp evaluation_criteria.cpp utilities.cpp )

target_link_libraries(svn_test_gt  ${OpenCV_LIBS})
