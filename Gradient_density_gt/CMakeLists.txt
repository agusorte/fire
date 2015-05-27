project(gradient_gt)
find_package( OpenCV REQUIRED )
add_executable(gradient_gt evaluation_criteria.cpp utilities.cpp main.cpp)

target_link_libraries(gradient_gt  ${OpenCV_LIBS} )
