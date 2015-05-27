project(Eval_Seg)
find_package( OpenCV REQUIRED )
add_executable(Eval_Seg main.cpp)

target_link_libraries(Eval_Seg  ${OpenCV_LIBS} )
