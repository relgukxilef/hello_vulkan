cmake_minimum_required(VERSION 3.5)

find_program(GLSLC glslc)

add_subdirectory(game_engine1)

add_executable(vulkan main.cpp)

target_link_libraries(vulkan game_engine1_vulkan)


function(add_shader TARGET SHADER)
    find_program(GLSLC glslc)

    set(current-shader-path ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER})
    set(current-output-path ${CMAKE_BINARY_DIR}/${SHADER})

    # Add a custom command to compile GLSL to SPIR-V.
    get_filename_component(current-output-dir ${current-output-path} DIRECTORY)
    file(MAKE_DIRECTORY ${current-output-dir})

    add_custom_command(
        OUTPUT ${current-output-path}.o
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMAND ${GLSLC} -o ${SHADER}.spv ${current-shader-path}
        COMMAND ld -r -b binary -o ${SHADER}.o ${SHADER}.spv
        DEPENDS ${current-shader-path}
        IMPLICIT_DEPENDS CXX ${current-shader-path}
        VERBATIM
    )

    # Make sure our build depends on this output.
    set_source_files_properties(
        ${current-output-path}.o PROPERTIES GENERATED TRUE
    )
    # shader is only listed here to make it show up in Qt Creator
    target_sources(
        ${TARGET} PRIVATE ${current-output-path}.o ${current-shader-path}
    )
endfunction(add_shader)

add_shader(vulkan shaders/triangle_vertex.glsl)
add_shader(vulkan shaders/black_fragment.glsl)
