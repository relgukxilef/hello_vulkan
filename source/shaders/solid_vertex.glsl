#version 450
#pragma shader_stage(vertex)

layout(location = 0) in vec2 position;
layout(location = 1) in vec3 color;
layout(location = 2) in mat4 matrix;

layout(location = 0) out vec3 vertex_color;

void main() {
    gl_Position = matrix * vec4(position, 0.0, 1.0);
    vertex_color = color;
}
