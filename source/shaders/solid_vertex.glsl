#version 450
#pragma shader_stage(vertex)

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in mat4 matrix;

layout(location = 0) out vec3 vertex_normal;

void main() {
    gl_Position = matrix * vec4(position, 1.0);
    vertex_normal = normalize(normal);
}
