#version 450
#pragma shader_stage(fragment)

layout(location = 0) in vec3 vertex_color;

layout(location = 0) out vec4 color;

void main() {
    color = vec4(pow(vertex_color, vec3(1.0 / 2.2)), 1.0);
}
