#version 450
#pragma shader_stage(fragment)

layout(location = 0) in vec3 vertex_normal;

layout(location = 0) out vec3 color;

void main() {
    color = vec3(0);
    color = vec3(max(dot(normalize(vec3(1, -1, 1)), vertex_normal), 0.0));
    color += vec3(max(dot(normalize(vec3(-1, -1, 1)), vertex_normal), 0.0));
    color *= vertex_normal * 0.5 + 0.5;

    color = pow(color, vec3(1.0 / 2.2));
}
