#version 460 core

out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D texHigh;
uniform sampler2D texMedium;
uniform sampler2D texLow;

uniform vec2 gaze;
uniform vec2 predicted;

uniform bool showShading;
uniform bool isSaccade;

void main() {
    vec2 gaze_center = isSaccade ? predicted : (gaze + 1.0) * 0.5;
    float foveation = isSaccade ? 0.3 : 0.1;
    float dist = distance(TexCoords, gaze_center);

    if (dist < foveation)
        FragColor = texture(texHigh, TexCoords);
    else if (dist < 0.5)
        FragColor = texture(texMedium, TexCoords);
    else
        FragColor = texture(texLow, TexCoords);

    if (showShading) {
        vec4 shade = vec4(0.0, 1.0, 0.0, 1.0);
        if (dist < foveation)
            shade = vec4(1.0, 0.0, 0.0, 1.0);
        else if (dist < 0.4)
            shade = vec4(1.0, 1.0, 0.0, 1.0);
        FragColor *= shade;

        if (distance(TexCoords, predicted) < 0.001)
            FragColor = vec4(1.0);
    }
}