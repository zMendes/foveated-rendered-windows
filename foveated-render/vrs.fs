#version 460
#extension GL_NV_shading_rate_image : require

struct Material {
    sampler2D texture_diffuse1;
    sampler2D texture_specular1;
    float shininess;

    bool hasDiffuseTexture;
    bool hasSpecularTexture;
    vec3 diffuseColor;
    vec3 specularColor;
};
struct DirLight {
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    bool hasDiffuseTexture;
    vec3 diffuseColor;
};

struct PointLight {
    vec3 position;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    float constant;
    float linear;
    float quadratic;
};

#define NR_POINT_LIGHTS 4

in vec3 normal;
in vec3 fragPos;
in vec2 texCoords;

uniform vec3 viewPos;
uniform Material material;
uniform DirLight dirLight;
uniform PointLight pointLights[NR_POINT_LIGHTS];
uniform bool showShading;

out vec4 FragColor;

vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir);
vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir);

void main() {
    vec3 norm = normalize(normal);
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 result = CalcDirLight(dirLight, norm, viewDir);
    //for(int i = 0; i < NR_POINT_LIGHTS; i++)
    //result += CalcPointLight(pointLights[i], norm, fragPos, viewDir);

    FragColor = vec4(result, 1.0);

    if (showShading) {
        int maxCoarse = max(gl_FragmentSizeNV.x, gl_FragmentSizeNV.y);

        if (maxCoarse == 1) {
            FragColor = mix(FragColor, vec4(1,0,0,1), 0.2);
        }
        else if (maxCoarse == 2) {
            FragColor= mix(FragColor,vec4(1,1,0,1), 0.2);
        }
        else if (maxCoarse == 4) {
            FragColor= mix(FragColor,vec4(0,1,0,1), 0.2);
        }
        else {
            FragColor= mix(FragColor,vec4(1,1,1,1), 0.2);
        }
    }
}
vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir) {
    vec3 lightDir = normalize(-light.direction);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);

    vec3 baseDiffuse = material.hasDiffuseTexture ? vec3(texture(material.texture_diffuse1, texCoords)) : material.diffuseColor;
    vec3 baseSpecular = material.hasSpecularTexture ? vec3(texture(material.texture_specular1, texCoords)) : material.specularColor;

    vec3 ambient = light.ambient * baseDiffuse;
    vec3 diffuse = light.diffuse * diff * baseDiffuse;
    vec3 specular = light.specular * spec * baseSpecular;

    return ambient + diffuse + specular;
}

vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
    vec3 lightDir = normalize(light.position - fragPos);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));

    vec3 baseDiffuse = material.hasDiffuseTexture ? vec3(texture(material.texture_diffuse1, texCoords)) : material.diffuseColor;
    vec3 baseSpecular = material.hasSpecularTexture ? vec3(texture(material.texture_specular1, texCoords)) : material.specularColor;

    vec3 ambient = light.ambient * baseDiffuse;
    vec3 diffuse = light.diffuse * diff * baseDiffuse;
    vec3 specular = light.specular * spec * baseSpecular;

    ambient *= attenuation;
    diffuse *= attenuation;
    specular *= attenuation;

    return ambient + diffuse + specular;
}