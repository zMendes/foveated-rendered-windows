#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <GLFW/glfw3.h>

#include <onnxruntime_cxx_api.h>
#include <tobii_gameintegration.h>

#include "shader.h"
#include "camera.h"
#include "stb_image.h"
#include "model.h"
#include "constants.h"

#include <iostream>
#include <algorithm>
#include <vector>
#include <deque>
#include <cmath>
#include <utility>
#include <chrono>

typedef struct
{
    unsigned int fbo;
    unsigned int texture;
    unsigned int rbo;
} FBO;

typedef void(APIENTRYP PFNGLBINDSHADINGRATEIMAGENVPROC)(GLuint texture);
PFNGLBINDSHADINGRATEIMAGENVPROC glBindShadingRateImageNV = nullptr;

typedef void(APIENTRYP PFNGLSHADINGRATEIMAGEPALETTENVPROC)(GLuint viewport, GLuint first, GLsizei count, const GLenum* rates);
PFNGLSHADINGRATEIMAGEPALETTENVPROC glShadingRateImagePaletteNV = nullptr;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
FBO createFBO(int width, int height);
void renderScene(Shader& shader, Model model);
glm::vec2 gazeAngleToNorm(float x_deg, float y_deg);
std::pair<float, float> pixelsToDegreesFromNormalized(float norm_x, float norm_y);
float angleToNormRadius(float deg, float diagInInches, float distMM, int scrWidth, int scrHeight);
void createFoveationTexture(glm::vec2 point, float error);
void uploadFoveationDataToTexture(GLuint texture);
void setupShadingRatePalette();
void createTexture(GLuint& glid);
bool InitNVShadingRateImageExtensions();
template <typename T>
bool LoadGLFunction(T& funcPtr, const char* name);
// settings
const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;
float diagonal_in_inches = 17.0f;

float diag_px = std::sqrt(SCR_WIDTH * SCR_WIDTH + SCR_HEIGHT * SCR_HEIGHT);
float diag_mm = diagonal_in_inches * 25.4f;
float SCR_WIDTH_MM = diag_mm * (SCR_WIDTH / diag_px);
float SCR_HEIGHT_MM = diag_mm * (SCR_HEIGHT / diag_px);
float DIST_MM = 600.0f;
float ASPECT_RATIO = (float)SCR_WIDTH / (float)SCR_HEIGHT;
float near = 0.1f;
float far = 10000.0f;
float INNER_R_DEG = 6.5f;
float MIDDLE_R_DEG = 14.25f;
float INNER_R = angleToNormRadius(INNER_R_DEG, diagonal_in_inches, DIST_MM, SCR_WIDTH, SCR_HEIGHT);
float MIDDLE_R = angleToNormRadius(MIDDLE_R_DEG, diagonal_in_inches, DIST_MM, SCR_WIDTH, SCR_HEIGHT);

float posX = 0.5;
float posY = 0.5;

// VRS stuff
GLuint fov_texture;
std::vector<uint8_t> m_shadingRateImageData;
uint32_t m_shadingRateImageWidth = 0;
uint32_t m_shadingRateImageHeight = 0;
GLint m_shadingRateImageTexelWidth;
GLint m_shadingRateImageTexelHeight;

// CAMERA
Camera camera(glm::vec3(0.0f, 2.0f, 8.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, 0.0f);
bool firstMouse = true;
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

bool showShading = false;
bool isCursorEnabled = false;
bool isSaccade = false;
bool isLastSaccade = false;

float PRECISION_DEG = 1.01f; //https://link.springer.com/chapter/10.1007/978-3-030-98404-5_36
float PRECISION = angleToNormRadius(PRECISION_DEG, diagonal_in_inches, DIST_MM, SCR_WIDTH, SCR_HEIGHT);

using namespace TobiiGameIntegration;

int main()
{
    //Eye tracking data
    ITobiiGameIntegrationApi* api = GetApi("Gaze Sample");
    IStreamsProvider* streamsProvider = api->GetStreamsProvider();
    ITrackerController* trackerController = api->GetTrackerController();

    api->GetTrackerController()->TrackRectangle({ 0,0,SCR_WIDTH,SCR_HEIGHT });
    const GazePoint* gazePoints = nullptr;
    TrackerInfo info;
    bool success = trackerController->GetTrackerInfo(info);

    if (success) {
        std::cout << "=== Tobii Eye Tracker Info ===" << std::endl;

        std::cout << "Model: " << info.ModelName << std::endl;
    } else {
        std::cerr << "Failed to retrieve tracker info." << std::endl;
    }

    std::deque<std::array<float, 2>> gaze_history;
    glm::vec2 predicted;
    std::pair<float, float> predicted_deg;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "saccade_predictor");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 3> input_shape = { 1, 10, 2 };
    const char* input_names[] = { "input" };
    const char* output_names[] = { "output" };

    // Load the model
    const wchar_t* model_path = L"C:/Users/loenardomm8/Documents/saccade_predictor.onnx";
    Ort::Session session(env, model_path, session_options);

    // glfw: initialize and configure
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);



    // glfw window creation
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Foveated render", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);


    if (!InitNVShadingRateImageExtensions()) {
        std::cerr << "Failed to initialize required NV shading rate extensions!" << std::endl;
    }

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1; // Correct return here, exiting if GLAD fails
    }

    // OPENGL STATE
    glEnable(GL_DEPTH_TEST);
    glEnable(NVShadingRate::IMAGE);


    glGetIntegerv(NVShadingRate::TEXEL_HEIGHT, &m_shadingRateImageTexelHeight);
    glGetIntegerv(NVShadingRate::TEXEL_WIDTH, &m_shadingRateImageTexelWidth);

    m_shadingRateImageWidth = (SCR_WIDTH + m_shadingRateImageTexelWidth - 1) / m_shadingRateImageTexelWidth;
    m_shadingRateImageHeight = (SCR_HEIGHT + m_shadingRateImageTexelHeight - 1) / m_shadingRateImageTexelHeight;
    m_shadingRateImageData.resize(m_shadingRateImageWidth * m_shadingRateImageHeight);

    createTexture(fov_texture);
    setupShadingRatePalette();

    Shader shader("vrs.vs", "vrs.fs");
    Shader screenShader("screen.vs", "screen.fs");
    shader.use();

    std::string path = "C:\\Users\\loenardomm8\\Documents\\sponza\\sponza.obj";

    Model conference(path);

    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR)
    {
        std::cerr << "OpenGL error before main loop: " << err << std::endl;
    }

    // QUAD VAO
    float quadVertices[] = {
        // positions   // texCoords
        -1.0f, 1.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f,
        1.0f, -1.0f, 1.0f, 0.0f,

        -1.0f, 1.0f, 0.0f, 1.0f,
        1.0f, -1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f };
    unsigned int quadVAO, quadVBO;
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    FBO fboHigh = createFBO(SCR_WIDTH, SCR_HEIGHT);

    glm::vec3 pointLightPositions[] = {
        glm::vec3(-47.7f, 58.2f, -78.0f),
        glm::vec3(-58.3f, 58.3f, 41.0f),
        glm::vec3(135.0f, 60.0f, 37.0f),
        glm::vec3(135.0f, 60.0f, -81.0f) };


    int count = 0;
    const GazePoint* last = nullptr;
    using clock = std::chrono::high_resolution_clock;

    while (!glfwWindowShouldClose(window))
    {
        processInput(window);

        auto frame_start = clock::now();
        float fps = 1.0f / deltaTime;
        //std::cout << "FPS: " << fps << std::endl;

        // ========== API UPDATE ==========
        auto t0 = clock::now();
        api->Update();
        auto t1 = clock::now();

        // ========== GET GAZE POINTS ==========
        int count = streamsProvider->GetGazePoints(gazePoints);
        auto t2 = clock::now();
        if (gazePoints != nullptr) {
            for (int i = 0; i < count; ++i) {
                const GazePoint& point = gazePoints[i];
                last = &gazePoints[i];
                auto [gaze_deg_x, gaze_deg_y] = pixelsToDegreesFromNormalized(point.X, point.Y);
                gaze_history.push_back({ gaze_deg_x, gaze_deg_y });
            }
        }

        // ========== TIME PROCESSING AND INFERENCE ==========
        auto t3 = clock::now();
        if (gaze_history.size() > 10) gaze_history.pop_front();

        if (gaze_history.size() == 10) {
            auto& prev = gaze_history[gaze_history.size() - 2];
            auto& curr = gaze_history.back();

            float dx = predicted_deg.first - curr[0];
            float dy = predicted_deg.second - curr[1];
            float raw_error = std::sqrt(dx * dx + dy * dy);
            float total_error = std::sqrt(raw_error * raw_error + PRECISION * PRECISION);
            std::cout << total_error << std::endl;

            std::vector<float> input_tensor_values;
            for (const auto& pt : gaze_history) {
                input_tensor_values.push_back(pt[0]);
                input_tensor_values.push_back(pt[1]);
            }

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(),
                input_tensor_values.size(), input_shape.data(), input_shape.size());

            auto output_tensors = session.Run(Ort::RunOptions{ nullptr },
                input_names, &input_tensor, 1,
                output_names, 1);

            float* output = output_tensors.front().GetTensorMutableData<float>();
            predicted_deg = { output[0], output[1] };
            predicted = gazeAngleToNorm(predicted_deg.first, predicted_deg.second);
            createFoveationTexture(predicted, total_error);
        }
        auto t4 = clock::now();

        // ========== RENDER PASS ==========
        glm::mat4 view = camera.GetViewMatrix();
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 10000.0f);
        glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.f, 0.0f));
        model = glm::scale(model, glm::vec3(.1f));

        shader.use();
        glEnable(NVShadingRate::IMAGE);
        createTexture(fov_texture);
        uploadFoveationDataToTexture(fov_texture);
        glBindShadingRateImageNV(fov_texture);

        shader.setMat4("view", view);
        shader.setMat4("projection", projection);
        shader.setMat4("model", model);
        shader.setVec3("viewPos", camera.Position);
        // directional light
        shader.setVec3("dirLight.direction", -0.2f, -10.0f, -0.3f);
        shader.setVec3("dirLight.ambient", 0.4f, 0.4f, 0.4f);
        shader.setVec3("dirLight.diffuse", 0.5f, 0.5f, 0.5f);
        shader.setVec3("dirLight.specular", 0.7f, 0.7f, 0.7f);
        /*for (int i = 0; i < 4; i++)
        {
            shader.setVec3("pointLights[" + std::to_string(i) + "].diffuse", 1.0f, 1.0f, 1.0f);
            shader.setVec3("pointLights[" + std::to_string(i) + "].specular", 1.0f, 1.0f, 1.0f);
        }

        shader.setVec3("dirLight.ambient", 0.3f, 0.3f, 0.3f);
        for (int i = 0; i < 4; i++)
        {
            shader.setVec3("pointLights[" + std::to_string(i) + "].ambient", 0.3f, 0.3f, 0.3f);
        }

        for (int i = 0; i < 4; i++)
        {
            shader.setFloat("pointLights[" + std::to_string(i) + "].constant", 1.0f);
            shader.setFloat("pointLights[" + std::to_string(i) + "].linear", 0.007f);
            shader.setFloat("pointLights[" + std::to_string(i) + "].quadratic", 0.0017f);
        }

        shader.setVec3("pointLights[0].position", pointLightPositions[0]);
        shader.setVec3("pointLights[1].position", pointLightPositions[1]);
        // point light 3
        shader.setVec3("pointLights[2].position", pointLightPositions[2]);
        shader.setVec3("pointLights[3].position", pointLightPositions[3]);*/
        shader.setMat4("view", view);
        shader.setMat4("model", model);
        shader.setVec3("viewPos", camera.Position);

        glBindFramebuffer(GL_FRAMEBUFFER, fboHigh.fbo);
        glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
        shader.setBool("showShading", showShading);
        renderScene(shader, conference);

        glDisable(GL_DEPTH_TEST);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        screenShader.use();
        glBindVertexArray(quadVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, fboHigh.texture);
        screenShader.setBool("showShading", showShading);
        screenShader.setInt("screenTexture", 0);
        screenShader.setVec2("predicted", predicted);
        if (gaze_history.size() > 0)
            screenShader.setVec2("true_gaze", glm::vec2((last->X + 1.0) / 2.0, (last->Y + 1) / 2.0));
        glDrawArrays(GL_TRIANGLES, 0, 6);

        auto t5 = clock::now();

        glfwSwapBuffers(window);
        glfwPollEvents();

        float t_api = std::chrono::duration<float, std::milli>(t1 - t0).count();
        float t_gaze = std::chrono::duration<float, std::milli>(t2 - t1).count();
        float t_infer = std::chrono::duration<float, std::milli>(t4 - t3).count();
        float t_render = std::chrono::duration<float, std::milli>(t5 - t4).count();
        float t_total = std::chrono::duration<float, std::milli>(t5 - frame_start).count();

        std::cout << "[ms] API: " << t_api
            << " | Gaze: " << t_gaze
            << " | Infer: " << t_infer
            << " | Render: " << t_render
            << " | Total: " << t_total << std::endl;

        // dt
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        while ((err = glGetError()) != GL_NO_ERROR)
            std::cout << "After process: GL Error " << err << std::endl;
}

    glfwTerminate();
    return 0;
}

void renderScene(Shader& shader, Model model)
{
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    shader.use();
    glActiveTexture(GL_TEXTURE0);

    model.Draw(shader);
}

void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS)
        showShading = !showShading;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        isCursorEnabled = true;
    }
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
    {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
        isCursorEnabled = false;

        firstMouse = true;
    }
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    const float sensitivity = 0.3f;
    if (!isCursorEnabled)
    {
        float xoffset_fov = ((xpos / SCR_WIDTH) - posX) * sensitivity;
        float yoffset_fov = ((1 - ypos / SCR_HEIGHT) - posY) * sensitivity;
        posX += xoffset_fov;
        posY += yoffset_fov;
        posX = std::clamp(posX, 0.f, 1.f);
        posY = std::clamp(posY, 0.f, 1.f);
        return;
    }
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow* window, double xoffest, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

FBO createFBO(int width, int height)
{
    FBO fboData;

    glGenFramebuffers(1, &fboData.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fboData.fbo);

    glGenTextures(1, &fboData.texture);
    glBindTexture(GL_TEXTURE_2D, fboData.texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fboData.texture, 0);

    glGenRenderbuffers(1, &fboData.rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, fboData.rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, fboData.rbo);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "Framebuffer not complete!" << std::endl;

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return fboData;
}

float deg2rad(float deg)
{
    return deg * 3.1415 / 180.0f;
}

glm::vec2 gazeAngleToNorm(float x_deg, float y_deg)
{

    float x_rad = deg2rad(x_deg);
    float y_rad = deg2rad(y_deg);

    float x_mm = std::tan(x_rad) * DIST_MM;
    float y_mm = std::tan(y_rad) * DIST_MM;

    float px_x = (x_mm / SCR_WIDTH_MM) * SCR_WIDTH + SCR_WIDTH / 2.0f;
    float px_y = (y_mm / SCR_HEIGHT_MM) * SCR_HEIGHT + SCR_HEIGHT / 2.0f;

    float norm_x = px_x / SCR_WIDTH;
    float norm_y = px_y / SCR_HEIGHT;

    norm_x = std::clamp(norm_x, 0.0f, 1.0f);
    norm_y = std::clamp(norm_y, 0.0f, 1.0f);

    return glm::vec2(norm_x, norm_y);
}

std::pair<float, float> pixelsToDegreesFromNormalized(float norm_x, float norm_y)
{
    float px_x = norm_x * SCR_WIDTH / 2.0;
    float px_y = norm_y * SCR_HEIGHT/ 2.0;

    float px_per_mm_x = SCR_WIDTH / SCR_WIDTH_MM;
    float px_per_mm_y = SCR_HEIGHT / SCR_HEIGHT_MM;

    float dx_mm = (px_x ) / px_per_mm_x;
    float dy_mm = (px_y ) / px_per_mm_y;

    float deg_x = std::atan2(dx_mm,  DIST_MM) * (180.0f / static_cast<float>(3.1415));
    float deg_y = std::atan2(dy_mm,  DIST_MM) * (180.0f / static_cast<float>(3.1415));

    return { deg_x, deg_y };
}

float angleToNormRadius(float deg, float diagInInches, float distMM, int scrWidth, int scrHeight) {
    float diagPx = std::sqrt(scrWidth * scrWidth + scrHeight * scrHeight);
    float diagMM = diagInInches * 25.4f;
    float pixelSizeMM = diagMM / diagPx;

    float rad = glm::radians(deg);
    float sizeMM = 2.0f * distMM * std::tan(rad / 2.0f);
    float sizePx = sizeMM / pixelSizeMM;

    float screenSizePx = std::min(scrWidth, scrHeight);  
    return (sizePx / screenSizePx) / 2.0f;  // radius
}

void createTexture(GLuint& glid)
{
    if (glid)
    {
        glDeleteTextures(1, &glid);
    }
    glGenTextures(1, &glid);
}

void setupShadingRatePalette()
{
    GLint palSize;
    glGetIntegerv(NVShadingRate::PALETTE_SIZE, &palSize);
    assert(palSize >= 4);

    GLenum* palette = new GLenum[palSize];

    palette[0] = NVShadingRate::NO_INVOCATIONS;
    palette[1] = NVShadingRate::ONE_INVOCATION_PER_PIXEL;
    palette[2] = NVShadingRate::ONE_INVOCATION_PER_2X2;
    palette[3] = NVShadingRate::ONE_INVOCATION_PER_4X4;

    for (int i = 4; i < palSize; ++i)
    {
        palette[i] = NVShadingRate::ONE_INVOCATION_PER_PIXEL;
    }

    glShadingRateImagePaletteNV(0, 0, palSize, palette);
    delete[] palette;
}

void createFoveationTexture(glm::vec2 point, float error)
{

    float centerX = point[0];
    float centerY = point[1];
    const int width = m_shadingRateImageWidth;
    const int height = m_shadingRateImageHeight;

    float scale = 0.1f; 
    float dynamicError = error * scale;
    float innerR = INNER_R + dynamicError;
    float middleR = MIDDLE_R + dynamicError;
    std::cout << dynamicError << "  dd" << std::endl;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            float fx = x / (float)width;
            float fy = y / (float)height;

            float d = std::sqrt((fx - centerX) * (fx - centerX) + (fy - centerY) * (fy - centerY));
            if (d < (innerR))
            {
                m_shadingRateImageData[x + y * width] = 1;
            }
            else if (d < (MIDDLE_R))
            {
                m_shadingRateImageData[x + y * width] = 2;
            }
            else
            {
                m_shadingRateImageData[x + y * width] = 3;
            }
        }
    }
}

void uploadFoveationDataToTexture(GLuint texture)
{
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R8UI, m_shadingRateImageWidth, m_shadingRateImageHeight);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_shadingRateImageWidth, m_shadingRateImageHeight, GL_RED_INTEGER, GL_UNSIGNED_BYTE, &m_shadingRateImageData[0]);
}

bool InitNVShadingRateImageExtensions() {
    bool allLoaded = true;

    allLoaded &= LoadGLFunction(glBindShadingRateImageNV, "glBindShadingRateImageNV");
    allLoaded &= LoadGLFunction(glShadingRateImagePaletteNV, "glShadingRateImagePaletteNV");


    if (!glfwExtensionSupported("GL_NV_shading_rate_image")) {
        std::cout << "GL_NV_shading_rate_image not supported!" << std::endl;
        allLoaded = false;
    }
    if (!glfwExtensionSupported("GL_NV_primitive_shading_rate")) {
        std::cout << "GL_NV_primitive_shading_rate not supported!" << std::endl;
        allLoaded = false;
    }

    return allLoaded;
}

template <typename T>
bool LoadGLFunction(T& funcPtr, const char* name) {
    funcPtr = reinterpret_cast<T>(glfwGetProcAddress(name));
    if (!funcPtr) {
        std::cerr << "Failed to load " << name << "!" << std::endl;
        return false;
    }
    else {
        std::cout << "Successfully loaded " << name << "!" << std::endl;
        return true;
    }
}