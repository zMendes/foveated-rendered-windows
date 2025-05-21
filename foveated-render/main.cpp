#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <GLFW/glfw3.h>

#include <deque>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <cmath>
#include <utility>
//#include <torch/script.h>
// 
// 

#include <onnxruntime_cxx_api.h>

//Eye tracking data
#include <tobii_gameintegration.h>


// #include "imgui.h"
// #include "imgui_impl_opengl3.h"
// #include "imgui_impl_glfw.h"

#include "shader.h"
#include "camera.h"
#include "stb_image.h"
#include "model.h"

#include <iostream>
#include <algorithm>
#include <vector>

typedef struct
{
    unsigned int fbo;
    unsigned int texture;
    unsigned int rbo;
} FBO;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
FBO createFBO(int width, int height);
void renderScene(Shader& shader, Model model);
glm::vec2 gazeAngleToNorm(float predicted_x_deg, float predicted_y_deg);
std::pair<float, float> pixelsToDegrees(float px_x, float px_y);
std::pair<float, float> pixelsToDegreesFromNormalized(float norm_x, float norm_y);


// settings
const unsigned int SCR_WIDTH = 2560;
const unsigned int SCR_HEIGHT = 1440;
float SCR_WIDTH_MM = 382.0f;
float SCR_HEIGHT_MM = 215.0f;
float DIST_MM = 800.0f;

float posX = 0.5;
float posY = 0.5;

// CAMERA
Camera camera(glm::vec3(0.0f, 2.0f, 8.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, 0.0f);
bool firstMouse = true;
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

bool showShading = false;
bool isCursorEnabled = false;

using namespace TobiiGameIntegration;


int main()
{

    // Eye tracking data
    ITobiiGameIntegrationApi* api = GetApi("Gaze Sample");
    IStreamsProvider* streamsProvider = api->GetStreamsProvider();
    std::cout << "Stream Provider " << streamsProvider;
    std::cout << "API " << api->GetStatistics();

    api->GetTrackerController()->TrackRectangle({ 0,0,2560,1440 });
    GazePoint gazePoint;

    std::deque<std::array<float, 2>> gaze_history;

    glm::vec2 predicted;


    std::ifstream f("C:/Users/leonardomm8/Documents/saccade_predictor_traced.pt");
    if (!f) {
        std::cerr << "File not found!" << std::endl;
        return -1;
    }
    else {
        std::cout << "FIle found" << std::endl;
    }
    f.close();

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "saccade_predictor");

    // Create session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // Load the model
    const wchar_t* model_path = L"C:/Users/leonardomm8/Documents/saccade_predictor.onnx";
    Ort::Session session(env, model_path, session_options);

    std::cout << "Model loaded successfully!" << std::endl;



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
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // IMGUI
    // ImGui::CreateContext();
    // ImGui_ImplGlfw_InitForOpenGL(window, true);
    // ImGui_ImplOpenGL3_Init("#version 460 core");

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1; // Correct return here, exiting if GLAD fails
    }

    // OPENGL STATE
    glEnable(GL_DEPTH_TEST);

    Shader shader("vrs.vs", "vrs.fs");
    Shader screenShader("screen.vs", "screen.fs");
    shader.use();

    std::string path = "C:\\Users\\leonardomm8\\Documents\\sponza\\sponza.obj";

    Model sponza(path);

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
    FBO fboMedium = createFBO(SCR_WIDTH / 2, SCR_HEIGHT / 2);
    FBO fboLow = createFBO(SCR_WIDTH / 4, SCR_HEIGHT / 4);

    glm::vec3 pointLightPositions[] = {
        glm::vec3(0.7f, 2.2f, 2.0f),
        glm::vec3(22.3f, 3.3f, -4.0f),
        glm::vec3(-14.0f, 6.0f, -12.0f),
        glm::vec3(10.0f, 50.0f, -3.0f) };

    while (!glfwWindowShouldClose(window))
    {

        api->Update();

        streamsProvider->GetLatestGazePoint(gazePoint);
        
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window);

        auto [gaze_deg_x, gaze_deg_y] = pixelsToDegreesFromNormalized(gazePoint.X, gazePoint.Y);
        std::cout << "Gaze degg  " << gaze_deg_x  << " - " << gaze_deg_y;
        gaze_history.push_back({ gaze_deg_x, gaze_deg_y });
        if (gaze_history.size() > 10)
            gaze_history.pop_front();

        // Infer only if we have 10 points
        if (gaze_history.size() == 10)
        {
            // Prepare input tensor data
            std::vector<float> input_tensor_values;
            for (const auto& pt : gaze_history) {
                input_tensor_values.push_back(pt[0]);  // x
                input_tensor_values.push_back(pt[1]);  // y
            }

            // Define input shape: [1, 10, 2]
            std::array<int64_t, 3> input_shape = { 1, 10, 2 };

            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, input_tensor_values.data(), input_tensor_values.size(),
                input_shape.data(), input_shape.size());

            // Prepare input & output names
            const char* input_names[] = { "input" };   // Use your actual input name
            const char* output_names[] = { "output" }; // Use your actual output name

            auto output_tensors = session.Run(Ort::RunOptions{ nullptr },
                input_names, &input_tensor, 1,
                output_names, 1);

            // Read output: should be shape [1, 2]
            float* output = output_tensors.front().GetTensorMutableData<float>();
            float predicted_x = output[0];
            float predicted_y = output[1];

            predicted = gazeAngleToNorm(predicted_x, predicted_y);
            

            std::cout << "Predicted landing: (" << predicted[0] << ", " << predicted[1] << ")\n";

        }


        // ImGui_ImplOpenGL3_NewFrame();
        // ImGui_ImplGlfw_NewFrame();
        // ImGui::NewFrame();

        glm::mat4 view = camera.GetViewMatrix();
        // projection matrix
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 10000.0f);

        // SPONZA
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(0.0f, 0.f, 0.0f));
        model = glm::scale(model, glm::vec3(.4f, .4f, .4f));

        shader.use();

        shader.setMat4("view", view);
        shader.setVec3("viewPos", camera.Position);
        shader.setMat4("projection", projection);
        // directional light
        shader.setVec3("dirLight.direction", -0.2f, 10.0f, -0.3f);
        shader.setVec3("dirLight.ambient", 0.2f, 0.2f, 0.2f);
        shader.setVec3("dirLight.diffuse", 0.4f, 0.4f, 0.4f);
        shader.setVec3("dirLight.specular", 0.5f, 0.5f, 0.5f);

        // point light 1
        shader.setVec3("pointLights[0].position", pointLightPositions[0]);
        shader.setVec3("pointLights[0].ambient", 0.2f, 0.2f, 0.2f);
        shader.setVec3("pointLights[0].diffuse", 0.8f, 0.8f, 0.8f);
        shader.setVec3("pointLights[0].specular", 1.0f, 1.0f, 1.0f);
        shader.setFloat("pointLights[0].constant", 1.0f);
        shader.setFloat("pointLights[0].linear", 0.09f);
        shader.setFloat("pointLights[0].quadratic", 0.032f);
        // point light 2
        shader.setVec3("pointLights[1].position", pointLightPositions[1]);
        shader.setVec3("pointLights[1].ambient", 0.2f, 0.2f, 0.2f);
        shader.setVec3("pointLights[1].diffuse", 0.8f, 0.8f, 0.8f);
        shader.setVec3("pointLights[1].specular", 1.0f, 1.0f, 1.0f);
        shader.setFloat("pointLights[1].constant", 1.0f);
        shader.setFloat("pointLights[1].linear", 0.09f);
        shader.setFloat("pointLights[1].quadratic", 0.032f);
        // point light 3
        shader.setVec3("pointLights[2].position", pointLightPositions[2]);
        shader.setVec3("pointLights[2].ambient", 0.2f, 0.2f, 0.2f);
        shader.setVec3("pointLights[2].diffuse", 0.8f, 0.8f, 0.8f);
        shader.setVec3("pointLights[2].specular", 1.0f, 1.0f, 1.0f);
        shader.setFloat("pointLights[2].constant", 1.0f);
        shader.setFloat("pointLights[2].linear", 0.09f);
        shader.setFloat("pointLights[2].quadratic", 0.032f);
        // point light 4
        shader.setVec3("pointLights[3].position", pointLightPositions[3]);
        shader.setVec3("pointLights[3].ambient", 0.2f, 0.2f, 0.2f);
        shader.setVec3("pointLights[3].diffuse", 0.8f, 0.8f, 0.8f);
        shader.setVec3("pointLights[3].specular", 1.0f, 1.0f, 1.0f);
        shader.setFloat("pointLights[3].constant", 1.0f);
        shader.setFloat("pointLights[3].linear", 0.09f);
        shader.setFloat("pointLights[3].quadratic", 0.032f);
        shader.setMat4("view", view);
        shader.setMat4("projection", projection);
        shader.setMat4("model", model);
        shader.setVec3("viewPos", camera.Position);

        glBindFramebuffer(GL_FRAMEBUFFER, fboHigh.fbo);
        glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
        renderScene(shader, sponza);

        glBindFramebuffer(GL_FRAMEBUFFER, fboMedium.fbo);
        glViewport(0, 0, SCR_WIDTH / 2, SCR_HEIGHT / 2);
        renderScene(shader, sponza);

        glBindFramebuffer(GL_FRAMEBUFFER, fboLow.fbo);
        glViewport(0, 0, SCR_WIDTH / 4, SCR_HEIGHT / 4);
        renderScene(shader, sponza);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);

        glDisable(GL_DEPTH_TEST);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        screenShader.use();
        glBindVertexArray(quadVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, fboHigh.texture);
        screenShader.setInt("texHigh", 0);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, fboMedium.texture);
        screenShader.setInt("texMedium", 1);

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, fboLow.texture);
        screenShader.setInt("texLow", 2);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        screenShader.setVec2("gaze", glm::vec2(gazePoint.X, gazePoint.Y));
        screenShader.setBool("showShading", showShading);
        screenShader.setVec2("predicted", predicted);

        while ((err = glGetError()) != GL_NO_ERROR)
        {
            std::cout << "After process" << err << std::endl;
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
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
        //posX = std::clamp(posX, 0.f, 1.f);
        //posY = std::clamp(posY, 0.f, 1.f);
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

float deg2rad(float deg) {
    return deg * 3.1415 / 180.0f;
}

glm::vec2 gazeAngleToNorm(float predicted_x_deg, float predicted_y_deg) {
    std::cout << "Ang" << predicted_x_deg << " - " << predicted_y_deg << std::endl;

    float screen_width_mm = 382.0f;
    float screen_height_mm = 215.0f;
    float distance_mm = 800.0f;

    int screen_width_px = 2560;
    int screen_height_px = 1440;

    float x_rad = deg2rad(predicted_x_deg);
    float y_rad = deg2rad(predicted_y_deg);

    float x_mm = std::tan(x_rad) * DIST_MM;
    float y_mm = std::tan(y_rad) * DIST_MM;

    float px_x = (x_mm / SCR_WIDTH_MM) * SCR_WIDTH + SCR_WIDTH / 2.0f;
    float px_y = (y_mm / SCR_HEIGHT_MM) * SCR_HEIGHT + SCR_HEIGHT/ 2.0f;
    std::cout << "PX" << px_x << " - " << px_y << std::endl;

    float norm_x = px_x / SCR_WIDTH;
    float norm_y = px_y / SCR_HEIGHT;

    return glm::vec2(norm_x, norm_y);
}



std::pair<float, float> pixelsToDegrees(float px_x, float px_y) {

    float x_mm = ((px_x - SCR_WIDTH / 2.0f) / SCR_WIDTH) * SCR_WIDTH_MM;
    float y_mm = ((px_y - SCR_HEIGHT / 2.0f) / SCR_HEIGHT) * SCR_HEIGHT_MM;

    float x_deg = atan2(x_mm, DIST_MM) * 180.0f / 3.1415;
    float y_deg = atan2(y_mm, DIST_MM) * 180.0f / 3.1415;

    return { x_deg, y_deg };
}

std::pair<float, float> pixelsToDegreesFromNormalized(float norm_x, float norm_y) {
    float px_x = norm_x * SCR_WIDTH;
    float px_y = norm_y * SCR_HEIGHT;

    float cx = SCR_WIDTH / 2.0f;
    float cy = SCR_HEIGHT / 2.0f;

    float px_per_mm_x = SCR_WIDTH / SCR_WIDTH_MM;
    float px_per_mm_y = SCR_HEIGHT / SCR_HEIGHT_MM;

    float dx_mm = (px_x - cx) / px_per_mm_x;
    float dy_mm = (px_y - cy) / px_per_mm_y;

    float deg_x = std::atan(dx_mm / DIST_MM) * (180.0f / static_cast<float>(3.1415));
    float deg_y = std::atan(dy_mm / DIST_MM) * (180.0f / static_cast<float>(3.1415));

    return { deg_x, deg_y };
}