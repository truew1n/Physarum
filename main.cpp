#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

struct FPoint2D {
    float x;
    float y;
};

struct Agent {
    FPoint2D Position;
    float Rotation;
};

#define WIDTH 1920
#define HEIGHT 1080
#define AGENT_COUNT 10000000
#define ERROR_INIT_FAILED -1

// Compute shader sources
const char* initAgentsSource = R"(
#version 430

layout (local_size_x = 1024) in;

struct FPoint2D {
    float x;
    float y;
};

struct Agent {
    FPoint2D Position;
    float Rotation;
};

layout (std430, binding = 0) buffer AgentsBuffer {
    Agent agents[];
};

uniform uint seed;
uniform uvec2 dimensions;

float random(inout uint state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return float(state) / 4294967295.0;
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= agents.length()) return;
    
    uint state = seed + idx;
    float RandomRadius = random(state) * 300;
    float RandomAngle = (random(state) - 0.5) * 2.0 * 3.14159265359;
    
    agents[idx].Position.x = dimensions.x / 2.0 + RandomRadius * cos(RandomAngle);
    agents[idx].Position.y = dimensions.y / 2.0 + RandomRadius * -sin(RandomAngle);
    agents[idx].Rotation = RandomAngle + 3.14159265359;
}
)";

const char* updateAgentsSource = R"(
#version 430

layout (local_size_x = 1024) in;

struct FPoint2D {
    float x;
    float y;
};

struct Agent {
    FPoint2D Position;
    float Rotation;
};

layout (std430, binding = 0) buffer AgentsBuffer {
    Agent agents[];
};

layout (std430, binding = 1) buffer TrailMapBuffer {
    float trailMap[];
};

uniform float deltaTime;
uniform float agentVelocity;
uniform float agentTurnSpeed;
uniform float agentSensorLength;
uniform float agentSensorAngle;
uniform uvec2 dimensions;

float sense(FPoint2D position, float rotation, float angle) {
    float sensorAngle = rotation + angle;
    FPoint2D sensorPosition = FPoint2D(
        position.x + agentSensorLength * cos(sensorAngle),
        position.y + agentSensorLength * -sin(sensorAngle)
    );
    
    if (sensorPosition.x < 0 || sensorPosition.x >= dimensions.x ||
        sensorPosition.y < 0 || sensorPosition.y >= dimensions.y) {
        return 0.0;
    }
    
    uint sensorIndex = uint(sensorPosition.y) * dimensions.x + uint(sensorPosition.x);
    return trailMap[sensorIndex];
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= agents.length()) return;
    
    FPoint2D position = agents[idx].Position;
    float rotation = agents[idx].Rotation;
    
    float forwardSensor = sense(position, rotation, 0);
    float leftSensor = sense(position, rotation, agentSensorAngle);
    float rightSensor = sense(position, rotation, -agentSensorAngle);
    
    if (forwardSensor > leftSensor && forwardSensor > rightSensor) {
        // keep going forward
    } else if (leftSensor > rightSensor) {
        rotation += agentTurnSpeed;
    } else if (rightSensor > leftSensor) {
        rotation -= agentTurnSpeed;
    } else {
        rotation += (fract(sin(gl_GlobalInvocationID.x) * 43758.5453) - 0.5) * 2.0 * agentTurnSpeed;
    }
    
    position.x += agentVelocity * cos(rotation);
    position.y += agentVelocity * -sin(rotation);
    
    // Handle boundary conditions
    if (position.x < 0) position.x = 0;
    if (position.x >= dimensions.x) position.x = dimensions.x - 1;
    if (position.y < 0) position.y = 0;
    if (position.y >= dimensions.y) position.y = dimensions.y - 1;
    
    agents[idx].Position = position;
    agents[idx].Rotation = rotation;
}
)";

const char* renderAgentsSource = R"(
#version 430

layout (local_size_x = 1024) in;

struct FPoint2D {
    float x;
    float y;
};

struct Agent {
    FPoint2D Position;
    float Rotation;
};

layout (std430, binding = 0) buffer AgentsBuffer {
    Agent agents[];
};

layout (std430, binding = 1) buffer TrailMapBuffer {
    float trailMap[];
};

uniform uvec2 dimensions;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= agents.length()) return;
    
    FPoint2D position = agents[idx].Position;
    
    if (position.x >= 0 && position.x < dimensions.x &&
        position.y >= 0 && position.y < dimensions.y) {
        uint trailIndex = uint(position.y) * dimensions.x + uint(position.x);
        trailMap[trailIndex] = 1.0;
    }
}
)";

const char* processTrailMapSource = R"(
#version 430

layout (local_size_x = 1024) in;

layout (std430, binding = 1) buffer TrailMapBuffer {
    float trailMap[];
};

layout (std430, binding = 2) buffer TrailMapCopyBuffer {
    float trailMapCopy[];
};

uniform float decayRate;
uniform float diffusionRate;
uniform uvec2 dimensions;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= dimensions.x * dimensions.y) return;

    uint x = idx % dimensions.x;
    uint y = idx / dimensions.x;
    
    float sum = 0.0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int nx = int(x) + i;
            int ny = int(y) + j;
            if (nx >= 0 && nx < dimensions.x && ny >= 0 && ny < dimensions.y) {
                uint neighborIdx = ny * dimensions.x + nx;
                sum += trailMapCopy[neighborIdx];
            }
        }
    }
    float blur = sum / 9.0;
    float diffused = mix(trailMap[idx], blur, diffusionRate);
    trailMap[idx] = diffused * decayRate;
}
)";

const char* renderTrailMapSource = R"(
#version 430

layout (local_size_x = 1024) in;

layout (std430, binding = 1) buffer TrailMapBuffer {
    float trailMap[];
};

layout (std430, binding = 3) buffer DisplayBuffer {
    uint display[];
};

uniform uvec2 dimensions;

uvec3 encodeColor(float value) {
    uint intensity = uint(value * 255);
    return uvec3(intensity, intensity, intensity);
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= dimensions.x * dimensions.y) return;

    float trailValue = trailMap[idx];
    uvec3 color = encodeColor(trailValue);
    display[idx] = (color.r << 16) | (color.g << 8) | color.b;
}
)";

GLuint compileShader(const char* source, GLenum type) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
        char buffer[512];
        glGetShaderInfoLog(shader, 512, nullptr, buffer);
        std::cerr << "Shader compile error: " << buffer << std::endl;
        exit(-1);
    }
    return shader;
}

GLuint createComputeProgram(const char* source) {
    GLuint program = glCreateProgram();
    GLuint shader = compileShader(source, GL_COMPUTE_SHADER);
    glAttachShader(program, shader);
    glLinkProgram(program);
    glDeleteShader(shader);

    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status != GL_TRUE) {
        char buffer[512];
        glGetProgramInfoLog(program, 512, nullptr, buffer);
        std::cerr << "Program link error: " << buffer << std::endl;
        exit(-1);
    }
    return program;
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return ERROR_INIT_FAILED;
    }
    
    // Create a windowed mode window and its OpenGL context
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Slime Mold Simulation", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return ERROR_INIT_FAILED;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return ERROR_INIT_FAILED;
    }

    // Create and bind SSBOs for agents and trail map
    GLuint agentsBuffer, trailMapBuffer, trailMapCopyBuffer, displayBuffer;
    glGenBuffers(1, &agentsBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, agentsBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, AGENT_COUNT * sizeof(Agent), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, agentsBuffer);

    glGenBuffers(1, &trailMapBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, trailMapBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, WIDTH * HEIGHT * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, trailMapBuffer);

    glGenBuffers(1, &trailMapCopyBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, trailMapCopyBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, WIDTH * HEIGHT * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, trailMapCopyBuffer);

    glGenBuffers(1, &displayBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, displayBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, WIDTH * HEIGHT * sizeof(uint32_t), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, displayBuffer);

    // Create compute programs
    GLuint initAgentsProgram = createComputeProgram(initAgentsSource);
    GLuint updateAgentsProgram = createComputeProgram(updateAgentsSource);
    GLuint renderAgentsProgram = createComputeProgram(renderAgentsSource);
    GLuint processTrailMapProgram = createComputeProgram(processTrailMapSource);
    GLuint renderTrailMapProgram = createComputeProgram(renderTrailMapSource);

    // Initialize agents
    glUseProgram(initAgentsProgram);
    glUniform1ui(glGetUniformLocation(initAgentsProgram, "seed"), static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    glUniform2uiv(glGetUniformLocation(initAgentsProgram, "dimensions"), 1, glm::value_ptr(glm::uvec2(WIDTH, HEIGHT)));
    glDispatchCompute((AGENT_COUNT + 1023) / 1024, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Main loop
    auto lastTime = std::chrono::high_resolution_clock::now();
    while (!glfwWindowShouldClose(window)) {
        // Compute delta time
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsedTime = currentTime - lastTime;
        lastTime = currentTime;
        float deltaTime = elapsedTime.count();

        // Update agents
        glUseProgram(updateAgentsProgram);
        glUniform1f(glGetUniformLocation(updateAgentsProgram, "deltaTime"), deltaTime);
        glUniform1f(glGetUniformLocation(updateAgentsProgram, "agentVelocity"), 1.0f);
        glUniform1f(glGetUniformLocation(updateAgentsProgram, "agentTurnSpeed"), 0.2f);
        glUniform1f(glGetUniformLocation(updateAgentsProgram, "agentSensorLength"), 10.0f);
        glUniform1f(glGetUniformLocation(updateAgentsProgram, "agentSensorAngle"), 0.0174532925f * 20.0f);
        glUniform2uiv(glGetUniformLocation(updateAgentsProgram, "dimensions"), 1, glm::value_ptr(glm::uvec2(WIDTH, HEIGHT)));
        glDispatchCompute((AGENT_COUNT + 1023) / 1024, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // Render agents to trail map
        glUseProgram(renderAgentsProgram);
        glUniform2uiv(glGetUniformLocation(renderAgentsProgram, "dimensions"), 1, glm::value_ptr(glm::uvec2(WIDTH, HEIGHT)));
        glDispatchCompute((AGENT_COUNT + 1023) / 1024, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // Process trail map
        glUseProgram(processTrailMapProgram);
        glUniform1f(glGetUniformLocation(processTrailMapProgram, "decayRate"), 0.999f);
        glUniform1f(glGetUniformLocation(processTrailMapProgram, "diffusionRate"), 0.13f);
        glUniform2uiv(glGetUniformLocation(processTrailMapProgram, "dimensions"), 1, glm::value_ptr(glm::uvec2(WIDTH, HEIGHT)));
        glDispatchCompute((WIDTH * HEIGHT + 1023) / 1024, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // Render trail map to display buffer
        glUseProgram(renderTrailMapProgram);
        glUniform2uiv(glGetUniformLocation(renderTrailMapProgram, "dimensions"), 1, glm::value_ptr(glm::uvec2(WIDTH, HEIGHT)));
        glDispatchCompute((WIDTH * HEIGHT + 1023) / 1024, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // Render display buffer to screen
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, displayBuffer);
        glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        // Swap buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();
    }

    // Clean up
    glDeleteBuffers(1, &agentsBuffer);
    glDeleteBuffers(1, &trailMapBuffer);
    glDeleteBuffers(1, &trailMapCopyBuffer);
    glDeleteBuffers(1, &displayBuffer);
    glDeleteProgram(initAgentsProgram);
    glDeleteProgram(updateAgentsProgram);
    glDeleteProgram(renderAgentsProgram);
    glDeleteProgram(processTrailMapProgram);
    glDeleteProgram(renderTrailMapProgram);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}