cmake_minimum_required(VERSION 3.20)

if(WIN32 AND NOT DEFINED CMAKE_SYSTEM_VERSION AND DEFINED ENV{LIPLAB_WINDOWS_SDK_VERSION})
    set(CMAKE_SYSTEM_VERSION "$ENV{LIPLAB_WINDOWS_SDK_VERSION}")
endif()

project(liplab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(liplab_core
    offgrid_dropin/Private/Lipsync/OffgridAICmudictData.cpp
    offgrid_dropin/Private/Lipsync/OffgridAIStreamingAudioFeatures.cpp
    offgrid_dropin/Private/Lipsync/OffgridAITextVisemePlanner.cpp
    offgrid_dropin/Private/Lipsync/OffgridAIVisemePerformer.cpp
    offgrid_dropin/Private/Lipsync/OffgridAILipsyncRuntimeAdapter.cpp
    offgrid_dropin/Private/Lipsync/OffgridAINeuralLipsyncBridge.cpp
)

target_include_directories(liplab_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/standalone_ue_shim
        ${CMAKE_CURRENT_SOURCE_DIR}/offgrid_dropin/Public
)

add_executable(liplab_runner harness/main.cpp)
target_link_libraries(liplab_runner PRIVATE liplab_core)

option(LIPLAB_ENABLE_TORCH "Build the optional LibTorch CUDA neural streamer trainer" OFF)
option(LIPLAB_ENABLE_NEURAL_RUNTIME "Build the standalone CUDA neural lipsync runtime" OFF)
if(LIPLAB_ENABLE_TORCH)
    set(LIPLAB_ENABLE_NEURAL_RUNTIME ON)
endif()

if(LIPLAB_ENABLE_NEURAL_RUNTIME)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Ship native machine code from Turing through Ada rather than relying on
    # a driver's PTX JIT. CUDA 12.8 added Blackwell sm_120 support, so newer
    # toolkits produce one RTX 20/30/40/50-series fat binary.
    if(NOT DEFINED LIPLAB_CUDA_ARCHITECTURES)
        set(LIPLAB_CUDA_ARCHITECTURES "75;86;89")
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
            list(APPEND LIPLAB_CUDA_ARCHITECTURES "120")
        endif()
    endif()
    add_library(liplab_neural_streamer_cuda STATIC
        offgrid_dropin/Private/Lipsync/OffgridAINeuralStreamerRuntime.cu
    )
    target_include_directories(liplab_neural_streamer_cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/offgrid_dropin/Public)
    target_link_libraries(liplab_neural_streamer_cuda PUBLIC CUDA::cudart_static)
    set_property(TARGET liplab_neural_streamer_cuda PROPERTY CUDA_STANDARD 17)
    set_property(TARGET liplab_neural_streamer_cuda PROPERTY CUDA_STANDARD_REQUIRED ON)
    set_property(TARGET liplab_neural_streamer_cuda PROPERTY CUDA_RUNTIME_LIBRARY Static)
    set_property(TARGET liplab_neural_streamer_cuda PROPERTY
        CUDA_ARCHITECTURES "${LIPLAB_CUDA_ARCHITECTURES}")
    message(STATUS "LipLab CUDA runtime architectures: ${LIPLAB_CUDA_ARCHITECTURES}")

    target_sources(liplab_core PRIVATE
        offgrid_dropin/Private/Lipsync/OffgridAINeuralStreamingAligner.cpp)
    target_include_directories(liplab_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/offgrid_dropin/Private)
    target_compile_definitions(liplab_core PUBLIC
        OFFGRIDAI_WITH_NEURAL_LIPSYNC=1)
    target_link_libraries(liplab_core PUBLIC liplab_neural_streamer_cuda)

    set(LIPLAB_NEURAL_MODEL
        ${CMAKE_CURRENT_SOURCE_DIR}/offgrid_dropin/Private/Lipsync/Models/OffgridAINeuralStreamerV5.bin)
    set(LIPLAB_EMBEDDED_MODEL_HEADER
        ${CMAKE_CURRENT_BINARY_DIR}/generated/OffgridAINeuralStreamerEmbeddedModel.h)
    add_custom_command(
        OUTPUT ${LIPLAB_EMBEDDED_MODEL_HEADER}
        COMMAND ${CMAKE_COMMAND}
            -DINPUT=${LIPLAB_NEURAL_MODEL}
            -DOUTPUT=${LIPLAB_EMBEDDED_MODEL_HEADER}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedBinary.cmake
        DEPENDS ${LIPLAB_NEURAL_MODEL} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedBinary.cmake
        VERBATIM)
    add_executable(liplab_neural_runtime_smoke
        harness/ml/neural_runtime_smoke.cpp
        ${LIPLAB_EMBEDDED_MODEL_HEADER})
    target_include_directories(liplab_neural_runtime_smoke PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/generated)
    target_link_libraries(liplab_neural_runtime_smoke PRIVATE
        liplab_neural_streamer_cuda liplab_core)
endif()

if(LIPLAB_ENABLE_TORCH)
    find_package(Torch CONFIG REQUIRED)
    add_executable(liplab_monotonic_aligner_torch harness/ml/monotonic_aligner_torch.cpp)
    target_include_directories(liplab_monotonic_aligner_torch PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/offgrid_dropin/Public)
    target_link_libraries(liplab_monotonic_aligner_torch PRIVATE
        ${TORCH_LIBRARIES}
        liplab_neural_streamer_cuda
    )
    # LibTorch 2.13 headers use C++20 designated initializers and bit-field
    # defaults. Keep the standalone harness on C++17 and raise only this
    # optional target.
    set_property(TARGET liplab_monotonic_aligner_torch PROPERTY CXX_STANDARD 20)
    set_property(TARGET liplab_monotonic_aligner_torch PROPERTY CXX_STANDARD_REQUIRED ON)
    if(MSVC)
        target_compile_options(liplab_monotonic_aligner_torch PRIVATE /bigobj)
    endif()
endif()
