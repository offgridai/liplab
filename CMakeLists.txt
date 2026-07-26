cmake_minimum_required(VERSION 3.20)

if(WIN32 AND NOT DEFINED CMAKE_SYSTEM_VERSION AND DEFINED ENV{LIPLAB_WINDOWS_SDK_VERSION})
    set(CMAKE_SYSTEM_VERSION "$ENV{LIPLAB_WINDOWS_SDK_VERSION}")
endif()

project(liplab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(liplab_core
    offgrid_dropin/Private/Lipsync/OffgridAICmudictData.cpp
    offgrid_dropin/Private/Lipsync/OffgridAIStreamingSpeechDetector.cpp
    offgrid_dropin/Private/Lipsync/OffgridAITextVisemePlanner.cpp
    offgrid_dropin/Private/Lipsync/OffgridAIVisemePerformer.cpp
    offgrid_dropin/Private/Lipsync/OffgridAILipsyncRuntimeAdapter.cpp
    offgrid_dropin/Private/Lipsync/OffgridAIAcousticEvidence.cpp
    offgrid_dropin/Private/Lipsync/OffgridAIStreamingEvidenceSurface.cpp
    offgrid_dropin/Private/Lipsync/OffgridAIStreamingSyllablePositionEstimator.cpp
)

target_include_directories(liplab_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/standalone_ue_shim
        ${CMAKE_CURRENT_SOURCE_DIR}/offgrid_dropin/Public
)

add_executable(liplab_runner harness/main.cpp)
target_link_libraries(liplab_runner PRIVATE liplab_core)

add_executable(liplab_timing_baseline harness/ml/timing_baseline.cpp)
target_include_directories(liplab_timing_baseline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/harness/ml)

option(LIPLAB_ENABLE_TORCH "Build the optional LibTorch CUDA timing trainer" OFF)
if(LIPLAB_ENABLE_TORCH)
    find_package(Torch CONFIG REQUIRED)
    add_executable(liplab_timing_torch harness/ml/timing_torch.cpp)
    add_executable(liplab_neural_scheduler_torch harness/ml/neural_scheduler_torch.cpp)
    add_executable(liplab_monotonic_aligner_torch harness/ml/monotonic_aligner_torch.cpp)
    target_include_directories(liplab_timing_torch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/harness/ml)
    target_include_directories(liplab_neural_scheduler_torch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/harness/ml)
    target_include_directories(liplab_monotonic_aligner_torch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/harness/ml)
    target_link_libraries(liplab_timing_torch PRIVATE ${TORCH_LIBRARIES})
    target_link_libraries(liplab_neural_scheduler_torch PRIVATE ${TORCH_LIBRARIES})
    target_link_libraries(liplab_monotonic_aligner_torch PRIVATE ${TORCH_LIBRARIES})
    # LibTorch 2.13 headers use C++20 designated initializers and bit-field
    # defaults. Keep the standalone harness on C++17 and raise only this
    # optional target.
    set_property(TARGET liplab_timing_torch PROPERTY CXX_STANDARD 20)
    set_property(TARGET liplab_timing_torch PROPERTY CXX_STANDARD_REQUIRED ON)
    set_property(TARGET liplab_neural_scheduler_torch PROPERTY CXX_STANDARD 20)
    set_property(TARGET liplab_neural_scheduler_torch PROPERTY CXX_STANDARD_REQUIRED ON)
    set_property(TARGET liplab_monotonic_aligner_torch PROPERTY CXX_STANDARD 20)
    set_property(TARGET liplab_monotonic_aligner_torch PROPERTY CXX_STANDARD_REQUIRED ON)
    if(MSVC)
        target_compile_options(liplab_timing_torch PRIVATE /bigobj)
        target_compile_options(liplab_neural_scheduler_torch PRIVATE /bigobj)
        target_compile_options(liplab_monotonic_aligner_torch PRIVATE /bigobj)
    endif()
endif()
