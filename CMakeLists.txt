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
)

target_include_directories(liplab_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/standalone_ue_shim
        ${CMAKE_CURRENT_SOURCE_DIR}/offgrid_dropin/Public
)

add_executable(liplab_runner harness/main.cpp)
target_link_libraries(liplab_runner PRIVATE liplab_core)
