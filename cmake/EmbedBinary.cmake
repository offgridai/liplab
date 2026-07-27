if(NOT DEFINED INPUT OR NOT EXISTS "${INPUT}")
    message(FATAL_ERROR "Embedded model input does not exist: ${INPUT}")
endif()
if(NOT DEFINED OUTPUT)
    message(FATAL_ERROR "Embedded model output was not specified")
endif()

file(READ "${INPUT}" MODEL_HEX HEX)
file(SHA256 "${INPUT}" MODEL_SHA256)
string(REGEX REPLACE "([0-9A-Fa-f][0-9A-Fa-f])" "0x\\1," MODEL_BYTES "${MODEL_HEX}")
get_filename_component(OUTPUT_DIRECTORY "${OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${OUTPUT_DIRECTORY}")
file(WRITE "${OUTPUT}"
    "#pragma once\n"
    "#include <cstddef>\n"
    "namespace offgridai::embedded_neural_streamer {\n"
    "inline constexpr unsigned char ModelBytes[] = {${MODEL_BYTES}};\n"
    "inline constexpr std::size_t ModelSize = sizeof(ModelBytes);\n"
    "inline constexpr const char* ModelSha256 = \"${MODEL_SHA256}\";\n"
    "}\n")
