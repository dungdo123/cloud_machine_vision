/**
 * @file mvas_plugin.h
 * @brief MVAS Native Plugin Interface v2.0
 * 
 * This header defines the standard interface that all MVAS native plugins
 * must implement. Compile your plugin as a shared library:
 * - Windows: plugin.dll
 * - Linux: plugin.so
 * - macOS: plugin.dylib
 * 
 * @version 2.0.0
 * @date 2025-12-17
 * 
 * @example
 * // Minimal plugin implementation
 * #include "mvas_plugin.h"
 * 
 * static MVASPluginInfo g_info = {
 *     .name = "My Plugin",
 *     .version = "1.0.0",
 *     .model_type = "anomaly_detection",
 *     .input_width = 256,
 *     .input_height = 256,
 *     .input_format = "RGB"
 * };
 * 
 * MVAS_EXPORT const MVASPluginInfo* mvas_get_info(void) {
 *     return &g_info;
 * }
 * 
 * MVAS_EXPORT int32_t mvas_init(const char* model_path, const char* config) {
 *     // Load model here
 *     return 0;
 * }
 * 
 * MVAS_EXPORT int32_t mvas_infer(const MVASImage* image, MVASResult* result) {
 *     // Run inference here
 *     result->decision = "pass";
 *     result->confidence = 0.95f;
 *     result->anomaly_score = 0.05f;
 *     return 0;
 * }
 * 
 * MVAS_EXPORT void mvas_cleanup(void) {
 *     // Release resources
 * }
 * 
 * MVAS_EXPORT const char* mvas_get_error(void) {
 *     return "";
 * }
 */

#ifndef MVAS_PLUGIN_H
#define MVAS_PLUGIN_H

#include <stdint.h>
#include <stdbool.h>

/* Version information */
#define MVAS_PLUGIN_VERSION_MAJOR 2
#define MVAS_PLUGIN_VERSION_MINOR 0
#define MVAS_PLUGIN_VERSION_PATCH 0
#define MVAS_PLUGIN_VERSION "2.0.0"

/* Export macro for shared library functions */
#ifdef _WIN32
    #ifdef MVAS_PLUGIN_BUILD
        #define MVAS_EXPORT __declspec(dllexport)
    #else
        #define MVAS_EXPORT __declspec(dllimport)
    #endif
#else
    #define MVAS_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Error Codes
 * ============================================================================ */

#define MVAS_OK                     0
#define MVAS_ERROR_INIT_FAILED     -1
#define MVAS_ERROR_MODEL_NOT_FOUND -2
#define MVAS_ERROR_INVALID_INPUT   -3
#define MVAS_ERROR_INFERENCE       -4
#define MVAS_ERROR_NOT_INITIALIZED -5
#define MVAS_ERROR_OUT_OF_MEMORY   -6
#define MVAS_ERROR_UNSUPPORTED     -7

/* ============================================================================
 * Data Structures
 * ============================================================================ */

/**
 * @brief Image input structure
 * 
 * Represents an input image in HWC format (Height x Width x Channels).
 * The plugin is responsible for any preprocessing (resize, normalize, etc.).
 */
typedef struct {
    uint8_t* data;          /**< Raw pixel data pointer (not owned by caller) */
    int32_t width;          /**< Image width in pixels */
    int32_t height;         /**< Image height in pixels */
    int32_t channels;       /**< Number of channels (1=Gray, 3=RGB/BGR, 4=RGBA) */
    int32_t stride;         /**< Row stride in bytes (usually width * channels) */
    const char* format;     /**< Pixel format: "RGB", "BGR", "GRAY", "RGBA" */
} MVASImage;

/**
 * @brief Inference result structure
 * 
 * Contains the output of an inference operation. The plugin is responsible
 * for allocating any dynamic memory (anomaly_map, bboxes, visualization).
 * The MVAS runtime will call mvas_free_result() to release this memory.
 * 
 * For segmentation models:
 * - anomaly_map contains the segmentation probability map (defect class)
 * - visualization contains the overlay image with colored regions
 * 
 * For detection models:
 * - bboxes contains [x1, y1, x2, y2, score, class] per detection
 */
typedef struct {
    /* Required fields */
    const char* decision;       /**< Decision: "pass", "fail", or "review" */
    float confidence;           /**< Confidence score (0.0 - 1.0) */
    float anomaly_score;        /**< Anomaly score (0.0 - 1.0, higher = more anomalous) */
    float inference_time_ms;    /**< Inference time in milliseconds */
    
    /* Segmentation/Anomaly output (set to NULL if not available) */
    float* anomaly_map;         /**< Probability map (height * width floats, range 0-1) */
    int32_t anomaly_map_width;  /**< Map width (can differ from input) */
    int32_t anomaly_map_height; /**< Map height (can differ from input) */
    
    /* Object detection output */
    float* bboxes;              /**< Array of [x1, y1, x2, y2, score, class] per box */
    int32_t num_bboxes;         /**< Number of bounding boxes */
    
    /* Visualization overlay (RGB image) */
    uint8_t* visualization;     /**< Visualization image data (HWC format) */
    int32_t viz_width;          /**< Visualization width */
    int32_t viz_height;         /**< Visualization height */
    
    /* Extended details as JSON string */
    const char* details_json;   /**< JSON with model-specific details, or NULL */
} MVASResult;

/**
 * @brief Plugin information structure
 * 
 * Provides metadata about the plugin. This is queried by mvas_get_info()
 * and used by the MVAS runtime for display and configuration.
 */
typedef struct {
    const char* name;           /**< Display name: "Bottle Cap Inspection" */
    const char* version;        /**< Version string: "1.0.0" */
    const char* author;         /**< Author/Company name */
    const char* description;    /**< Brief description of the plugin */
    const char* model_type;     /**< Type: "anomaly_detection", "classification", 
                                     "object_detection", "segmentation" */
    int32_t input_width;        /**< Expected input width (for display only) */
    int32_t input_height;       /**< Expected input height (for display only) */
    const char* input_format;   /**< Preferred input format: "RGB" or "BGR" */
} MVASPluginInfo;

/**
 * @brief Configuration parameter for runtime updates
 * 
 * Used with mvas_set_config() to update parameters at runtime.
 */
typedef struct {
    const char* key;            /**< Parameter name: "threshold" */
    const char* value;          /**< Parameter value as string: "0.5" */
} MVASConfigParam;

/* ============================================================================
 * Required Functions - Must be implemented by all plugins
 * ============================================================================ */

/**
 * @brief Get plugin information
 * 
 * Called by the MVAS runtime to get metadata about the plugin.
 * This function may be called before mvas_init().
 * 
 * @return Pointer to static MVASPluginInfo structure (must not be freed)
 */
MVAS_EXPORT const MVASPluginInfo* mvas_get_info(void);

/**
 * @brief Initialize the plugin
 * 
 * Called once when the plugin is loaded. The plugin should:
 * - Load the model from model_path
 * - Parse config_json for initial settings
 * - Allocate any required resources
 * - Perform warmup if needed
 * 
 * @param model_path Path to the model file (engine, onnx, etc.)
 * @param config_json JSON string with configuration, or "{}" if empty
 * @return MVAS_OK (0) on success, negative error code on failure
 */
MVAS_EXPORT int32_t mvas_init(const char* model_path, const char* config_json);

/**
 * @brief Run inference on an image
 * 
 * The main inference function. The plugin should:
 * 1. Preprocess the input image
 * 2. Run model inference
 * 3. Postprocess results
 * 4. Fill the result structure
 * 
 * @param image Input image (not owned, do not modify or free)
 * @param result Output result structure (plugin fills this)
 * @return MVAS_OK (0) on success, negative error code on failure
 */
MVAS_EXPORT int32_t mvas_infer(const MVASImage* image, MVASResult* result);

/**
 * @brief Cleanup and release resources
 * 
 * Called when the plugin is being unloaded. The plugin should:
 * - Release all allocated memory
 * - Unload the model
 * - Close any open handles
 */
MVAS_EXPORT void mvas_cleanup(void);

/**
 * @brief Get the last error message
 * 
 * Called after a function returns an error code to get details.
 * 
 * @return Error message string (must not be freed)
 */
MVAS_EXPORT const char* mvas_get_error(void);

/* ============================================================================
 * Optional Functions - Implement for enhanced functionality
 * ============================================================================ */

/**
 * @brief Warmup the model
 * 
 * Run dummy inferences to initialize GPU kernels and optimize performance.
 * If not implemented, the runtime will perform manual warmup.
 * 
 * @param iterations Number of warmup iterations
 * @return MVAS_OK on success
 */
MVAS_EXPORT int32_t mvas_warmup(int32_t iterations);

/**
 * @brief Update configuration at runtime
 * 
 * Allows changing parameters (like thresholds) without reloading.
 * 
 * @param params Array of key-value parameters
 * @param num_params Number of parameters
 * @return MVAS_OK on success
 */
MVAS_EXPORT int32_t mvas_set_config(const MVASConfigParam* params, int32_t num_params);

/**
 * @brief Get current configuration
 * 
 * Returns the current configuration as a JSON string.
 * 
 * @return JSON string (must not be freed), or NULL if not supported
 */
MVAS_EXPORT const char* mvas_get_config(void);

/**
 * @brief Free memory allocated in MVASResult
 * 
 * Called by the runtime after copying data from MVASResult.
 * Free any memory allocated by the plugin for anomaly_map, bboxes, etc.
 * 
 * @param result Result structure to free
 */
MVAS_EXPORT void mvas_free_result(MVASResult* result);

/**
 * @brief Run batch inference
 * 
 * Process multiple images in a single call for better throughput.
 * 
 * @param images Array of input images
 * @param num_images Number of images
 * @param results Array of result structures (pre-allocated by caller)
 * @return MVAS_OK on success
 */
MVAS_EXPORT int32_t mvas_infer_batch(
    const MVASImage* images,
    int32_t num_images,
    MVASResult* results
);

#ifdef __cplusplus
}
#endif

#endif /* MVAS_PLUGIN_H */

