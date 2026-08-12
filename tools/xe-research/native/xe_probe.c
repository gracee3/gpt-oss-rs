#define _GNU_SOURCE
#include "xe_probe.h"

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <level_zero/ze_api.h>

#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef CL_DEVICE_ID_INTEL
#define CL_DEVICE_ID_INTEL 0x4251
#endif

#define XE_TIMEOUT_FOREVER UINT64_MAX

struct cl_api {
    void *library;
    __typeof__(&clGetPlatformIDs) clGetPlatformIDs;
    __typeof__(&clGetPlatformInfo) clGetPlatformInfo;
    __typeof__(&clGetDeviceIDs) clGetDeviceIDs;
    __typeof__(&clGetDeviceInfo) clGetDeviceInfo;
    __typeof__(&clCreateContext) clCreateContext;
    __typeof__(&clReleaseContext) clReleaseContext;
    __typeof__(&clCreateCommandQueueWithProperties) clCreateCommandQueueWithProperties;
    __typeof__(&clReleaseCommandQueue) clReleaseCommandQueue;
    __typeof__(&clCreateProgramWithSource) clCreateProgramWithSource;
    __typeof__(&clCreateProgramWithIL) clCreateProgramWithIL;
    __typeof__(&clCreateProgramWithBinary) clCreateProgramWithBinary;
    __typeof__(&clBuildProgram) clBuildProgram;
    __typeof__(&clGetProgramInfo) clGetProgramInfo;
    __typeof__(&clGetProgramBuildInfo) clGetProgramBuildInfo;
    __typeof__(&clReleaseProgram) clReleaseProgram;
    __typeof__(&clCreateKernel) clCreateKernel;
    __typeof__(&clSetKernelArg) clSetKernelArg;
    __typeof__(&clReleaseKernel) clReleaseKernel;
    __typeof__(&clCreateBuffer) clCreateBuffer;
    __typeof__(&clReleaseMemObject) clReleaseMemObject;
    __typeof__(&clEnqueueWriteBuffer) clEnqueueWriteBuffer;
    __typeof__(&clEnqueueReadBuffer) clEnqueueReadBuffer;
    __typeof__(&clEnqueueMapBuffer) clEnqueueMapBuffer;
    __typeof__(&clEnqueueUnmapMemObject) clEnqueueUnmapMemObject;
    __typeof__(&clSVMAlloc) clSVMAlloc;
    __typeof__(&clSVMFree) clSVMFree;
    __typeof__(&clEnqueueSVMMap) clEnqueueSVMMap;
    __typeof__(&clEnqueueSVMUnmap) clEnqueueSVMUnmap;
    __typeof__(&clEnqueueNDRangeKernel) clEnqueueNDRangeKernel;
    __typeof__(&clFinish) clFinish;
    __typeof__(&clGetEventProfilingInfo) clGetEventProfilingInfo;
    __typeof__(&clReleaseEvent) clReleaseEvent;
    __typeof__(&clGetDeviceAndHostTimer) clGetDeviceAndHostTimer;
};

struct ze_api {
    void *library;
    __typeof__(&zeInit) zeInit;
    __typeof__(&zeDriverGet) zeDriverGet;
    __typeof__(&zeDriverGetApiVersion) zeDriverGetApiVersion;
    __typeof__(&zeDeviceGet) zeDeviceGet;
    __typeof__(&zeDeviceGetProperties) zeDeviceGetProperties;
    __typeof__(&zeDeviceGetComputeProperties) zeDeviceGetComputeProperties;
    __typeof__(&zeDeviceGetMemoryProperties) zeDeviceGetMemoryProperties;
    __typeof__(&zeDeviceGetCommandQueueGroupProperties) zeDeviceGetCommandQueueGroupProperties;
    __typeof__(&zeDeviceGetModuleProperties) zeDeviceGetModuleProperties;
    __typeof__(&zeDeviceGetGlobalTimestamps) zeDeviceGetGlobalTimestamps;
    __typeof__(&zeContextCreate) zeContextCreate;
    __typeof__(&zeContextDestroy) zeContextDestroy;
    __typeof__(&zeCommandQueueCreate) zeCommandQueueCreate;
    __typeof__(&zeCommandQueueDestroy) zeCommandQueueDestroy;
    __typeof__(&zeCommandQueueExecuteCommandLists) zeCommandQueueExecuteCommandLists;
    __typeof__(&zeCommandQueueSynchronize) zeCommandQueueSynchronize;
    __typeof__(&zeCommandListCreate) zeCommandListCreate;
    __typeof__(&zeCommandListCreateImmediate) zeCommandListCreateImmediate;
    __typeof__(&zeCommandListDestroy) zeCommandListDestroy;
    __typeof__(&zeCommandListClose) zeCommandListClose;
    __typeof__(&zeCommandListReset) zeCommandListReset;
    __typeof__(&zeCommandListHostSynchronize) zeCommandListHostSynchronize;
    __typeof__(&zeCommandListAppendMemoryCopy) zeCommandListAppendMemoryCopy;
    __typeof__(&zeCommandListAppendLaunchKernel) zeCommandListAppendLaunchKernel;
    __typeof__(&zeMemAllocHost) zeMemAllocHost;
    __typeof__(&zeMemAllocDevice) zeMemAllocDevice;
    __typeof__(&zeMemAllocShared) zeMemAllocShared;
    __typeof__(&zeMemFree) zeMemFree;
    __typeof__(&zeModuleCreate) zeModuleCreate;
    __typeof__(&zeModuleDestroy) zeModuleDestroy;
    __typeof__(&zeModuleGetNativeBinary) zeModuleGetNativeBinary;
    __typeof__(&zeModuleBuildLogGetString) zeModuleBuildLogGetString;
    __typeof__(&zeModuleBuildLogDestroy) zeModuleBuildLogDestroy;
    __typeof__(&zeKernelCreate) zeKernelCreate;
    __typeof__(&zeKernelDestroy) zeKernelDestroy;
    __typeof__(&zeKernelSetArgumentValue) zeKernelSetArgumentValue;
    __typeof__(&zeKernelSetGroupSize) zeKernelSetGroupSize;
    __typeof__(&zeEventPoolCreate) zeEventPoolCreate;
    __typeof__(&zeEventPoolDestroy) zeEventPoolDestroy;
    __typeof__(&zeEventCreate) zeEventCreate;
    __typeof__(&zeEventDestroy) zeEventDestroy;
    __typeof__(&zeEventHostReset) zeEventHostReset;
    __typeof__(&zeEventQueryKernelTimestamp) zeEventQueryKernelTimestamp;
};

struct xe_session {
    uint32_t backend;
    uint8_t immediate;
    uint64_t timer_resolution;
    struct xe_session_info info;
    union {
        struct {
            struct cl_api api;
            cl_platform_id platform;
            cl_device_id device;
            cl_context context;
            cl_command_queue queue;
            cl_program program;
            cl_kernel kernel;
        } cl;
        struct {
            struct ze_api api;
            ze_driver_handle_t driver;
            ze_device_handle_t device;
            ze_context_handle_t context;
            ze_command_queue_handle_t queue;
            ze_command_list_handle_t list;
            ze_module_handle_t module;
            ze_kernel_handle_t kernel;
            ze_event_pool_handle_t event_pool;
            ze_event_handle_t event;
            uint32_t queue_ordinal;
        } ze;
    } u;
};

struct xe_buffer {
    uint32_t backend;
    uint32_t kind;
    size_t size;
    union {
        cl_mem cl;
        void *ze;
    } handle;
};

static uint64_t monotonic_ns(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0) {
        return 0;
    }
    return (uint64_t)value.tv_sec * 1000000000ULL + (uint64_t)value.tv_nsec;
}

static void text_copy(char *destination, size_t capacity, const char *source) {
    if (capacity == 0) {
        return;
    }
    if (source == NULL) {
        destination[0] = '\0';
        return;
    }
    snprintf(destination, capacity, "%s", source);
}

static void error_code(char *destination, size_t capacity, const char *operation, int64_t code) {
    if (capacity != 0) {
        snprintf(destination, capacity, "%s failed with status %lld", operation, (long long)code);
    }
}

static void library_path(void *symbol, char *destination, size_t capacity) {
    Dl_info information;
    memset(&information, 0, sizeof(information));
    if (dladdr(symbol, &information) != 0 && information.dli_fname != NULL) {
        text_copy(destination, capacity, information.dli_fname);
    }
}

#define LOAD_SYMBOL(table, name, error)                                                    \
    do {                                                                                  \
        *(void **)(&(table)->name) = dlsym((table)->library, #name);                      \
        if ((table)->name == NULL) {                                                       \
            snprintf((error), XE_TEXT, "missing symbol %s: %s", #name, dlerror());       \
            return -1;                                                                    \
        }                                                                                 \
    } while (0)

static int load_cl(struct cl_api *api, char *error) {
    memset(api, 0, sizeof(*api));
    api->library = dlopen("/usr/lib/x86_64-linux-gnu/libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL);
    if (api->library == NULL) {
        text_copy(error, XE_TEXT, dlerror());
        return -1;
    }
    LOAD_SYMBOL(api, clGetPlatformIDs, error);
    LOAD_SYMBOL(api, clGetPlatformInfo, error);
    LOAD_SYMBOL(api, clGetDeviceIDs, error);
    LOAD_SYMBOL(api, clGetDeviceInfo, error);
    LOAD_SYMBOL(api, clCreateContext, error);
    LOAD_SYMBOL(api, clReleaseContext, error);
    LOAD_SYMBOL(api, clCreateCommandQueueWithProperties, error);
    LOAD_SYMBOL(api, clReleaseCommandQueue, error);
    LOAD_SYMBOL(api, clCreateProgramWithSource, error);
    LOAD_SYMBOL(api, clCreateProgramWithIL, error);
    LOAD_SYMBOL(api, clCreateProgramWithBinary, error);
    LOAD_SYMBOL(api, clBuildProgram, error);
    LOAD_SYMBOL(api, clGetProgramInfo, error);
    LOAD_SYMBOL(api, clGetProgramBuildInfo, error);
    LOAD_SYMBOL(api, clReleaseProgram, error);
    LOAD_SYMBOL(api, clCreateKernel, error);
    LOAD_SYMBOL(api, clSetKernelArg, error);
    LOAD_SYMBOL(api, clReleaseKernel, error);
    LOAD_SYMBOL(api, clCreateBuffer, error);
    LOAD_SYMBOL(api, clReleaseMemObject, error);
    LOAD_SYMBOL(api, clEnqueueWriteBuffer, error);
    LOAD_SYMBOL(api, clEnqueueReadBuffer, error);
    LOAD_SYMBOL(api, clEnqueueMapBuffer, error);
    LOAD_SYMBOL(api, clEnqueueUnmapMemObject, error);
    LOAD_SYMBOL(api, clSVMAlloc, error);
    LOAD_SYMBOL(api, clSVMFree, error);
    LOAD_SYMBOL(api, clEnqueueSVMMap, error);
    LOAD_SYMBOL(api, clEnqueueSVMUnmap, error);
    LOAD_SYMBOL(api, clEnqueueNDRangeKernel, error);
    LOAD_SYMBOL(api, clFinish, error);
    LOAD_SYMBOL(api, clGetEventProfilingInfo, error);
    LOAD_SYMBOL(api, clReleaseEvent, error);
    LOAD_SYMBOL(api, clGetDeviceAndHostTimer, error);
    return 0;
}

static int load_ze(struct ze_api *api, char *error) {
    memset(api, 0, sizeof(*api));
    api->library = dlopen("/usr/lib/x86_64-linux-gnu/libze_loader.so.1", RTLD_NOW | RTLD_LOCAL);
    if (api->library == NULL) {
        text_copy(error, XE_TEXT, dlerror());
        return -1;
    }
    LOAD_SYMBOL(api, zeInit, error);
    LOAD_SYMBOL(api, zeDriverGet, error);
    LOAD_SYMBOL(api, zeDriverGetApiVersion, error);
    LOAD_SYMBOL(api, zeDeviceGet, error);
    LOAD_SYMBOL(api, zeDeviceGetProperties, error);
    LOAD_SYMBOL(api, zeDeviceGetComputeProperties, error);
    LOAD_SYMBOL(api, zeDeviceGetMemoryProperties, error);
    LOAD_SYMBOL(api, zeDeviceGetCommandQueueGroupProperties, error);
    LOAD_SYMBOL(api, zeDeviceGetModuleProperties, error);
    LOAD_SYMBOL(api, zeDeviceGetGlobalTimestamps, error);
    LOAD_SYMBOL(api, zeContextCreate, error);
    LOAD_SYMBOL(api, zeContextDestroy, error);
    LOAD_SYMBOL(api, zeCommandQueueCreate, error);
    LOAD_SYMBOL(api, zeCommandQueueDestroy, error);
    LOAD_SYMBOL(api, zeCommandQueueExecuteCommandLists, error);
    LOAD_SYMBOL(api, zeCommandQueueSynchronize, error);
    LOAD_SYMBOL(api, zeCommandListCreate, error);
    LOAD_SYMBOL(api, zeCommandListCreateImmediate, error);
    LOAD_SYMBOL(api, zeCommandListDestroy, error);
    LOAD_SYMBOL(api, zeCommandListClose, error);
    LOAD_SYMBOL(api, zeCommandListReset, error);
    LOAD_SYMBOL(api, zeCommandListHostSynchronize, error);
    LOAD_SYMBOL(api, zeCommandListAppendMemoryCopy, error);
    LOAD_SYMBOL(api, zeCommandListAppendLaunchKernel, error);
    LOAD_SYMBOL(api, zeMemAllocHost, error);
    LOAD_SYMBOL(api, zeMemAllocDevice, error);
    LOAD_SYMBOL(api, zeMemAllocShared, error);
    LOAD_SYMBOL(api, zeMemFree, error);
    LOAD_SYMBOL(api, zeModuleCreate, error);
    LOAD_SYMBOL(api, zeModuleDestroy, error);
    LOAD_SYMBOL(api, zeModuleGetNativeBinary, error);
    LOAD_SYMBOL(api, zeModuleBuildLogGetString, error);
    LOAD_SYMBOL(api, zeModuleBuildLogDestroy, error);
    LOAD_SYMBOL(api, zeKernelCreate, error);
    LOAD_SYMBOL(api, zeKernelDestroy, error);
    LOAD_SYMBOL(api, zeKernelSetArgumentValue, error);
    LOAD_SYMBOL(api, zeKernelSetGroupSize, error);
    LOAD_SYMBOL(api, zeEventPoolCreate, error);
    LOAD_SYMBOL(api, zeEventPoolDestroy, error);
    LOAD_SYMBOL(api, zeEventCreate, error);
    LOAD_SYMBOL(api, zeEventDestroy, error);
    LOAD_SYMBOL(api, zeEventHostReset, error);
    LOAD_SYMBOL(api, zeEventQueryKernelTimestamp, error);
    return 0;
}

static void close_cl(struct cl_api *api) {
    if (api->library != NULL) {
        dlclose(api->library);
        api->library = NULL;
    }
}

static void close_ze(struct ze_api *api) {
    if (api->library != NULL) {
        dlclose(api->library);
        api->library = NULL;
    }
}

static void cl_string(
    struct cl_api *api,
    cl_device_id device,
    cl_device_info property,
    char *destination,
    size_t capacity) {
    size_t length = 0;
    if (api->clGetDeviceInfo(device, property, 0, NULL, &length) != CL_SUCCESS || length == 0) {
        destination[0] = '\0';
        return;
    }
    char *temporary = calloc(length + 1, 1);
    if (temporary == NULL) {
        destination[0] = '\0';
        return;
    }
    if (api->clGetDeviceInfo(device, property, length, temporary, NULL) == CL_SUCCESS) {
        text_copy(destination, capacity, temporary);
    } else {
        destination[0] = '\0';
    }
    free(temporary);
}

static void cl_platform_string(
    struct cl_api *api,
    cl_platform_id platform,
    cl_platform_info property,
    char *destination,
    size_t capacity) {
    size_t length = 0;
    if (api->clGetPlatformInfo(platform, property, 0, NULL, &length) != CL_SUCCESS || length == 0) {
        destination[0] = '\0';
        return;
    }
    char *temporary = calloc(length + 1, 1);
    if (temporary == NULL) {
        destination[0] = '\0';
        return;
    }
    if (api->clGetPlatformInfo(platform, property, length, temporary, NULL) == CL_SUCCESS) {
        text_copy(destination, capacity, temporary);
    } else {
        destination[0] = '\0';
    }
    free(temporary);
}

static int cl_select_device(
    struct xe_session *session,
    uint32_t expected_vendor,
    uint32_t expected_device,
    char *error) {
    struct cl_api *api = &session->u.cl.api;
    cl_uint platform_count = 0;
    cl_int result = api->clGetPlatformIDs(0, NULL, &platform_count);
    if (result != CL_SUCCESS || platform_count == 0) {
        error_code(error, XE_TEXT, "clGetPlatformIDs", result);
        return -1;
    }
    cl_platform_id *platforms = calloc(platform_count, sizeof(*platforms));
    if (platforms == NULL) {
        text_copy(error, XE_TEXT, "out of memory enumerating OpenCL platforms");
        return -1;
    }
    result = api->clGetPlatformIDs(platform_count, platforms, NULL);
    if (result != CL_SUCCESS) {
        free(platforms);
        error_code(error, XE_TEXT, "clGetPlatformIDs(list)", result);
        return -1;
    }

    for (cl_uint platform_index = 0; platform_index < platform_count; ++platform_index) {
        cl_uint device_count = 0;
        result = api->clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_GPU, 0, NULL, &device_count);
        if (result != CL_SUCCESS || device_count == 0) {
            continue;
        }
        cl_device_id *devices = calloc(device_count, sizeof(*devices));
        if (devices == NULL) {
            continue;
        }
        result = api->clGetDeviceIDs(
            platforms[platform_index], CL_DEVICE_TYPE_GPU, device_count, devices, NULL);
        if (result != CL_SUCCESS) {
            free(devices);
            continue;
        }
        for (cl_uint device_index = 0; device_index < device_count; ++device_index) {
            cl_uint vendor = 0;
            cl_uint device_id = 0;
            api->clGetDeviceInfo(
                devices[device_index], CL_DEVICE_VENDOR_ID, sizeof(vendor), &vendor, NULL);
            api->clGetDeviceInfo(
                devices[device_index], CL_DEVICE_ID_INTEL, sizeof(device_id), &device_id, NULL);
            if (vendor == expected_vendor && device_id == expected_device) {
                session->u.cl.platform = platforms[platform_index];
                session->u.cl.device = devices[device_index];
                free(devices);
                free(platforms);
                return 0;
            }
        }
        free(devices);
    }
    free(platforms);
    snprintf(error, XE_TEXT, "OpenCL device %04x:%04x was not found", expected_vendor, expected_device);
    return -1;
}

static void fill_cl_info(struct xe_session *session) {
    struct cl_api *api = &session->u.cl.api;
    cl_device_id device = session->u.cl.device;
    struct xe_session_info *info = &session->info;
    info->backend = XE_BACKEND_OPENCL;
    info->vendor_id = 0;
    info->device_id = 0;
    api->clGetDeviceInfo(
        device, CL_DEVICE_VENDOR_ID, sizeof(info->vendor_id), &info->vendor_id, NULL);
    api->clGetDeviceInfo(
        device, CL_DEVICE_ID_INTEL, sizeof(info->device_id), &info->device_id, NULL);
    cl_platform_string(
        api, session->u.cl.platform, CL_PLATFORM_NAME, info->platform_name, sizeof(info->platform_name));
    cl_string(api, device, CL_DEVICE_NAME, info->device_name, sizeof(info->device_name));
    cl_string(api, device, CL_DRIVER_VERSION, info->driver_version, sizeof(info->driver_version));
    cl_string(api, device, CL_DEVICE_VERSION, info->device_version, sizeof(info->device_version));
    cl_string(api, device, CL_DEVICE_EXTENSIONS, info->extensions, sizeof(info->extensions));
    library_path((void *)api->clGetPlatformIDs, info->library_path, sizeof(info->library_path));

    cl_uint numeric = 0;
    api->clGetDeviceInfo(device, CL_DEVICE_NUMERIC_VERSION, sizeof(numeric), &numeric, NULL);
    info->api_version = numeric;
    api->clGetDeviceInfo(
        device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(info->compute_units), &info->compute_units, NULL);
    size_t group_size = 0;
    api->clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(group_size), &group_size, NULL);
    info->max_group_size = (uint32_t)group_size;
    cl_ulong memory = 0;
    api->clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memory), &memory, NULL);
    info->global_memory_bytes = memory;
    api->clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(memory), &memory, NULL);
    info->max_allocation_bytes = memory;
    api->clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(memory), &memory, NULL);
    info->local_memory_bytes = memory;
    size_t timer_resolution = 0;
    api->clGetDeviceInfo(
        device,
        CL_DEVICE_PROFILING_TIMER_RESOLUTION,
        sizeof(timer_resolution),
        &timer_resolution,
        NULL);
    info->timer_resolution = timer_resolution;
    cl_bool boolean = CL_FALSE;
    api->clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(boolean), &boolean, NULL);
    info->integrated = boolean == CL_TRUE;
    api->clGetDeviceInfo(device, CL_DEVICE_COMPILER_AVAILABLE, sizeof(boolean), &boolean, NULL);
    info->compiler_available = boolean == CL_TRUE;
    info->il_supported = strstr(info->extensions, "cl_khr_il_program") != NULL;
    info->integer_dot_supported = strstr(info->extensions, "cl_khr_integer_dot_product") != NULL;
    cl_ulong device_timestamp = 0;
    cl_ulong host_timestamp = 0;
    info->host_clock_correlation_supported =
        api->clGetDeviceAndHostTimer != NULL &&
        api->clGetDeviceAndHostTimer(device, &device_timestamp, &host_timestamp) == CL_SUCCESS;

#ifdef CL_DEVICE_SUB_GROUP_SIZES_INTEL
    size_t subgroup_bytes = 0;
    if (api->clGetDeviceInfo(
            device, CL_DEVICE_SUB_GROUP_SIZES_INTEL, 0, NULL, &subgroup_bytes) == CL_SUCCESS) {
        size_t subgroup_count = subgroup_bytes / sizeof(size_t);
        size_t temporary[8] = {0};
        if (subgroup_count > 8) {
            subgroup_count = 8;
        }
        if (api->clGetDeviceInfo(
                device,
                CL_DEVICE_SUB_GROUP_SIZES_INTEL,
                subgroup_count * sizeof(size_t),
                temporary,
                NULL) == CL_SUCCESS) {
            info->subgroup_count = (uint32_t)subgroup_count;
            for (size_t index = 0; index < subgroup_count; ++index) {
                info->subgroups[index] = (uint32_t)temporary[index];
            }
        }
    }
#endif
}

static int ze_select_device(
    struct xe_session *session,
    uint32_t expected_vendor,
    uint32_t expected_device,
    char *error) {
    struct ze_api *api = &session->u.ze.api;
    ze_result_t result = api->zeInit(ZE_INIT_FLAG_GPU_ONLY);
    if (result != ZE_RESULT_SUCCESS) {
        error_code(error, XE_TEXT, "zeInit", result);
        return -1;
    }
    uint32_t driver_count = 0;
    result = api->zeDriverGet(&driver_count, NULL);
    if (result != ZE_RESULT_SUCCESS || driver_count == 0) {
        error_code(error, XE_TEXT, "zeDriverGet", result);
        return -1;
    }
    ze_driver_handle_t *drivers = calloc(driver_count, sizeof(*drivers));
    if (drivers == NULL) {
        text_copy(error, XE_TEXT, "out of memory enumerating Level Zero drivers");
        return -1;
    }
    result = api->zeDriverGet(&driver_count, drivers);
    if (result != ZE_RESULT_SUCCESS) {
        free(drivers);
        error_code(error, XE_TEXT, "zeDriverGet(list)", result);
        return -1;
    }
    for (uint32_t driver_index = 0; driver_index < driver_count; ++driver_index) {
        uint32_t device_count = 0;
        if (api->zeDeviceGet(drivers[driver_index], &device_count, NULL) != ZE_RESULT_SUCCESS ||
            device_count == 0) {
            continue;
        }
        ze_device_handle_t *devices = calloc(device_count, sizeof(*devices));
        if (devices == NULL) {
            continue;
        }
        if (api->zeDeviceGet(drivers[driver_index], &device_count, devices) != ZE_RESULT_SUCCESS) {
            free(devices);
            continue;
        }
        for (uint32_t device_index = 0; device_index < device_count; ++device_index) {
            ze_device_properties_t properties = {
                .stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES,
                .pNext = NULL,
            };
            if (api->zeDeviceGetProperties(devices[device_index], &properties) == ZE_RESULT_SUCCESS &&
                properties.vendorId == expected_vendor && properties.deviceId == expected_device) {
                session->u.ze.driver = drivers[driver_index];
                session->u.ze.device = devices[device_index];
                free(devices);
                free(drivers);
                return 0;
            }
        }
        free(devices);
    }
    free(drivers);
    snprintf(error, XE_TEXT, "Level Zero device %04x:%04x was not found", expected_vendor, expected_device);
    return -1;
}

static int ze_queue_ordinal(struct xe_session *session, char *error) {
    struct ze_api *api = &session->u.ze.api;
    uint32_t count = 0;
    ze_result_t result =
        api->zeDeviceGetCommandQueueGroupProperties(session->u.ze.device, &count, NULL);
    if (result != ZE_RESULT_SUCCESS || count == 0) {
        error_code(error, XE_TEXT, "zeDeviceGetCommandQueueGroupProperties", result);
        return -1;
    }
    ze_command_queue_group_properties_t *properties = calloc(count, sizeof(*properties));
    if (properties == NULL) {
        text_copy(error, XE_TEXT, "out of memory enumerating Level Zero queue groups");
        return -1;
    }
    result = api->zeDeviceGetCommandQueueGroupProperties(session->u.ze.device, &count, properties);
    if (result != ZE_RESULT_SUCCESS) {
        free(properties);
        error_code(error, XE_TEXT, "zeDeviceGetCommandQueueGroupProperties(list)", result);
        return -1;
    }
    for (uint32_t index = 0; index < count; ++index) {
        if ((properties[index].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) != 0) {
            session->u.ze.queue_ordinal = index;
            free(properties);
            return 0;
        }
    }
    free(properties);
    text_copy(error, XE_TEXT, "selected Level Zero device has no compute queue group");
    return -1;
}

static void fill_ze_info(struct xe_session *session) {
    struct ze_api *api = &session->u.ze.api;
    struct xe_session_info *info = &session->info;
    info->backend = XE_BACKEND_LEVEL_ZERO;
    ze_device_properties_t properties = {
        .stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES,
        .pNext = NULL,
    };
    if (api->zeDeviceGetProperties(session->u.ze.device, &properties) == ZE_RESULT_SUCCESS) {
        info->vendor_id = properties.vendorId;
        info->device_id = properties.deviceId;
        info->compute_units = properties.numSlices * properties.numSubslicesPerSlice *
                              properties.numEUsPerSubslice;
        info->integrated = (properties.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) != 0;
        info->timer_resolution = properties.timerResolution;
        info->timestamp_valid_bits = properties.timestampValidBits;
        info->kernel_timestamp_valid_bits = properties.kernelTimestampValidBits;
        info->max_allocation_bytes = properties.maxMemAllocSize;
        text_copy(info->device_name, sizeof(info->device_name), properties.name);
        session->timer_resolution = properties.timerResolution;
    }
    ze_api_version_t api_version = 0;
    if (api->zeDriverGetApiVersion(session->u.ze.driver, &api_version) == ZE_RESULT_SUCCESS) {
        info->api_version = api_version;
    }
    ze_device_compute_properties_t compute = {
        .stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES,
        .pNext = NULL,
    };
    if (api->zeDeviceGetComputeProperties(session->u.ze.device, &compute) == ZE_RESULT_SUCCESS) {
        info->max_group_size = compute.maxTotalGroupSize;
        info->local_memory_bytes = compute.maxSharedLocalMemory;
        info->subgroup_count = compute.numSubGroupSizes > 8 ? 8 : compute.numSubGroupSizes;
        for (uint32_t index = 0; index < info->subgroup_count; ++index) {
            info->subgroups[index] = compute.subGroupSizes[index];
        }
    }
    uint32_t memory_count = 0;
    if (api->zeDeviceGetMemoryProperties(session->u.ze.device, &memory_count, NULL) ==
            ZE_RESULT_SUCCESS &&
        memory_count != 0) {
        ze_device_memory_properties_t *memory = calloc(memory_count, sizeof(*memory));
        if (memory != NULL) {
            for (uint32_t index = 0; index < memory_count; ++index) {
                memory[index].stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
            }
            if (api->zeDeviceGetMemoryProperties(session->u.ze.device, &memory_count, memory) ==
                ZE_RESULT_SUCCESS) {
                for (uint32_t index = 0; index < memory_count; ++index) {
                    info->global_memory_bytes += memory[index].totalSize;
                }
            }
            free(memory);
        }
    }
    ze_device_module_properties_t module = {
        .stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES,
        .pNext = NULL,
    };
    if (api->zeDeviceGetModuleProperties(session->u.ze.device, &module) == ZE_RESULT_SUCCESS) {
        snprintf(
            info->device_version,
            sizeof(info->device_version),
            "SPIR-V %u.%u",
            ZE_MAJOR_VERSION(module.spirvVersionSupported),
            ZE_MINOR_VERSION(module.spirvVersionSupported));
        snprintf(
            info->extensions,
            sizeof(info->extensions),
            "module_flags=0x%x fp_flags=0x%x",
            module.flags,
            module.fp32flags);
        info->il_supported = 1;
    }
    info->compiler_available = 1;
    /* Core Level Zero exposes SPIR-V ingestion but no integer-dot capability bit.
     * Keep this false until accepted compiler/native-code evidence proves lowering. */
    info->integer_dot_supported = 0;
    uint64_t host = 0;
    uint64_t device = 0;
    info->host_clock_correlation_supported =
        api->zeDeviceGetGlobalTimestamps(session->u.ze.device, &host, &device) == ZE_RESULT_SUCCESS;
    library_path((void *)api->zeInit, info->library_path, sizeof(info->library_path));
    text_copy(info->platform_name, sizeof(info->platform_name), "Intel Level Zero");
    text_copy(info->driver_version, sizeof(info->driver_version), "reported by loaded driver API");
}

static void cl_build_log(struct xe_session *session) {
    size_t length = 0;
    struct cl_api *api = &session->u.cl.api;
    if (session->u.cl.program == NULL) {
        return;
    }
    if (api->clGetProgramBuildInfo(
            session->u.cl.program,
            session->u.cl.device,
            CL_PROGRAM_BUILD_LOG,
            0,
            NULL,
            &length) != CL_SUCCESS ||
        length == 0) {
        return;
    }
    char *temporary = calloc(length + 1, 1);
    if (temporary == NULL) {
        return;
    }
    if (api->clGetProgramBuildInfo(
            session->u.cl.program,
            session->u.cl.device,
            CL_PROGRAM_BUILD_LOG,
            length,
            temporary,
            NULL) == CL_SUCCESS) {
        text_copy(session->info.build_log, sizeof(session->info.build_log), temporary);
    }
    free(temporary);
}

static int initialize_cl(
    struct xe_session *session,
    uint32_t expected_vendor,
    uint32_t expected_device,
    uint32_t artifact_kind,
    const uint8_t *artifact,
    size_t artifact_len,
    const char *build_options,
    const char *entry_point) {
    struct cl_api *api = &session->u.cl.api;
    if (load_cl(api, session->info.error) != 0 ||
        cl_select_device(session, expected_vendor, expected_device, session->info.error) != 0) {
        return -1;
    }
    fill_cl_info(session);
    cl_int result = CL_SUCCESS;
    session->u.cl.context = api->clCreateContext(
        NULL, 1, &session->u.cl.device, NULL, NULL, &result);
    if (session->u.cl.context == NULL || result != CL_SUCCESS) {
        error_code(session->info.error, XE_TEXT, "clCreateContext", result);
        return -1;
    }
    const cl_queue_properties properties[] = {
        CL_QUEUE_PROPERTIES,
        CL_QUEUE_PROFILING_ENABLE,
        0,
    };
    session->u.cl.queue = api->clCreateCommandQueueWithProperties(
        session->u.cl.context, session->u.cl.device, properties, &result);
    if (session->u.cl.queue == NULL || result != CL_SUCCESS) {
        error_code(session->info.error, XE_TEXT, "clCreateCommandQueueWithProperties", result);
        return -1;
    }
    if (artifact == NULL || artifact_len == 0) {
        return 0;
    }
    if (artifact_kind == XE_ARTIFACT_OPENCL_SOURCE) {
        const char *source = (const char *)artifact;
        session->u.cl.program = api->clCreateProgramWithSource(
            session->u.cl.context, 1, &source, &artifact_len, &result);
    } else if (artifact_kind == XE_ARTIFACT_SPIRV) {
        session->u.cl.program = api->clCreateProgramWithIL(
            session->u.cl.context, artifact, artifact_len, &result);
    } else if (artifact_kind == XE_ARTIFACT_OPENCL_BINARY) {
        const unsigned char *binary = artifact;
        cl_int binary_status = CL_SUCCESS;
        session->u.cl.program = api->clCreateProgramWithBinary(
            session->u.cl.context,
            1,
            &session->u.cl.device,
            &artifact_len,
            &binary,
            &binary_status,
            &result);
        if (binary_status != CL_SUCCESS && result == CL_SUCCESS) {
            result = binary_status;
        }
    } else {
        text_copy(session->info.error, XE_TEXT, "unsupported artifact kind for OpenCL");
        return -1;
    }
    if (session->u.cl.program == NULL || result != CL_SUCCESS) {
        error_code(session->info.error, XE_TEXT, "OpenCL program creation", result);
        return -1;
    }
    result = api->clBuildProgram(
        session->u.cl.program,
        1,
        &session->u.cl.device,
        build_options != NULL ? build_options : "",
        NULL,
        NULL);
    cl_build_log(session);
    if (result != CL_SUCCESS) {
        error_code(session->info.error, XE_TEXT, "clBuildProgram", result);
        return -1;
    }
    session->u.cl.kernel = api->clCreateKernel(session->u.cl.program, entry_point, &result);
    if (session->u.cl.kernel == NULL || result != CL_SUCCESS) {
        error_code(session->info.error, XE_TEXT, "clCreateKernel", result);
        return -1;
    }
    return 0;
}

static void ze_capture_build_log(struct xe_session *session, ze_module_build_log_handle_t log) {
    if (log == NULL) {
        return;
    }
    size_t length = 0;
    struct ze_api *api = &session->u.ze.api;
    if (api->zeModuleBuildLogGetString(log, &length, NULL) == ZE_RESULT_SUCCESS && length != 0) {
        char *temporary = calloc(length + 1, 1);
        if (temporary != NULL) {
            if (api->zeModuleBuildLogGetString(log, &length, temporary) == ZE_RESULT_SUCCESS) {
                text_copy(session->info.build_log, sizeof(session->info.build_log), temporary);
            }
            free(temporary);
        }
    }
    api->zeModuleBuildLogDestroy(log);
}

static int initialize_ze(
    struct xe_session *session,
    uint32_t expected_vendor,
    uint32_t expected_device,
    uint32_t artifact_kind,
    const uint8_t *artifact,
    size_t artifact_len,
    const char *build_options,
    const char *entry_point) {
    struct ze_api *api = &session->u.ze.api;
    if (load_ze(api, session->info.error) != 0 ||
        ze_select_device(session, expected_vendor, expected_device, session->info.error) != 0 ||
        ze_queue_ordinal(session, session->info.error) != 0) {
        return -1;
    }
    fill_ze_info(session);
    ze_context_desc_t context_desc = {
        .stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC,
        .pNext = NULL,
        .flags = 0,
    };
    ze_result_t result = api->zeContextCreate(session->u.ze.driver, &context_desc, &session->u.ze.context);
    if (result != ZE_RESULT_SUCCESS) {
        error_code(session->info.error, XE_TEXT, "zeContextCreate", result);
        return -1;
    }
    ze_command_queue_desc_t queue_desc = {
        .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        .pNext = NULL,
        .ordinal = session->u.ze.queue_ordinal,
        .index = 0,
        .flags = 0,
        .mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
        .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
    };
    if (session->immediate) {
        result = api->zeCommandListCreateImmediate(
            session->u.ze.context, session->u.ze.device, &queue_desc, &session->u.ze.list);
    } else {
        result = api->zeCommandQueueCreate(
            session->u.ze.context, session->u.ze.device, &queue_desc, &session->u.ze.queue);
        if (result == ZE_RESULT_SUCCESS) {
            ze_command_list_desc_t list_desc = {
                .stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
                .pNext = NULL,
                .commandQueueGroupOrdinal = session->u.ze.queue_ordinal,
                .flags = 0,
            };
            result = api->zeCommandListCreate(
                session->u.ze.context, session->u.ze.device, &list_desc, &session->u.ze.list);
        }
    }
    if (result != ZE_RESULT_SUCCESS) {
        error_code(session->info.error, XE_TEXT, "Level Zero queue/list creation", result);
        return -1;
    }

    ze_event_pool_desc_t pool_desc = {
        .stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
        .pNext = NULL,
        .flags = ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP | ZE_EVENT_POOL_FLAG_HOST_VISIBLE,
        .count = 1,
    };
    result = api->zeEventPoolCreate(
        session->u.ze.context, &pool_desc, 1, &session->u.ze.device, &session->u.ze.event_pool);
    if (result == ZE_RESULT_SUCCESS) {
        ze_event_desc_t event_desc = {
            .stype = ZE_STRUCTURE_TYPE_EVENT_DESC,
            .pNext = NULL,
            .index = 0,
            .signal = ZE_EVENT_SCOPE_FLAG_DEVICE,
            .wait = ZE_EVENT_SCOPE_FLAG_HOST,
        };
        result = api->zeEventCreate(session->u.ze.event_pool, &event_desc, &session->u.ze.event);
    }
    if (result != ZE_RESULT_SUCCESS) {
        session->u.ze.event = NULL;
        session->u.ze.event_pool = NULL;
    }

    if (artifact == NULL || artifact_len == 0) {
        return 0;
    }
    if (artifact_kind != XE_ARTIFACT_SPIRV && artifact_kind != XE_ARTIFACT_NATIVE) {
        text_copy(session->info.error, XE_TEXT, "unsupported artifact kind for Level Zero");
        return -1;
    }
    ze_module_desc_t module_desc = {
        .stype = ZE_STRUCTURE_TYPE_MODULE_DESC,
        .pNext = NULL,
        .format = artifact_kind == XE_ARTIFACT_NATIVE ? ZE_MODULE_FORMAT_NATIVE : ZE_MODULE_FORMAT_IL_SPIRV,
        .inputSize = artifact_len,
        .pInputModule = artifact,
        .pBuildFlags = build_options != NULL ? build_options : "",
        .pConstants = NULL,
    };
    ze_module_build_log_handle_t log = NULL;
    result = api->zeModuleCreate(
        session->u.ze.context,
        session->u.ze.device,
        &module_desc,
        &session->u.ze.module,
        &log);
    ze_capture_build_log(session, log);
    if (result != ZE_RESULT_SUCCESS) {
        error_code(session->info.error, XE_TEXT, "zeModuleCreate", result);
        return -1;
    }
    ze_kernel_desc_t kernel_desc = {
        .stype = ZE_STRUCTURE_TYPE_KERNEL_DESC,
        .pNext = NULL,
        .flags = 0,
        .pKernelName = entry_point,
    };
    result = api->zeKernelCreate(session->u.ze.module, &kernel_desc, &session->u.ze.kernel);
    if (result != ZE_RESULT_SUCCESS) {
        error_code(session->info.error, XE_TEXT, "zeKernelCreate", result);
        return -1;
    }
    return 0;
}

void *xe_session_create(
    uint32_t backend,
    uint32_t expected_vendor,
    uint32_t expected_device,
    uint32_t artifact_kind,
    const uint8_t *artifact,
    size_t artifact_len,
    const char *build_options,
    const char *entry_point,
    uint8_t immediate,
    struct xe_session_info *info) {
    if (info == NULL) {
        return NULL;
    }
    memset(info, 0, sizeof(*info));
    struct xe_session *session = calloc(1, sizeof(*session));
    if (session == NULL) {
        text_copy(info->error, sizeof(info->error), "out of memory creating Xe session");
        info->status = -1;
        return NULL;
    }
    session->backend = backend;
    session->immediate = immediate;
    session->info.backend = backend;
    session->info.immediate = immediate;
    uint64_t start = monotonic_ns();
    int result = -1;
    if (backend == XE_BACKEND_OPENCL) {
        result = initialize_cl(
            session,
            expected_vendor,
            expected_device,
            artifact_kind,
            artifact,
            artifact_len,
            build_options,
            entry_point);
    } else if (backend == XE_BACKEND_LEVEL_ZERO) {
        result = initialize_ze(
            session,
            expected_vendor,
            expected_device,
            artifact_kind,
            artifact,
            artifact_len,
            build_options,
            entry_point);
    } else {
        text_copy(session->info.error, XE_TEXT, "unknown Xe backend");
    }
    session->info.creation_ns = monotonic_ns() - start;
    session->info.status = result;
    *info = session->info;
    if (result != 0) {
        xe_session_destroy(session);
        return NULL;
    }
    return session;
}

int32_t xe_session_probe(
    uint32_t backend,
    uint32_t expected_vendor,
    uint32_t expected_device,
    uint8_t immediate,
    struct xe_session_info *info) {
    void *session = xe_session_create(
        backend,
        expected_vendor,
        expected_device,
        0,
        NULL,
        0,
        "",
        "",
        immediate,
        info);
    if (session == NULL) {
        return info != NULL ? info->status : -1;
    }
    xe_session_destroy(session);
    return 0;
}

int32_t xe_session_native_binary(
    void *opaque,
    uint8_t **bytes,
    size_t *length,
    char *error,
    size_t error_len) {
    struct xe_session *session = opaque;
    if (session == NULL || bytes == NULL || length == NULL) {
        text_copy(error, error_len, "invalid native-binary arguments");
        return -1;
    }
    *bytes = NULL;
    *length = 0;
    if (session->backend == XE_BACKEND_OPENCL) {
        struct cl_api *api = &session->u.cl.api;
        size_t binary_size = 0;
        cl_int result = api->clGetProgramInfo(
            session->u.cl.program,
            CL_PROGRAM_BINARY_SIZES,
            sizeof(binary_size),
            &binary_size,
            NULL);
        if (result != CL_SUCCESS || binary_size == 0) {
            error_code(error, error_len, "clGetProgramInfo(binary size)", result);
            return result;
        }
        uint8_t *binary = malloc(binary_size);
        if (binary == NULL) {
            text_copy(error, error_len, "out of memory retrieving OpenCL binary");
            return -1;
        }
        unsigned char *pointer = binary;
        result = api->clGetProgramInfo(
            session->u.cl.program,
            CL_PROGRAM_BINARIES,
            sizeof(pointer),
            &pointer,
            NULL);
        if (result != CL_SUCCESS) {
            free(binary);
            error_code(error, error_len, "clGetProgramInfo(binary)", result);
            return result;
        }
        *bytes = binary;
        *length = binary_size;
        return 0;
    }
    if (session->backend == XE_BACKEND_LEVEL_ZERO) {
        struct ze_api *api = &session->u.ze.api;
        size_t binary_size = 0;
        ze_result_t result = api->zeModuleGetNativeBinary(session->u.ze.module, &binary_size, NULL);
        if (result != ZE_RESULT_SUCCESS || binary_size == 0) {
            error_code(error, error_len, "zeModuleGetNativeBinary(size)", result);
            return result;
        }
        uint8_t *binary = malloc(binary_size);
        if (binary == NULL) {
            text_copy(error, error_len, "out of memory retrieving Level Zero binary");
            return -1;
        }
        result = api->zeModuleGetNativeBinary(session->u.ze.module, &binary_size, binary);
        if (result != ZE_RESULT_SUCCESS) {
            free(binary);
            error_code(error, error_len, "zeModuleGetNativeBinary", result);
            return result;
        }
        *bytes = binary;
        *length = binary_size;
        return 0;
    }
    text_copy(error, error_len, "unknown backend retrieving native binary");
    return -1;
}

void xe_bytes_free(uint8_t *bytes) {
    free(bytes);
}

int32_t xe_session_select_kernel(
    void *opaque,
    const char *entry_point,
    char *error,
    size_t error_len) {
    struct xe_session *session = opaque;
    if (session == NULL || entry_point == NULL || entry_point[0] == '\0') {
        text_copy(error, error_len, "invalid kernel selection arguments");
        return -1;
    }
    if (session->backend == XE_BACKEND_OPENCL) {
        cl_int result = CL_SUCCESS;
        cl_kernel next = session->u.cl.api.clCreateKernel(
            session->u.cl.program, entry_point, &result);
        if (next == NULL || result != CL_SUCCESS) {
            error_code(error, error_len, "clCreateKernel(select)", result);
            return result != CL_SUCCESS ? result : -1;
        }
        if (session->u.cl.kernel != NULL) {
            session->u.cl.api.clReleaseKernel(session->u.cl.kernel);
        }
        session->u.cl.kernel = next;
        return CL_SUCCESS;
    }
    if (session->backend == XE_BACKEND_LEVEL_ZERO) {
        ze_kernel_desc_t description = {
            .stype = ZE_STRUCTURE_TYPE_KERNEL_DESC,
            .pNext = NULL,
            .flags = 0,
            .pKernelName = entry_point,
        };
        ze_kernel_handle_t next = NULL;
        ze_result_t result = session->u.ze.api.zeKernelCreate(
            session->u.ze.module, &description, &next);
        if (result != ZE_RESULT_SUCCESS) {
            error_code(error, error_len, "zeKernelCreate(select)", result);
            return result;
        }
        if (session->u.ze.kernel != NULL) {
            session->u.ze.api.zeKernelDestroy(session->u.ze.kernel);
        }
        session->u.ze.kernel = next;
        return ZE_RESULT_SUCCESS;
    }
    text_copy(error, error_len, "unknown backend selecting kernel");
    return -1;
}

void *xe_buffer_create(
    void *opaque,
    uint32_t kind,
    size_t size,
    struct xe_run_timing *timing) {
    struct xe_session *session = opaque;
    if (timing == NULL) {
        return NULL;
    }
    memset(timing, 0, sizeof(*timing));
    if (session == NULL || size == 0) {
        text_copy(timing->error, XE_TEXT, "invalid buffer create arguments");
        timing->status = -1;
        return NULL;
    }
    struct xe_buffer *buffer = calloc(1, sizeof(*buffer));
    if (buffer == NULL) {
        text_copy(timing->error, XE_TEXT, "out of memory creating buffer handle");
        timing->status = -1;
        return NULL;
    }
    buffer->backend = session->backend;
    buffer->kind = kind;
    buffer->size = size;
    uint64_t start = monotonic_ns();
    if (session->backend == XE_BACKEND_OPENCL) {
        cl_mem_flags flags = CL_MEM_READ_WRITE;
        if (kind == XE_MEMORY_HOST || kind == XE_MEMORY_MAPPED || kind == XE_MEMORY_SHARED) {
            flags |= CL_MEM_ALLOC_HOST_PTR;
        }
        cl_int result = CL_SUCCESS;
        buffer->handle.cl = session->u.cl.api.clCreateBuffer(
            session->u.cl.context, flags, size, NULL, &result);
        timing->status = result;
        if (buffer->handle.cl == NULL || result != CL_SUCCESS) {
            error_code(timing->error, XE_TEXT, "clCreateBuffer", result);
        }
    } else if (session->backend == XE_BACKEND_LEVEL_ZERO) {
        ze_result_t result = ZE_RESULT_ERROR_INVALID_ARGUMENT;
        if (kind == XE_MEMORY_HOST) {
            ze_host_mem_alloc_desc_t description = {
                .stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                .pNext = NULL,
                .flags = 0,
            };
            result = session->u.ze.api.zeMemAllocHost(
                session->u.ze.context, &description, size, 64, &buffer->handle.ze);
        } else if (kind == XE_MEMORY_SHARED || kind == XE_MEMORY_MAPPED) {
            ze_device_mem_alloc_desc_t device_description = {
                .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                .pNext = NULL,
                .flags = 0,
                .ordinal = 0,
            };
            ze_host_mem_alloc_desc_t host_description = {
                .stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                .pNext = NULL,
                .flags = 0,
            };
            result = session->u.ze.api.zeMemAllocShared(
                session->u.ze.context,
                &device_description,
                &host_description,
                size,
                64,
                session->u.ze.device,
                &buffer->handle.ze);
        } else {
            ze_device_mem_alloc_desc_t description = {
                .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                .pNext = NULL,
                .flags = 0,
                .ordinal = 0,
            };
            result = session->u.ze.api.zeMemAllocDevice(
                session->u.ze.context,
                &description,
                size,
                64,
                session->u.ze.device,
                &buffer->handle.ze);
        }
        timing->status = result;
        if (result != ZE_RESULT_SUCCESS) {
            error_code(timing->error, XE_TEXT, "Level Zero allocation", result);
        }
    } else {
        timing->status = -1;
        text_copy(timing->error, XE_TEXT, "unknown backend creating buffer");
    }
    timing->host_ns = monotonic_ns() - start;
    if (timing->status != 0) {
        free(buffer);
        return NULL;
    }
    return buffer;
}

static ze_result_t ze_submit_list(struct xe_session *session, uint64_t timeout_ns) {
    struct ze_api *api = &session->u.ze.api;
    ze_result_t result;
    if (session->immediate) {
        return api->zeCommandListHostSynchronize(session->u.ze.list, timeout_ns);
    }
    result = api->zeCommandListClose(session->u.ze.list);
    if (result == ZE_RESULT_SUCCESS) {
        result = api->zeCommandQueueExecuteCommandLists(
            session->u.ze.queue, 1, &session->u.ze.list, NULL);
    }
    if (result == ZE_RESULT_SUCCESS) {
        result = api->zeCommandQueueSynchronize(session->u.ze.queue, timeout_ns);
    }
    ze_result_t reset = api->zeCommandListReset(session->u.ze.list);
    return result == ZE_RESULT_SUCCESS ? reset : result;
}

int32_t xe_buffer_write(
    void *opaque,
    void *opaque_buffer,
    const void *source,
    size_t size,
    struct xe_run_timing *timing) {
    struct xe_session *session = opaque;
    struct xe_buffer *buffer = opaque_buffer;
    if (timing == NULL) {
        return -1;
    }
    memset(timing, 0, sizeof(*timing));
    if (session == NULL || buffer == NULL || source == NULL || size > buffer->size) {
        text_copy(timing->error, XE_TEXT, "invalid buffer write arguments");
        timing->status = -1;
        return -1;
    }
    uint64_t start = monotonic_ns();
    if (session->backend == XE_BACKEND_OPENCL) {
        cl_int result = session->u.cl.api.clEnqueueWriteBuffer(
            session->u.cl.queue,
            buffer->handle.cl,
            CL_TRUE,
            0,
            size,
            source,
            0,
            NULL,
            NULL);
        timing->status = result;
        if (result != CL_SUCCESS) {
            error_code(timing->error, XE_TEXT, "clEnqueueWriteBuffer", result);
        }
    } else {
        ze_result_t result = session->u.ze.api.zeCommandListAppendMemoryCopy(
            session->u.ze.list, buffer->handle.ze, source, size, NULL, 0, NULL);
        if (result == ZE_RESULT_SUCCESS) {
            result = ze_submit_list(session, XE_TIMEOUT_FOREVER);
        }
        timing->status = result;
        if (result != ZE_RESULT_SUCCESS) {
            error_code(timing->error, XE_TEXT, "Level Zero buffer write", result);
        }
    }
    timing->host_ns = monotonic_ns() - start;
    return timing->status;
}

int32_t xe_buffer_read(
    void *opaque,
    void *opaque_buffer,
    void *destination,
    size_t size,
    struct xe_run_timing *timing) {
    struct xe_session *session = opaque;
    struct xe_buffer *buffer = opaque_buffer;
    if (timing == NULL) {
        return -1;
    }
    memset(timing, 0, sizeof(*timing));
    if (session == NULL || buffer == NULL || destination == NULL || size > buffer->size) {
        text_copy(timing->error, XE_TEXT, "invalid buffer read arguments");
        timing->status = -1;
        return -1;
    }
    uint64_t start = monotonic_ns();
    if (session->backend == XE_BACKEND_OPENCL) {
        cl_int result = session->u.cl.api.clEnqueueReadBuffer(
            session->u.cl.queue,
            buffer->handle.cl,
            CL_TRUE,
            0,
            size,
            destination,
            0,
            NULL,
            NULL);
        timing->status = result;
        if (result != CL_SUCCESS) {
            error_code(timing->error, XE_TEXT, "clEnqueueReadBuffer", result);
        }
    } else {
        ze_result_t result = session->u.ze.api.zeCommandListAppendMemoryCopy(
            session->u.ze.list, destination, buffer->handle.ze, size, NULL, 0, NULL);
        if (result == ZE_RESULT_SUCCESS) {
            result = ze_submit_list(session, XE_TIMEOUT_FOREVER);
        }
        timing->status = result;
        if (result != ZE_RESULT_SUCCESS) {
            error_code(timing->error, XE_TEXT, "Level Zero buffer read", result);
        }
    }
    timing->host_ns = monotonic_ns() - start;
    return timing->status;
}

int32_t xe_kernel_arg_buffer(
    void *opaque,
    uint32_t index,
    void *opaque_buffer,
    char *error,
    size_t error_len) {
    struct xe_session *session = opaque;
    struct xe_buffer *buffer = opaque_buffer;
    if (session == NULL || buffer == NULL || session->backend != buffer->backend) {
        text_copy(error, error_len, "invalid buffer kernel argument");
        return -1;
    }
    if (session->backend == XE_BACKEND_OPENCL) {
        cl_int result = session->u.cl.api.clSetKernelArg(
            session->u.cl.kernel, index, sizeof(buffer->handle.cl), &buffer->handle.cl);
        if (result != CL_SUCCESS) {
            error_code(error, error_len, "clSetKernelArg(buffer)", result);
        }
        return result;
    }
    ze_result_t result = session->u.ze.api.zeKernelSetArgumentValue(
        session->u.ze.kernel, index, sizeof(buffer->handle.ze), &buffer->handle.ze);
    if (result != ZE_RESULT_SUCCESS) {
        error_code(error, error_len, "zeKernelSetArgumentValue(buffer)", result);
    }
    return result;
}

int32_t xe_kernel_arg_scalar(
    void *opaque,
    uint32_t index,
    const void *value,
    size_t size,
    char *error,
    size_t error_len) {
    struct xe_session *session = opaque;
    if (session == NULL || value == NULL || size == 0) {
        text_copy(error, error_len, "invalid scalar kernel argument");
        return -1;
    }
    if (session->backend == XE_BACKEND_OPENCL) {
        cl_int result = session->u.cl.api.clSetKernelArg(session->u.cl.kernel, index, size, value);
        if (result != CL_SUCCESS) {
            error_code(error, error_len, "clSetKernelArg(scalar)", result);
        }
        return result;
    }
    ze_result_t result =
        session->u.ze.api.zeKernelSetArgumentValue(session->u.ze.kernel, index, size, value);
    if (result != ZE_RESULT_SUCCESS) {
        error_code(error, error_len, "zeKernelSetArgumentValue(scalar)", result);
    }
    return result;
}

int32_t xe_kernel_group_size(
    void *opaque,
    uint32_t x,
    uint32_t y,
    uint32_t z,
    char *error,
    size_t error_len) {
    struct xe_session *session = opaque;
    if (session == NULL || x == 0 || y == 0 || z == 0) {
        text_copy(error, error_len, "invalid kernel group size");
        return -1;
    }
    if (session->backend == XE_BACKEND_OPENCL) {
        return 0;
    }
    ze_result_t result = session->u.ze.api.zeKernelSetGroupSize(session->u.ze.kernel, x, y, z);
    if (result != ZE_RESULT_SUCCESS) {
        error_code(error, error_len, "zeKernelSetGroupSize", result);
    }
    return result;
}

int32_t xe_kernel_run(
    void *opaque,
    size_t global_x,
    size_t global_y,
    size_t global_z,
    size_t local_x,
    size_t local_y,
    size_t local_z,
    uint64_t timeout_ns,
    struct xe_run_timing *timing) {
    struct xe_session *session = opaque;
    if (timing == NULL) {
        return -1;
    }
    memset(timing, 0, sizeof(*timing));
    if (session == NULL || global_x == 0 || global_y == 0 || global_z == 0) {
        text_copy(timing->error, XE_TEXT, "invalid kernel launch dimensions");
        timing->status = -1;
        return -1;
    }
    uint64_t start = monotonic_ns();
    if (session->backend == XE_BACKEND_OPENCL) {
        size_t global[] = {global_x, global_y, global_z};
        size_t local[] = {local_x, local_y, local_z};
        cl_uint dimensions = global_z > 1 ? 3 : (global_y > 1 ? 2 : 1);
        cl_event event = NULL;
        uint64_t submit_start = monotonic_ns();
        cl_int result = session->u.cl.api.clEnqueueNDRangeKernel(
            session->u.cl.queue,
            session->u.cl.kernel,
            dimensions,
            NULL,
            global,
            local_x == 0 ? NULL : local,
            0,
            NULL,
            &event);
        timing->submit_ns = monotonic_ns() - submit_start;
        if (result == CL_SUCCESS) {
            uint64_t wait_start = monotonic_ns();
            result = session->u.cl.api.clFinish(session->u.cl.queue);
            timing->wait_ns = monotonic_ns() - wait_start;
        }
        timing->host_ns = monotonic_ns() - start;
        if (result == CL_SUCCESS && event != NULL) {
            cl_ulong device_start = 0;
            cl_ulong device_end = 0;
            if (session->u.cl.api.clGetEventProfilingInfo(
                    event,
                    CL_PROFILING_COMMAND_START,
                    sizeof(device_start),
                    &device_start,
                    NULL) == CL_SUCCESS &&
                session->u.cl.api.clGetEventProfilingInfo(
                    event,
                    CL_PROFILING_COMMAND_END,
                    sizeof(device_end),
                    &device_end,
                    NULL) == CL_SUCCESS &&
                device_end >= device_start) {
                timing->device_ns = device_end - device_start;
            }
        }
        if (event != NULL) {
            session->u.cl.api.clReleaseEvent(event);
        }
        timing->status = result;
        if (result != CL_SUCCESS) {
            error_code(timing->error, XE_TEXT, "OpenCL kernel launch", result);
        }
        return result;
    }

    struct ze_api *api = &session->u.ze.api;
    if (session->u.ze.event != NULL) {
        api->zeEventHostReset(session->u.ze.event);
    }
    ze_group_count_t groups = {
        .groupCountX = (uint32_t)((global_x + local_x - 1) / local_x),
        .groupCountY = (uint32_t)((global_y + local_y - 1) / local_y),
        .groupCountZ = (uint32_t)((global_z + local_z - 1) / local_z),
    };
    ze_result_t result = api->zeCommandListAppendLaunchKernel(
        session->u.ze.list, session->u.ze.kernel, &groups, session->u.ze.event, 0, NULL);
    timing->submit_ns = monotonic_ns() - start;
    if (result == ZE_RESULT_SUCCESS) {
        uint64_t wait_start = monotonic_ns();
        result = ze_submit_list(session, timeout_ns);
        timing->wait_ns = monotonic_ns() - wait_start;
    }
    timing->host_ns = monotonic_ns() - start;
    if (result == ZE_RESULT_SUCCESS && session->u.ze.event != NULL) {
        ze_kernel_timestamp_result_t timestamp;
        memset(&timestamp, 0, sizeof(timestamp));
        if (api->zeEventQueryKernelTimestamp(session->u.ze.event, &timestamp) == ZE_RESULT_SUCCESS) {
            uint64_t ticks = timestamp.global.kernelEnd - timestamp.global.kernelStart;
            timing->device_ns = ticks * session->timer_resolution;
        }
    }
    timing->status = result;
    if (result != ZE_RESULT_SUCCESS) {
        error_code(timing->error, XE_TEXT, "Level Zero kernel launch", result);
    }
    return result;
}

int32_t xe_buffer_destroy(
    void *opaque,
    void *opaque_buffer,
    struct xe_run_timing *timing) {
    struct xe_session *session = opaque;
    struct xe_buffer *buffer = opaque_buffer;
    if (timing == NULL) {
        return -1;
    }
    memset(timing, 0, sizeof(*timing));
    if (session == NULL || buffer == NULL) {
        text_copy(timing->error, XE_TEXT, "invalid buffer destroy arguments");
        timing->status = -1;
        return -1;
    }
    uint64_t start = monotonic_ns();
    if (session->backend == XE_BACKEND_OPENCL) {
        timing->status = session->u.cl.api.clReleaseMemObject(buffer->handle.cl);
        if (timing->status != CL_SUCCESS) {
            error_code(timing->error, XE_TEXT, "clReleaseMemObject", timing->status);
        }
    } else {
        timing->status = session->u.ze.api.zeMemFree(session->u.ze.context, buffer->handle.ze);
        if (timing->status != ZE_RESULT_SUCCESS) {
            error_code(timing->error, XE_TEXT, "zeMemFree", timing->status);
        }
    }
    timing->host_ns = monotonic_ns() - start;
    free(buffer);
    return timing->status;
}

void xe_session_destroy(void *opaque) {
    struct xe_session *session = opaque;
    if (session == NULL) {
        return;
    }
    if (session->backend == XE_BACKEND_OPENCL) {
        struct cl_api *api = &session->u.cl.api;
        if (session->u.cl.kernel != NULL) {
            api->clReleaseKernel(session->u.cl.kernel);
        }
        if (session->u.cl.program != NULL) {
            api->clReleaseProgram(session->u.cl.program);
        }
        if (session->u.cl.queue != NULL) {
            api->clReleaseCommandQueue(session->u.cl.queue);
        }
        if (session->u.cl.context != NULL) {
            api->clReleaseContext(session->u.cl.context);
        }
        close_cl(api);
    } else if (session->backend == XE_BACKEND_LEVEL_ZERO) {
        struct ze_api *api = &session->u.ze.api;
        if (session->u.ze.event != NULL) {
            api->zeEventDestroy(session->u.ze.event);
        }
        if (session->u.ze.event_pool != NULL) {
            api->zeEventPoolDestroy(session->u.ze.event_pool);
        }
        if (session->u.ze.kernel != NULL) {
            api->zeKernelDestroy(session->u.ze.kernel);
        }
        if (session->u.ze.module != NULL) {
            api->zeModuleDestroy(session->u.ze.module);
        }
        if (session->u.ze.list != NULL) {
            api->zeCommandListDestroy(session->u.ze.list);
        }
        if (session->u.ze.queue != NULL) {
            api->zeCommandQueueDestroy(session->u.ze.queue);
        }
        if (session->u.ze.context != NULL) {
            api->zeContextDestroy(session->u.ze.context);
        }
        close_ze(api);
    }
    free(session);
}

int32_t xe_memory_roundtrip(
    uint32_t backend,
    uint32_t expected_vendor,
    uint32_t expected_device,
    uint32_t kind,
    size_t size,
    uint8_t immediate,
    struct xe_memory_timing *timing) {
    if (timing == NULL) {
        return -1;
    }
    memset(timing, 0, sizeof(*timing));
    struct xe_session_info info;
    void *session = xe_session_create(
        backend,
        expected_vendor,
        expected_device,
        0,
        NULL,
        0,
        "",
        "",
        immediate,
        &info);
    if (session == NULL) {
        timing->status = info.status;
        text_copy(timing->error, XE_TEXT, info.error);
        return timing->status;
    }
    uint8_t *source = malloc(size);
    uint8_t *destination = malloc(size);
    if (source == NULL || destination == NULL) {
        timing->status = -1;
        text_copy(timing->error, XE_TEXT, "host allocation failed in memory roundtrip");
        free(source);
        free(destination);
        xe_session_destroy(session);
        return -1;
    }
    for (size_t index = 0; index < size; ++index) {
        source[index] = (uint8_t)(index * 131U + 17U);
    }
    if (backend == XE_BACKEND_OPENCL && kind == XE_MEMORY_SHARED) {
        struct xe_session *typed_session = session;
        struct cl_api *api = &typed_session->u.cl.api;
        uint64_t operation_start = monotonic_ns();
        void *svm = api->clSVMAlloc(typed_session->u.cl.context, CL_MEM_READ_WRITE, size, 64);
        timing->allocation_ns = monotonic_ns() - operation_start;
        if (svm == NULL) {
            timing->status = CL_MEM_OBJECT_ALLOCATION_FAILURE;
            text_copy(timing->error, XE_TEXT, "clSVMAlloc returned NULL");
            goto cleanup;
        }
        operation_start = monotonic_ns();
        cl_int cl_result = api->clEnqueueSVMMap(
            typed_session->u.cl.queue,
            CL_TRUE,
            CL_MAP_WRITE_INVALIDATE_REGION,
            svm,
            size,
            0,
            NULL,
            NULL);
        if (cl_result == CL_SUCCESS) {
            memcpy(svm, source, size);
            cl_result = api->clEnqueueSVMUnmap(
                typed_session->u.cl.queue, svm, 0, NULL, NULL);
        }
        if (cl_result == CL_SUCCESS) {
            cl_result = api->clFinish(typed_session->u.cl.queue);
        }
        timing->first_write_ns = monotonic_ns() - operation_start;
        operation_start = monotonic_ns();
        if (cl_result == CL_SUCCESS) {
            cl_result = api->clEnqueueSVMMap(
                typed_session->u.cl.queue, CL_TRUE, CL_MAP_READ, svm, size, 0, NULL, NULL);
        }
        if (cl_result == CL_SUCCESS) {
            memcpy(destination, svm, size);
            cl_result = api->clEnqueueSVMUnmap(
                typed_session->u.cl.queue, svm, 0, NULL, NULL);
        }
        if (cl_result == CL_SUCCESS) {
            cl_result = api->clFinish(typed_session->u.cl.queue);
        }
        timing->read_ns = monotonic_ns() - operation_start;
        if (cl_result == CL_SUCCESS && memcmp(source, destination, size) != 0) {
            cl_result = -2;
            text_copy(timing->error, XE_TEXT, "OpenCL SVM validation mismatch");
        }
        operation_start = monotonic_ns();
        if (cl_result == CL_SUCCESS) {
            cl_result = api->clEnqueueSVMMap(
                typed_session->u.cl.queue,
                CL_TRUE,
                CL_MAP_WRITE_INVALIDATE_REGION,
                svm,
                size,
                0,
                NULL,
                NULL);
        }
        if (cl_result == CL_SUCCESS) {
            memcpy(svm, source, size);
            cl_result = api->clEnqueueSVMUnmap(
                typed_session->u.cl.queue, svm, 0, NULL, NULL);
        }
        if (cl_result == CL_SUCCESS) {
            cl_result = api->clFinish(typed_session->u.cl.queue);
        }
        timing->reuse_write_ns = monotonic_ns() - operation_start;
        operation_start = monotonic_ns();
        api->clSVMFree(typed_session->u.cl.context, svm);
        timing->cleanup_ns = monotonic_ns() - operation_start;
        timing->status = cl_result;
        if (cl_result != CL_SUCCESS && timing->error[0] == '\0') {
            error_code(timing->error, XE_TEXT, "OpenCL SVM operation", cl_result);
        }
        goto cleanup;
    }
    struct xe_run_timing operation;
    void *buffer = xe_buffer_create(session, kind, size, &operation);
    timing->allocation_ns = operation.host_ns;
    if (buffer == NULL) {
        timing->status = operation.status;
        text_copy(timing->error, XE_TEXT, operation.error);
        goto cleanup;
    }
    if (backend == XE_BACKEND_OPENCL && kind == XE_MEMORY_MAPPED) {
        struct xe_session *typed_session = session;
        struct xe_buffer *typed_buffer = buffer;
        struct cl_api *api = &typed_session->u.cl.api;
        cl_int cl_result = CL_SUCCESS;
        uint64_t operation_start = monotonic_ns();
        void *mapped = api->clEnqueueMapBuffer(
            typed_session->u.cl.queue,
            typed_buffer->handle.cl,
            CL_TRUE,
            CL_MAP_WRITE_INVALIDATE_REGION,
            0,
            size,
            0,
            NULL,
            NULL,
            &cl_result);
        if (cl_result == CL_SUCCESS && mapped != NULL) {
            memcpy(mapped, source, size);
            cl_result = api->clEnqueueUnmapMemObject(
                typed_session->u.cl.queue, typed_buffer->handle.cl, mapped, 0, NULL, NULL);
        }
        if (cl_result == CL_SUCCESS) {
            cl_result = api->clFinish(typed_session->u.cl.queue);
        }
        timing->first_write_ns = monotonic_ns() - operation_start;
        operation_start = monotonic_ns();
        mapped = NULL;
        if (cl_result == CL_SUCCESS) {
            mapped = api->clEnqueueMapBuffer(
                typed_session->u.cl.queue,
                typed_buffer->handle.cl,
                CL_TRUE,
                CL_MAP_READ,
                0,
                size,
                0,
                NULL,
                NULL,
                &cl_result);
        }
        if (cl_result == CL_SUCCESS && mapped != NULL) {
            memcpy(destination, mapped, size);
            cl_result = api->clEnqueueUnmapMemObject(
                typed_session->u.cl.queue, typed_buffer->handle.cl, mapped, 0, NULL, NULL);
        }
        if (cl_result == CL_SUCCESS) {
            cl_result = api->clFinish(typed_session->u.cl.queue);
        }
        timing->read_ns = monotonic_ns() - operation_start;
        if (cl_result == CL_SUCCESS && memcmp(source, destination, size) != 0) {
            cl_result = -2;
            text_copy(timing->error, XE_TEXT, "OpenCL mapped-buffer validation mismatch");
        }
        operation_start = monotonic_ns();
        mapped = NULL;
        if (cl_result == CL_SUCCESS) {
            mapped = api->clEnqueueMapBuffer(
                typed_session->u.cl.queue,
                typed_buffer->handle.cl,
                CL_TRUE,
                CL_MAP_WRITE_INVALIDATE_REGION,
                0,
                size,
                0,
                NULL,
                NULL,
                &cl_result);
        }
        if (cl_result == CL_SUCCESS && mapped != NULL) {
            memcpy(mapped, source, size);
            cl_result = api->clEnqueueUnmapMemObject(
                typed_session->u.cl.queue, typed_buffer->handle.cl, mapped, 0, NULL, NULL);
        }
        if (cl_result == CL_SUCCESS) {
            cl_result = api->clFinish(typed_session->u.cl.queue);
        }
        timing->reuse_write_ns = monotonic_ns() - operation_start;
        timing->status = cl_result;
        if (cl_result != CL_SUCCESS && timing->error[0] == '\0') {
            error_code(timing->error, XE_TEXT, "OpenCL mapped-buffer operation", cl_result);
        }
        goto destroy_buffer;
    }
    xe_buffer_write(session, buffer, source, size, &operation);
    timing->first_write_ns = operation.host_ns;
    if (operation.status != 0) {
        timing->status = operation.status;
        text_copy(timing->error, XE_TEXT, operation.error);
        goto destroy_buffer;
    }
    memset(destination, 0, size);
    xe_buffer_read(session, buffer, destination, size, &operation);
    timing->read_ns = operation.host_ns;
    if (operation.status != 0 || memcmp(source, destination, size) != 0) {
        timing->status = operation.status != 0 ? operation.status : -2;
        text_copy(
            timing->error,
            XE_TEXT,
            operation.status != 0 ? operation.error : "memory roundtrip validation mismatch");
        goto destroy_buffer;
    }
    xe_buffer_write(session, buffer, source, size, &operation);
    timing->reuse_write_ns = operation.host_ns;
    timing->status = operation.status;
destroy_buffer:
    xe_buffer_destroy(session, buffer, &operation);
    timing->cleanup_ns = operation.host_ns;
cleanup:
    free(source);
    free(destination);
    xe_session_destroy(session);
    return timing->status;
}
