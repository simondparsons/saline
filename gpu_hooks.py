# Suggestion from ChatGPT for quicker computation of vegetative indices.

# Not yet implemented. Needs cupy, which itself needs CUDA.

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def computeIndexGPU(img, fn):
    """
    GPU version of computeIndex using CuPy.
    Falls back to CPU if CuPy is unavailable.
    """
    if not GPU_AVAILABLE:
        return computeIndex(img, fn)

    img_gpu = cp.asarray(img)

    b = img_gpu[:, :, 0]
    g = img_gpu[:, :, 1]
    r = img_gpu[:, :, 2]

    result_gpu = fn(b, g, r)

    return cp.asnumpy(result_gpu)

# Builds on the previous list of index names
def computeIndexByNameGPU(img, index_name):
    if index_name not in INDEX_FUNCTIONS:
        raise ValueError(f"Unknown index '{index_name}'")

    return computeIndexGPU(img, INDEX_FUNCTIONS[index_name])
