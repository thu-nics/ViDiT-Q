export TORCH_CUDA_FLAGS="-U__CUDA_NO_BFLOAT16_CONVERSIONS__"

rm -rf build/ *.egg-info
find . -name "*.so" -delete
pip uninstall -y viditq_extension
pip install -e . --no-build-isolation --no-cache-dir