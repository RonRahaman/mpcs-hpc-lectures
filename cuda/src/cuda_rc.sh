CUDA_DIR=/usr/local/cuda-11.1
if [ -d $CUDA_DIR ]; then
  export PATH=$CUDA_DIR/bin${PATH:+:${PATH}}
  export LD_LIBRARY_PATH=$CUDA_DIR/lib64\
    ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi