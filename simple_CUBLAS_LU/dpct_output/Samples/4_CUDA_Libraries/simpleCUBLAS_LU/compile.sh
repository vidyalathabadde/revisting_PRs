export PATH=/home/tcs2/vidya/open-src-clang/bin/:$PATH
export LD_LIBRARY_PATH=/home/tcs2/vidya/open-src-clang/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/tcs2/vidya/opensrcmkl/oneMKL/build/install/lib/:$LD_LIBRARY_PATH

clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda *.cpp -I../../../Common/ -I ../../../include/ -I /home/tcs2/vidya/opensrcmkl/oneMKL/build/install/include/ -L /home/tcs2/vidya/opensrcmkl/oneMKL/build/install/lib/ -lonemkl_blas_cublas -lonemkl

