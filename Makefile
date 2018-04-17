NVCC = /usr/local/cuda/bin/nvcc

gpma_bfs_demo:
	$(NVCC) -I./ -O3 -std=c++11 -w -gencode arch=compute_61,code=sm_61 -odir "." -M -o "gpma_bfs_demo.d" "./gpma_bfs_demo.cu"
	$(NVCC) -I./ -O3 -std=c++11 -w --compile --relocatable-device-code=true -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61 -x cu -o "gpma_bfs_demo.o" "gpma_bfs_demo.cu"
	$(NVCC) --cudart static --relocatable-device-code=true -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61 -link -o "gpma_bfs_demo" ./gpma_bfs_demo.o

clean:
	rm ./gpma_bfs_demo.o ./gpma_bfs_demo.d gpma_bfs_demo
