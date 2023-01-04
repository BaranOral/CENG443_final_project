final=$(pwd)
cd /usr/local/cuda/extras/demo_suite && ./deviceQuery
 
cd $final


echo "Executing Paper Approach"
nvcc -o paperOut paperBubbleSort.cu && ./paperOut | head -n 2  
echo "Executing Chunked Approach"
nvcc -o chunkedOut chunkedBubleSort.cu && ./chunkedOut | head -n 1 
echo "Executing Shared Memory"
nvcc -o sharedOut sharedBubleSort.cu && ./sharedOut | head -n 1


