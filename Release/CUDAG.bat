md CUDAResults
echo Gerstner
oceantest -size=128,128  -waves=10 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=CUDAResults\ger_CUDA_1_128.csv -time=60
oceantest -size=256,256  -waves=10 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=CUDAResults\ger_CUDA_1_256.csv -time=60
oceantest -size=512,512  -waves=10 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=CUDAResults\ger_CUDA_1_512.csv -time=60
oceantest -size=1024,1024  -waves=10 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=CUDAResults\ger_CUDA_1_1024.csv -time=60
oceantest -size=2048,2048  -waves=10 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=CUDAResults\ger_CUDA_1_2048.csv -time=60