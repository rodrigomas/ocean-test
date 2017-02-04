@echo off
echo Starting Test Sequency

echo CUDA Tests
md CUDAResults
echo Gerstner
oceantest -size=128,128  -waves=10 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=CUDAResults\ger_CUDA_1_128.csv -time=60
oceantest -size=256,256  -waves=10 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=CUDAResults\ger_CUDA_1_256.csv -time=60
oceantest -size=512,512  -waves=10 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=CUDAResults\ger_CUDA_1_512.csv -time=60
oceantest -size=1024,1024  -waves=10 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=CUDAResults\ger_CUDA_1_1024.csv -time=60
oceantest -size=2048,2048  -waves=10 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=CUDAResults\ger_CUDA_1_2048.csv -time=60

echo FFT
echo 1 Core =========================
oceantest -size=128,128  -mode=CUDA -heightmap=FFT -normalmap=FFT -output=CUDAResults\fft_CUDA_1_128.csv -time=60
oceantest -size=256,256  -mode=CUDA -heightmap=FFT -normalmap=FFT -output=CUDAResults\fft_CUDA_1_256.csv -time=60
oceantest -size=512,512  -mode=CUDA -heightmap=FFT -normalmap=FFT -output=CUDAResults\fft_CUDA_1_512.csv -time=60
oceantest -size=1024,1024  -mode=CUDA -heightmap=FFT -normalmap=FFT -output=CUDAResults\fft_CUDA_1_1024.csv -time=60
oceantest -size=2048,2048  -mode=CUDA -heightmap=FFT -normalmap=FFT -output=CUDAResults\fft_CUDA_1_2048.csv -time=60

echo Perlin
echo 1 Core =========================
oceantest -size=128,128  -mode=CUDA -heightmap=Perlin -normalmap=Perlin -output=CUDAResults\Perlin_CUDA_1_128.csv -time=60
oceantest -size=256,256  -mode=CUDA -heightmap=Perlin -normalmap=Perlin -output=CUDAResults\Perlin_CUDA_1_256.csv -time=60
oceantest -size=512,512  -mode=CUDA -heightmap=Perlin -normalmap=Perlin -output=CUDAResults\Perlin_CUDA_1_512.csv -time=60
oceantest -size=1024,1024  -mode=CUDA -heightmap=Perlin -normalmap=Perlin -output=CUDAResults\Perlin_CUDA_1_1024.csv -time=60
oceantest -size=2048,2048  -mode=CUDA -heightmap=Perlin -normalmap=Perlin -output=CUDAResults\Perlin_CUDA_1_2048.csv -time=60

echo GPGPU Tests
md GPGPUResults
echo Gerstner
oceantest -size=128,128  -waves=10 -mode=GPGPU -heightmap=Gerstner -normalmap=Gerstner -output=GPGPUResults\ger_GPGPU_1_128.csv -time=60
oceantest -size=256,256  -waves=10 -mode=GPGPU -heightmap=Gerstner -normalmap=Gerstner -output=GPGPUResults\ger_GPGPU_1_256.csv -time=60
oceantest -size=512,512  -waves=10 -mode=GPGPU -heightmap=Gerstner -normalmap=Gerstner -output=GPGPUResults\ger_GPGPU_1_512.csv -time=60
oceantest -size=1024,1024  -waves=10 -mode=GPGPU -heightmap=Gerstner -normalmap=Gerstner -output=GPGPUResults\ger_GPGPU_1_1024.csv -time=60
oceantest -size=2048,2048  -waves=10 -mode=GPGPU -heightmap=Gerstner -normalmap=Gerstner -output=GPGPUResults\ger_GPGPU_1_2048.csv -time=60

echo Perlin
echo 1 Core =========================
oceantest -size=128,128  -mode=GPGPU -heightmap=Perlin -normalmap=Perlin -output=GPGPUResults\Perlin_GPGPU_1_128.csv -time=60
oceantest -size=256,256  -mode=GPGPU -heightmap=Perlin -normalmap=Perlin -output=GPGPUResults\Perlin_GPGPU_1_256.csv -time=60
oceantest -size=512,512  -mode=GPGPU -heightmap=Perlin -normalmap=Perlin -output=GPGPUResults\Perlin_GPGPU_1_512.csv -time=60
oceantest -size=1024,1024  -mode=GPGPU -heightmap=Perlin -normalmap=Perlin -output=GPGPUResults\Perlin_GPGPU_1_1024.csv -time=60
oceantest -size=2048,2048  -mode=GPGPU -heightmap=Perlin -normalmap=Perlin -output=GPGPUResults\Perlin_GPGPU_1_2048.csv -time=60


echo CPU Tests
md CPUResults

echo Gerstner
echo 1 Core =========================
oceantest -size=128,128 -cores=1 -waves=10 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=CPUResults\ger_CPU_1_128.csv -time=60
oceantest -size=256,256 -cores=1 -waves=10 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=CPUResults\ger_CPU_1_256.csv -time=60
oceantest -size=512,512 -cores=1 -waves=10 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=CPUResults\ger_CPU_1_512.csv -time=60
oceantest -size=1024,1024 -cores=1 -waves=10 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=CPUResults\ger_CPU_1_1024.csv -time=60
oceantest -size=2048,2048 -cores=1 -waves=10 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=CPUResults\ger_CPU_1_2048.csv -time=60
echo 2 Core ==========================
oceantest -size=128,128 -cores=2 -waves=10 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=CPUResults\ger_CPU_2_128.csv -time=60
oceantest -size=256,256 -cores=2 -waves=10 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=CPUResults\ger_CPU_2_256.csv -time=60
oceantest -size=512,512 -cores=2 -waves=10 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=CPUResults\ger_CPU_2_512.csv -time=60
oceantest -size=1024,1024 -cores=2 -waves=10 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=CPUResults\ger_CPU_2_1024.csv -time=60
oceantest -size=2048,2048 -cores=2 -waves=10 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=CPUResults\ger_CPU_2_2048.csv -time=60
echo 4 core =========================
oceantest -size=128,128 -cores=4 -waves=10 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=CPUResults\ger_CPU_4_128.csv -time=60
oceantest -size=256,256 -cores=4 -waves=10 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=CPUResults\ger_CPU_4_256.csv -time=60
oceantest -size=512,512 -cores=4 -waves=10 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=CPUResults\ger_CPU_4_512.csv -time=60
oceantest -size=1024,1024 -cores=4 -waves=10 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=CPUResults\ger_CPU_4_1024.csv -time=60
oceantest -size=2048,2048 -cores=4 -waves=10 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=CPUResults\ger_CPU_4_2048.csv -time=60

echo FFT
echo 1 Core =========================
oceantest -size=128,128 -cores=1 -waves=10 -mode=CPU -heightmap=FFT -normalmap=FFT -output=CPUResults\fft_CPU_1_128.csv -time=60
oceantest -size=256,256 -cores=1 -waves=10 -mode=CPU -heightmap=FFT -normalmap=FFT -output=CPUResults\fft_CPU_1_256.csv -time=60
oceantest -size=512,512 -cores=1 -waves=10 -mode=CPU -heightmap=FFT -normalmap=FFT -output=CPUResults\fft_CPU_1_512.csv -time=60
oceantest -size=1024,1024 -cores=1 -waves=10 -mode=CPU -heightmap=FFT -normalmap=FFT -output=CPUResults\fft_CPU_1_1024.csv -time=60
oceantest -size=2048,2048 -cores=1 -waves=10 -mode=CPU -heightmap=FFT -normalmap=FFT -output=CPUResults\fft_CPU_1_2048.csv -time=60
echo 2 Core ==========================
oceantest -size=128,128 -cores=2 -waves=10 -mode=CPU -heightmap=FFT -normalmap=FFT -output=CPUResults\fft_CPU_2_128.csv -time=60
oceantest -size=256,256 -cores=2 -waves=10 -mode=CPU -heightmap=FFT -normalmap=FFT -output=CPUResults\fft_CPU_2_256.csv -time=60
oceantest -size=512,512 -cores=2 -waves=10 -mode=CPU -heightmap=FFT -normalmap=FFT -output=CPUResults\fft_CPU_2_512.csv -time=60
oceantest -size=1024,1024 -cores=2 -waves=10 -mode=CPU -heightmap=FFT -normalmap=FFT -output=CPUResults\fft_CPU_2_1024.csv -time=60
oceantest -size=2048,2048 -cores=2 -waves=10 -mode=CPU -heightmap=FFT -normalmap=FFT -output=CPUResults\fft_CPU_2_2048.csv -time=60
echo 4 core =========================
oceantest -size=128,128 -cores=4 -waves=10 -mode=CPU -heightmap=FFT -normalmap=FFT -output=CPUResults\fft_CPU_4_128.csv -time=60
oceantest -size=256,256 -cores=4 -waves=10 -mode=CPU -heightmap=FFT -normalmap=FFT -output=CPUResults\fft_CPU_4_256.csv -time=60
oceantest -size=512,512 -cores=4 -waves=10 -mode=CPU -heightmap=FFT -normalmap=FFT -output=CPUResults\fft_CPU_4_512.csv -time=60
oceantest -size=1024,1024 -cores=4 -waves=10 -mode=CPU -heightmap=FFT -normalmap=FFT -output=CPUResults\fft_CPU_4_1024.csv -time=60
oceantest -size=2048,2048 -cores=4 -waves=10 -mode=CPU -heightmap=FFT -normalmap=FFT -output=CPUResults\fft_CPU_4_2048.csv -time=60

echo Perlin
echo 1 Core =========================
oceantest -size=128,128 -cores=1 -waves=10 -mode=CPU -heightmap=Perlin -normalmap=Perlin -output=CPUResults\Perlin_CPU_1_128.csv -time=60
oceantest -size=256,256 -cores=1 -waves=10 -mode=CPU -heightmap=Perlin -normalmap=Perlin -output=CPUResults\Perlin_CPU_1_256.csv -time=60
oceantest -size=512,512 -cores=1 -waves=10 -mode=CPU -heightmap=Perlin -normalmap=Perlin -output=CPUResults\Perlin_CPU_1_512.csv -time=60
oceantest -size=1024,1024 -cores=1 -waves=10 -mode=CPU -heightmap=Perlin -normalmap=Perlin -output=CPUResults\Perlin_CPU_1_1024.csv -time=60
oceantest -size=2048,2048 -cores=1 -waves=10 -mode=CPU -heightmap=Perlin -normalmap=Perlin -output=CPUResults\Perlin_CPU_1_2048.csv -time=60
echo 2 Core ==========================
oceantest -size=128,128 -cores=2 -waves=10 -mode=CPU -heightmap=Perlin -normalmap=Perlin -output=CPUResults\Perlin_CPU_2_128.csv -time=60
oceantest -size=256,256 -cores=2 -waves=10 -mode=CPU -heightmap=Perlin -normalmap=Perlin -output=CPUResults\Perlin_CPU_2_256.csv -time=60
oceantest -size=512,512 -cores=2 -waves=10 -mode=CPU -heightmap=Perlin -normalmap=Perlin -output=CPUResults\Perlin_CPU_2_512.csv -time=60
oceantest -size=1024,1024 -cores=2 -waves=10 -mode=CPU -heightmap=Perlin -normalmap=Perlin -output=CPUResults\Perlin_CPU_2_1024.csv -time=60
oceantest -size=2048,2048 -cores=2 -waves=10 -mode=CPU -heightmap=Perlin -normalmap=Perlin -output=CPUResults\Perlin_CPU_2_2048.csv -time=60
echo 4 core =========================
oceantest -size=128,128 -cores=4 -waves=10 -mode=CPU -heightmap=Perlin -normalmap=Perlin -output=CPUResults\Perlin_CPU_4_128.csv -time=60
oceantest -size=256,256 -cores=4 -waves=10 -mode=CPU -heightmap=Perlin -normalmap=Perlin -output=CPUResults\Perlin_CPU_4_256.csv -time=60
oceantest -size=512,512 -cores=4 -waves=10 -mode=CPU -heightmap=Perlin -normalmap=Perlin -output=CPUResults\Perlin_CPU_4_512.csv -time=60
oceantest -size=1024,1024 -cores=4 -waves=10 -mode=CPU -heightmap=Perlin -normalmap=Perlin -output=CPUResults\Perlin_CPU_4_1024.csv -time=60
oceantest -size=2048,2048 -cores=4 -waves=10 -mode=CPU -heightmap=Perlin -normalmap=Perlin -output=CPUResults\Perlin_CPU_4_2048.csv -time=60


echo Gerstner Tests
md GerstnerResults
echo 1 Core =========================
oceantest -size=256,256 -cores=1 -waves=1 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_1_256_1.csv -time=60
oceantest -size=256,256 -cores=1 -waves=2 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_1_256_2.csv -time=60
oceantest -size=256,256 -cores=1 -waves=3 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_1_256_3.csv -time=60
oceantest -size=256,256 -cores=1 -waves=4 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_1_256_4.csv -time=60
oceantest -size=256,256 -cores=1 -waves=5 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_1_256_5.csv -time=60
oceantest -size=256,256 -cores=1 -waves=6 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_1_256_6.csv -time=60
oceantest -size=256,256 -cores=1 -waves=7 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_1_256_7.csv -time=60
oceantest -size=256,256 -cores=1 -waves=8 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_1_256_8.csv -time=60
oceantest -size=256,256 -cores=1 -waves=9 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_1_256_9.csv -time=60
oceantest -size=256,256 -cores=1 -waves=10 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_1_256_10.csv -time=60
oceantest -size=256,256 -cores=1 -waves=11 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_1_256_11.csv -time=60
oceantest -size=256,256 -cores=1 -waves=12 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_1_256_12.csv -time=60
oceantest -size=256,256 -cores=1 -waves=13 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_1_256_13.csv -time=60
oceantest -size=256,256 -cores=1 -waves=14 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_1_256_14.csv -time=60
oceantest -size=256,256 -cores=1 -waves=15 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_1_256_15.csv -time=60
oceantest -size=256,256 -cores=1 -waves=16 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_1_256_16.csv -time=60
oceantest -size=256,256 -cores=1 -waves=17 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_1_256_17.csv -time=60
oceantest -size=256,256 -cores=1 -waves=18 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_1_256_18.csv -time=60
oceantest -size=256,256 -cores=1 -waves=19 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_1_256_19.csv -time=60
oceantest -size=256,256 -cores=1 -waves=20 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_1_256_20.csv -time=60


echo 2 Core ==========================
oceantest -size=256,256 -cores=2 -waves=1 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_2_256_1.csv -time=60
oceantest -size=256,256 -cores=2 -waves=2 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_2_256_2.csv -time=60
oceantest -size=256,256 -cores=2 -waves=3 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_2_256_3.csv -time=60
oceantest -size=256,256 -cores=2 -waves=4 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_2_256_4.csv -time=60
oceantest -size=256,256 -cores=2 -waves=5 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_2_256_5.csv -time=60
oceantest -size=256,256 -cores=2 -waves=6 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_2_256_6.csv -time=60
oceantest -size=256,256 -cores=2 -waves=7 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_2_256_7.csv -time=60
oceantest -size=256,256 -cores=2 -waves=8 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_2_256_8.csv -time=60
oceantest -size=256,256 -cores=2 -waves=9 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_2_256_9.csv -time=60
oceantest -size=256,256 -cores=2 -waves=10 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_2_256_10.csv -time=60
oceantest -size=256,256 -cores=2 -waves=11 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_2_256_11.csv -time=60
oceantest -size=256,256 -cores=2 -waves=12 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_2_256_12.csv -time=60
oceantest -size=256,256 -cores=2 -waves=13 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_2_256_13.csv -time=60
oceantest -size=256,256 -cores=2 -waves=14 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_2_256_14.csv -time=60
oceantest -size=256,256 -cores=2 -waves=15 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_2_256_15.csv -time=60
oceantest -size=256,256 -cores=2 -waves=16 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_2_256_16.csv -time=60
oceantest -size=256,256 -cores=2 -waves=17 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_2_256_17.csv -time=60
oceantest -size=256,256 -cores=2 -waves=18 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_2_256_18.csv -time=60
oceantest -size=256,256 -cores=2 -waves=19 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_2_256_19.csv -time=60
oceantest -size=256,256 -cores=2 -waves=20 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_2_256_20.csv -time=60

echo 4 core =========================
oceantest -size=256,256 -cores=4 -waves=1 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_4_256_1.csv -time=60
oceantest -size=256,256 -cores=4 -waves=2 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_4_256_2.csv -time=60
oceantest -size=256,256 -cores=4 -waves=3 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_4_256_3.csv -time=60
oceantest -size=256,256 -cores=4 -waves=4 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_4_256_4.csv -time=60
oceantest -size=256,256 -cores=4 -waves=5 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_4_256_5.csv -time=60
oceantest -size=256,256 -cores=4 -waves=6 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_4_256_6.csv -time=60
oceantest -size=256,256 -cores=4 -waves=7 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_4_256_7.csv -time=60
oceantest -size=256,256 -cores=4 -waves=8 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_4_256_8.csv -time=60
oceantest -size=256,256 -cores=4 -waves=9 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_4_256_9.csv -time=60
oceantest -size=256,256 -cores=4 -waves=10 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_4_256_10.csv -time=60
oceantest -size=256,256 -cores=4 -waves=11 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_4_256_11.csv -time=60
oceantest -size=256,256 -cores=4 -waves=12 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_4_256_12.csv -time=60
oceantest -size=256,256 -cores=4 -waves=13 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_4_256_13.csv -time=60
oceantest -size=256,256 -cores=4 -waves=14 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_4_256_14.csv -time=60
oceantest -size=256,256 -cores=4 -waves=15 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_4_256_15.csv -time=60
oceantest -size=256,256 -cores=4 -waves=16 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_4_256_16.csv -time=60
oceantest -size=256,256 -cores=4 -waves=17 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_4_256_17.csv -time=60
oceantest -size=256,256 -cores=4 -waves=18 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_4_256_18.csv -time=60
oceantest -size=256,256 -cores=4 -waves=19 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_4_256_19.csv -time=60
oceantest -size=256,256 -cores=4 -waves=20 -mode=CPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CPU_4_256_20.csv -time=60


echo CUDA =========================
oceantest -size=256,256 -cores=4 -waves=1 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CUDA_4_256_1.csv -time=60
oceantest -size=256,256 -cores=4 -waves=2 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CUDA_4_256_2.csv -time=60
oceantest -size=256,256 -cores=4 -waves=3 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CUDA_4_256_3.csv -time=60
oceantest -size=256,256 -cores=4 -waves=4 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CUDA_4_256_4.csv -time=60
oceantest -size=256,256 -cores=4 -waves=5 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CUDA_4_256_5.csv -time=60
oceantest -size=256,256 -cores=4 -waves=6 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CUDA_4_256_6.csv -time=60
oceantest -size=256,256 -cores=4 -waves=7 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CUDA_4_256_7.csv -time=60
oceantest -size=256,256 -cores=4 -waves=8 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CUDA_4_256_8.csv -time=60
oceantest -size=256,256 -cores=4 -waves=9 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CUDA_4_256_9.csv -time=60
oceantest -size=256,256 -cores=4 -waves=10 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CUDA_4_256_10.csv -time=60
oceantest -size=256,256 -cores=4 -waves=11 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CUDA_4_256_11.csv -time=60
oceantest -size=256,256 -cores=4 -waves=12 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CUDA_4_256_12.csv -time=60
oceantest -size=256,256 -cores=4 -waves=13 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CUDA_4_256_13.csv -time=60
oceantest -size=256,256 -cores=4 -waves=14 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CUDA_4_256_14.csv -time=60
oceantest -size=256,256 -cores=4 -waves=15 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CUDA_4_256_15.csv -time=60
oceantest -size=256,256 -cores=4 -waves=16 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CUDA_4_256_16.csv -time=60
oceantest -size=256,256 -cores=4 -waves=17 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CUDA_4_256_17.csv -time=60
oceantest -size=256,256 -cores=4 -waves=18 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CUDA_4_256_18.csv -time=60
oceantest -size=256,256 -cores=4 -waves=19 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CUDA_4_256_19.csv -time=60
oceantest -size=256,256 -cores=4 -waves=20 -mode=CUDA -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_CUDA_4_256_20.csv -time=60


echo GPUPGU =========================
oceantest -size=256,256 -cores=4 -waves=1 -mode=GPGPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_GPGPU_4_256_1.csv -time=60
oceantest -size=256,256 -cores=4 -waves=2 -mode=GPGPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_GPGPU_4_256_2.csv -time=60
oceantest -size=256,256 -cores=4 -waves=3 -mode=GPGPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_GPGPU_4_256_3.csv -time=60
oceantest -size=256,256 -cores=4 -waves=4 -mode=GPGPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_GPGPU_4_256_4.csv -time=60
oceantest -size=256,256 -cores=4 -waves=5 -mode=GPGPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_GPGPU_4_256_5.csv -time=60
oceantest -size=256,256 -cores=4 -waves=6 -mode=GPGPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_GPGPU_4_256_6.csv -time=60
oceantest -size=256,256 -cores=4 -waves=7 -mode=GPGPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_GPGPU_4_256_7.csv -time=60
oceantest -size=256,256 -cores=4 -waves=8 -mode=GPGPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_GPGPU_4_256_8.csv -time=60
oceantest -size=256,256 -cores=4 -waves=9 -mode=GPGPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_GPGPU_4_256_9.csv -time=60
oceantest -size=256,256 -cores=4 -waves=10 -mode=GPGPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_GPGPU_4_256_10.csv -time=60
oceantest -size=256,256 -cores=4 -waves=11 -mode=GPGPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_GPGPU_4_256_11.csv -time=60
oceantest -size=256,256 -cores=4 -waves=12 -mode=GPGPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_GPGPU_4_256_12.csv -time=60
oceantest -size=256,256 -cores=4 -waves=13 -mode=GPGPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_GPGPU_4_256_13.csv -time=60
oceantest -size=256,256 -cores=4 -waves=14 -mode=GPGPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_GPGPU_4_256_14.csv -time=60
oceantest -size=256,256 -cores=4 -waves=15 -mode=GPGPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_GPGPU_4_256_15.csv -time=60
oceantest -size=256,256 -cores=4 -waves=16 -mode=GPGPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_GPGPU_4_256_16.csv -time=60
oceantest -size=256,256 -cores=4 -waves=17 -mode=GPGPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_GPGPU_4_256_17.csv -time=60
oceantest -size=256,256 -cores=4 -waves=18 -mode=GPGPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_GPGPU_4_256_18.csv -time=60
oceantest -size=256,256 -cores=4 -waves=19 -mode=GPGPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_GPGPU_4_256_19.csv -time=60
oceantest -size=256,256 -cores=4 -waves=20 -mode=GPGPU -heightmap=Gerstner -normalmap=Gerstner -output=GerstnerResults\ger_GPGPU_4_256_20.csv -time=60