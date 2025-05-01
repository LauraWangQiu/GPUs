# image_salt_pepper

> **Autora:** Yi (Laura) Wang Qiu

Se ha implementado en [kernels.cpp](./kernels.cpp) la versión de SYCL de la función `remove_noise_SYCL`.

Pasos para utilizar oneAPI con GPU de NVIDIA en Windows:

- Instalar [oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
- Abrir Intel oneAPI command prompt for Intel 64 for Visual Studio 2022

    ```cmd
    cd "C:\path\to\project"
    icpx mainWin.cpp kernels.cpp -fsycl -fsycl-targets=nvptx64-nvidia-cuda -O3 -o main.exe
    ```
