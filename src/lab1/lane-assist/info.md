
## Notas

Es necesario pasarle los flags -g -G a nvcc para poder debuggear con cuda-gdb.

```
cuda-gdb ./transpose
break kernelName
run img0.png g
```

La función image_RGB2BW convierte una imagen en color (RGB) a una imagen en escala de grises (BW). En una imagen RGB, cada píxel se representa con tres componentes de color: rojo (R), verde (G) y azul (B). Por lo tanto, la imagen de entrada image_in tiene un tamaño de 3 * width * height bytes, donde cada píxel ocupa 3 bytes.