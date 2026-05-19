# Тесты

Директория с тестами для проверки функциональности.
OpenBLAS добавлена в виде бинарников из-за проблем с линковкой

## Запуск тестов

```bash
# Перейти в директорию с тестами
cd ./tests

# Запустить все тесты
make run
```


Директория gemm для сравнения производительности самописного gemm и gemm из OpenBLAS.
## Запуск
```
gcc -O3 -march=native -mtune=native -mfma -mavx2 -pthread main.c -I"../OpenBLAS-bin/include" -L"../OpenBLAS-bin/lib" -lopenblas -o gemm_bench.exe

./gemm_bench.exe
```
