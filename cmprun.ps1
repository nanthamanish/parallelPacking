nvcc -o packer.exe packing.cu
.\packer.exe wtpack1_0.txt
Remove-Item *.exe
Remove-Item *.lib
Remove-Item *.exp