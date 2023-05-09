nvcc -o packer.exe packer.cu
.\packer.exe input/wtpack1_0.txt
Remove-Item *.exe
Remove-Item *.lib
Remove-Item *.exp