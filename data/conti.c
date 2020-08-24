
#include<stdio.h>
#include<windows.h>

int main(){
    FILE* file = fopen("test.wav", "rb");
    fseek(file, 0, SEEK_END);
    long fsize = ftell(file);
    rewind(file);
    unsigned char* fileData = malloc(fsize + 1);
    fread(fileData, 1, fsize, file);
    fclose(file);
    int map[256 * 256];
    for(int i = 0; i < 256 * 256; i++)
        map[i] = 0;
    for(int i = 0; i < fsize - 2; i++)
        map[255 * fileData[i] + fileData[i + 1]]++;
    HDC consoleDC = GetDC(GetConsoleWindow());
    for(int i = 0; i < 255 * 255; i++)
        if(map[i] != 0)
            SetPixelV(consoleDC, i / 255, i % 255, RGB(255,0,0));
    free(fileData);
    system("pause");
}