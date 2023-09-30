#define UNICODE
#define _UNICODE
#include <windows.h>
#include <stdbool.h>
#include <stdint.h>
#include "slib/mandelbrot.cuh"

#define ID_COLOR_1 1
#define ID_COLOR_2 2
#define ID_COLOR_3 3

float rmax = 1.5f;
float rmin = -1.5f;
float imin = -1.5f * 9.0f / 16.0f;
float imax = 1.5f * 9.0f / 16.0f;

static int mouseX = 0;
static int mouseY = 0;

static int colorChoice = 0;

static bool quit = false;
double x_min = -2.0, x_max = 1.0, y_min = -1.5, y_max = 1.5;
double zoom_factor = 0.99f;

struct {
    int width;
    int height;
    uint32_t *pixels;
} frame = {0};

LRESULT CALLBACK WindowProcessMessage(HWND, UINT, WPARAM, LPARAM);
#if RAND_MAX == 32767
#define Rand32() ((rand() << 16) + (rand() << 1) + (rand() & 1))
#else
#define Rand32() rand()
#endif

static BITMAPINFO frame_bitmap_info;
static HBITMAP frame_bitmap = 0;
static HDC frame_device_context = 0;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR pCmdLine, int nCmdShow) {
    initializeCUDA();

    const wchar_t window_class_name[] = L"My Window Class";
    static WNDCLASS window_class = { 0 };
    window_class.lpfnWndProc = WindowProcessMessage;
    window_class.hInstance = hInstance;
    window_class.lpszClassName = window_class_name;
    RegisterClass(&window_class);

    frame_bitmap_info.bmiHeader.biSize = sizeof(frame_bitmap_info.bmiHeader);
    frame_bitmap_info.bmiHeader.biPlanes = 1;
    frame_bitmap_info.bmiHeader.biBitCount = 32;
    frame_bitmap_info.bmiHeader.biCompression = BI_RGB;
    frame_device_context = CreateCompatibleDC(0);

    static HWND window_handle;
    window_handle = CreateWindow(window_class_name, L"Drawing Pixels", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                                 640, 300, 640, 480, NULL, NULL, hInstance, NULL);
    if(window_handle == NULL) { return -1; }
    HMENU hMenu = CreateMenu();

    AppendMenu(hMenu, MF_STRING, ID_COLOR_1, L"Color 1");
    AppendMenu(hMenu, MF_STRING, ID_COLOR_2, L"Color 2");
    AppendMenu(hMenu, MF_STRING, ID_COLOR_3, L"Color 2");

    SetMenu(window_handle, hMenu);

    while(!quit) {
        static MSG message = { 0 };
        while(PeekMessage(&message, NULL, 0, 0, PM_REMOVE)) { DispatchMessage(&message); }

        int* pixels = malloc(sizeof(int) * frame.width * frame.height);
        computeMandelbrot(pixels, frame.width, frame.height);

        for (int i = 0; i < frame.width * frame.height; i++) {
            //frame.pixels[i] = defColor(pixels[i]);
            //frame.pixels[i] = graidiantColor(pixels[i]);
            //frame.pixels[i] = fullSpectrumColor(pixels[i]);
            frame.pixels[i] = color(colorChoice, pixels[i]);
        }

        // Call the CUDA function to compute Mandelbrot
        computeMandelbrotGPU(pixels, frame.width, frame.height);  // This function should be defined in your CUDA code.
        InvalidateRect(window_handle, NULL, FALSE);
        UpdateWindow(window_handle);
        free(pixels);
    }

    return 0;
}

LRESULT CALLBACK WindowProcessMessage(HWND window_handle, UINT message, WPARAM wParam, LPARAM lParam) {
    switch(message) {
        case WM_QUIT:
        case WM_DESTROY: {
            quit = true;
        } break;
        case WM_COMMAND: {
            switch (wParam) {
                case ID_COLOR_1:
                    colorChoice = 0;
                    InvalidateRect(window_handle, NULL, FALSE);
                    break;
                case ID_COLOR_2:
                    colorChoice = 1;
                    InvalidateRect(window_handle, NULL, FALSE);
                    break;
                case ID_COLOR_3:
                    colorChoice = 2;
                    InvalidateRect(window_handle, NULL, FALSE);
                    break;
            }
        } break;
        case WM_LBUTTONDOWN: {
            mouseX = LOWORD(lParam);
            mouseY = HIWORD(lParam);
            mouseY = frame.height - mouseY;
            
            // 3rd parameter is for frequency
            SetTimer(window_handle, 1, 0, NULL);

        } break;

        case WM_LBUTTONUP: {
            KillTimer(window_handle, 1);
        }

        case WM_TIMER: {
            double r_range = (rmax - rmin) * zoom_factor;
            double i_range = (imax - imin) * zoom_factor;

            // Calculate the ratio of clicked point to viewport edges
            double ratio_x = (double)(mouseX) / frame.width;
            double ratio_y = (double)(mouseY) / frame.height;

            double x = (double)mouseX / frame.width * (rmax - rmin) + rmin;
            double y = (double)mouseY / frame.height * (imax - imin) + imin;

            // Adjust the new ranges based on the ratios
            double new_rmin = x - r_range * ratio_x;
            double new_rmax = x + r_range * (1 - ratio_x);
            double new_imin = y - i_range * ratio_y;
            double new_imax = y + i_range * (1 - ratio_y);

            imax = new_imax;
            imin = new_imin;
            rmax = new_rmax;
            rmin = new_rmin;
            set_mandelbrot_range(new_rmin, new_rmax, new_imin, new_imax);

            InvalidateRect(window_handle, NULL, FALSE);
            UpdateWindow(window_handle);
        } break;

        case WM_PAINT: {
            static PAINTSTRUCT paint;
            static HDC device_context;
            device_context = BeginPaint(window_handle, &paint);
            BitBlt(device_context,
                   paint.rcPaint.left, paint.rcPaint.top,
                   paint.rcPaint.right - paint.rcPaint.left, paint.rcPaint.bottom - paint.rcPaint.top,
                   frame_device_context,
                   paint.rcPaint.left, paint.rcPaint.top,
                   SRCCOPY);
            EndPaint(window_handle, &paint);
        } break;

        case WM_SIZE: {
            frame_bitmap_info.bmiHeader.biWidth  = LOWORD(lParam);
            frame_bitmap_info.bmiHeader.biHeight = HIWORD(lParam);

            if(frame_bitmap) DeleteObject(frame_bitmap);
            frame_bitmap = CreateDIBSection(NULL, &frame_bitmap_info, DIB_RGB_COLORS, (void**)&frame.pixels, 0, 0);
            SelectObject(frame_device_context, frame_bitmap);

            frame.width =  LOWORD(lParam);
            frame.height = HIWORD(lParam);
        } break;

        default: {
            return DefWindowProc(window_handle, message, wParam, lParam);
        }
    }
    return 0;
}