/*
* a kernel that applies the gaussian blur to a bitmap
*/
__kernel void blur(
    __global uchar4 *inputPixels,
    __global uchar4 *outputPixels,
    __global double *cKernel,
    __constant int *rows,
    __constant int *cols,
    __constant int *cKernelDimension)
{
    int currentRow = get_global_id(0);
    int currentCol = get_global_id(1);
    double4 tempPixel = (double4)(0.0);

    int cKernelX, cKernelY;

    for(cKernelY = 0; cKernelY < (*cKernelDimension); cKernelY++)
    {
        int y = currentRow - *cKernelDimension / 2 + cKernelY;
        if(y < 0 || y >= *rows)
        {
            y = currentRow;
        }

        for(cKernelX = 0; cKernelX < (*cKernelDimension); cKernelX++)
        {
            int x = currentCol - *cKernelDimension / 2 + cKernelX;
            if(x < 0 || x >= *cols)
            {
                x = currentCol;
            }

            tempPixel +=  convert_double4( (inputPixels[((y * (*cols) + x))])) * cKernel[(cKernelY * (*cKernelDimension)) + cKernelX];
        }
    }

    // convert to uchar4 and adjust values
    outputPixels[currentRow * (*cols) + currentCol] = convert_uchar4_sat(tempPixel);
}