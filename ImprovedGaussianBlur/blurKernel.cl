/*
 * a kernel that applies the gaussian blur to a bitmap
 */
__kernel void blur(__global uchar4 *inputPixels,
                   __global uchar4 *outputPixels,
                   __global double *cKernel,
                   __constant int *rows,
                   __constant int *cols,
                   __constant int *cKernelDimension,
                   __constant int *isHorizontal,
                   __local uchar4 *localInputRowPixels,
                   __local uchar4 *localInputColumnPixels) {

  int currentRow = get_global_id(0);
  int currentCol = get_global_id(1);
  double4 tempPixel = (double4)(0.0);

  int currentPixelIdx = currentRow * (*cols) + currentCol;

  // Write input pixel value into local memory
  //localInputPixels[currentPixelIdx] = inputPixels[currentPixelIdx];
  if(*isHorizontal == 1)
  {
    localInputRowPixels[currentCol] = inputPixels[currentPixelIdx];
  }
  else
  {
    localInputColumnPixels[currentRow] = inputPixels[currentPixelIdx];
  }

  // Device synchronization: Wait for all pixels to write to local memory
  barrier(CLK_LOCAL_MEM_FENCE);

  int cKernelX, cKernelY;

  cKernelY = (*cKernelDimension) / 2;

  // Horizonal call
  if (*isHorizontal == 1) {
    int y = currentRow;
    for (cKernelX = 0; cKernelX < (*cKernelDimension); cKernelX++) {
      int x = currentCol - *cKernelDimension / 2 + cKernelX;
      if (x < 0 || x >= *cols) {
        x = currentCol;
      }

      // Was multiplied by the cKernelDimension due to not using the full kernel
      // matrix for each pixel.
      tempPixel += convert_double4(localInputRowPixels[x]) *
                   (*cKernelDimension) *
                   cKernel[(cKernelY * (*cKernelDimension)) + cKernelX];
    }
  }
  // Vertical call
  else {
    int x = currentCol;

    for (cKernelY = 0; cKernelY < (*cKernelDimension); cKernelY++) {
      int y = currentRow - *cKernelDimension / 2 + cKernelY;

      if (y < 0 || y >= *rows) {
        y = currentRow;
      }

      tempPixel += convert_double4(localInputColumnPixels[y]) *
                   (*cKernelDimension) *
                   cKernel[(cKernelY * (*cKernelDimension)) + cKernelX];
    }
  }

  outputPixels[currentPixelIdx] =
      convert_uchar4_sat(tempPixel);
}