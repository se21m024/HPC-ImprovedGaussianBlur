/*
 * a kernel that applies the gaussian blur to a bitmap
 */
__kernel void blur(__global uchar4 *inputPixels, __global uchar4 *outputPixels,
                   __global double *cKernel, __constant int *rows,
                   __constant int *cols, __constant int *cKernelDimension,
                   __constant int *isHorizontal) {

  int currentRow = get_global_id(0);
  int currentCol = get_global_id(1);
  double4 tempPixel = (double4)(0.0);

  if (get_global_id(0) == 0 && get_global_id(1) == 3) {
    printf("currentRow: %u \n", currentRow);
    printf("currentCol: %u \n", currentCol);
    printf("isHorizontal: %u \n", *isHorizontal);
    // printf("value is: %u", *cKernelDimension);
  }

  int cKernelX, cKernelY;

  cKernelY = (*cKernelDimension) / 2;

  if (*isHorizontal == 1) {
    int y = currentRow;
    for (cKernelX = 0; cKernelX < (*cKernelDimension); cKernelX++) {
      int x = currentCol - *cKernelDimension / 2 + cKernelX;
      if (x < 0 || x >= *cols) {
        x = currentCol;
      }

      // Was multiplied by the cKernelDimension due to not using the full kernel
      // matrix for each pixel.
      tempPixel += convert_double4((inputPixels[((y * (*cols) + x))])) *
                   (*cKernelDimension) *
                   cKernel[(cKernelY * (*cKernelDimension)) + cKernelX];
    }
  } else {
    int x = currentCol;

    for (cKernelY = 0; cKernelY < (*cKernelDimension); cKernelY++) {
      int y = currentRow - *cKernelDimension / 2 + cKernelY;

      if (y < 0 || y >= *rows) {
        y = currentRow;
      }

      tempPixel += convert_double4((inputPixels[((y * (*cols) + x))])) *
                   (*cKernelDimension) *
                   cKernel[(cKernelY * (*cKernelDimension)) + cKernelX];
    }
  }

  outputPixels[currentRow * (*cols) + currentCol] =
      convert_uchar4_sat(tempPixel);
}