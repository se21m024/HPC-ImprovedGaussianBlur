using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using OpenCL.Net;

namespace ImprovedGaussianBlur
{
    public class GaussianOpenCl
    {
        public static Bitmap ApplyGaussianBlur(Bitmap inputImage)
        {
            const int kernelDimension = 7;
            var kernel = CreateKernel(kernelDimension, kernelDimension);
            return Convolve(inputImage, kernel, kernelDimension);
        }

        // The CreateKernel method is based on: https://epochabuse.com/gaussian-blur/
        private static double[,] CreateKernel(int length, double weight)
        {
            var kernel = new double[length, length];
            double kernelSum = 0;
            var foff = (length - 1) / 2;
            var constant = 1d / (2 * Math.PI * weight * weight);

            for (var y = -foff; y <= foff; y++)
            {
                for (var x = -foff; x <= foff; x++)
                {
                    var distance = ((y * y) + (x * x)) / (2 * weight * weight);
                    kernel[y + foff, x + foff] = constant * Math.Exp(-distance);
                    kernelSum += kernel[y + foff, x + foff];
                }
            }

            for (var y = 0; y < length; y++)
            {
                for (var x = 0; x < length; x++)
                {
                    kernel[y, x] = kernel[y, x] * 1d / kernelSum;
                }
            }

            return kernel;
        }

        private static Bitmap Convolve(Bitmap inputImage, double[,] kernelMatrix, int kernelDim)
        {
            // convert bitmap to byte array
            int width = inputImage.Width;
            int height = inputImage.Height;
            BitmapData srcData = inputImage.LockBits(new Rectangle(0, 0, width, height),
                ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);

            int bytes = srcData.Stride * srcData.Height;
            byte[] inputImageArray = new byte[bytes];

            Marshal.Copy(srcData.Scan0, inputImageArray, 0, bytes);
            inputImage.UnlockBits(srcData);

            // Flatten kernel array
            var flattenedKernel = kernelMatrix.Cast<double>().ToArray();

            // input and output data
            var imageElementSize = bytes;
            var imageDataSize = imageElementSize * sizeof(byte);

            var inputArray = inputImageArray;
            var outputArray = new byte[imageElementSize];

            var kernelElementSize = flattenedKernel.Length;
            var kernelDataSize = kernelElementSize * sizeof(double);

            var kernelArray = flattenedKernel;

            var rows = inputImage.Height;
            var cols = inputImage.Width;
            var intDataSize = sizeof(int);

            // used for checking error status of api calls
            ErrorCode status;

            // retrieve the number of platforms
            uint numPlatforms = 0;
            CheckStatus(Cl.GetPlatformIDs(0, null, out numPlatforms));

            if (numPlatforms == 0)
            {
                Console.WriteLine("Error: No OpenCL platform available!");
                Console.ReadLine();
                System.Environment.Exit(1);
            }

            // select the platform
            Platform[] platforms = new Platform[numPlatforms];
            CheckStatus(Cl.GetPlatformIDs(1, platforms, out numPlatforms));
            Platform platform = platforms[0];

            // retrieve the number of devices
            uint numDevices = 0;
            CheckStatus(Cl.GetDeviceIDs(platform, DeviceType.All, 0, null, out numDevices));

            if (numDevices == 0)
            {
                Console.WriteLine("Error: No OpenCL device available for platform!");
                Console.ReadLine();
                System.Environment.Exit(1);
            }

            // select the device
            var devices = new Device[numDevices];
            CheckStatus(Cl.GetDeviceIDs(platform, DeviceType.All, numDevices, devices, out numDevices));
            var device = devices[0];

            // create context
            var context = Cl.CreateContext(null, 1, new Device[] { device }, null, IntPtr.Zero, out status);
            CheckStatus(status);

            // create command queue
            var commandQueue = Cl.CreateCommandQueue(context, device, 0, out status);
            CheckStatus(status);

            // allocate pixels input buffer
            var bufferInputImage = Cl.CreateBuffer<int>(context, MemFlags.ReadOnly, imageDataSize, out status);
            CheckStatus(status);

            // allocate kernel matrix input buffer
            var bufferKernel = Cl.CreateBuffer<double>(context, MemFlags.ReadOnly, kernelDataSize, out status);
            CheckStatus(status);

            // allocate pixels output buffer
            var bufferOutputImage = Cl.CreateBuffer<int>(context, MemFlags.WriteOnly, imageDataSize, out status);
            CheckStatus(status);

            // allocate scalar input buffers
            var bufferRows = Cl.CreateBuffer<int>(context, MemFlags.ReadOnly, 1, out status);
            CheckStatus(status);
            var bufferCols = Cl.CreateBuffer<int>(context, MemFlags.ReadOnly, 1, out status);
            CheckStatus(status);
            var bufferKernelDim = Cl.CreateBuffer<int>(context, MemFlags.ReadOnly, 1, out status);
            CheckStatus(status);
            var bufferIsHorizontal = Cl.CreateBuffer<int>(context, MemFlags.ReadOnly, 1, out status);
            CheckStatus(status);

            var isHorizontal = 1;

            // write data from the input arrays to the buffers
            CheckStatus(Cl.EnqueueWriteBuffer(commandQueue, bufferInputImage, Bool.True, IntPtr.Zero, new IntPtr(imageDataSize), inputArray, 0, null, out _));
            CheckStatus(Cl.EnqueueWriteBuffer(commandQueue, bufferKernel, Bool.True, IntPtr.Zero, new IntPtr(kernelDataSize), kernelArray, 0, null, out _));
            CheckStatus(Cl.EnqueueWriteBuffer(commandQueue, bufferRows, Bool.True, IntPtr.Zero, new IntPtr(intDataSize), rows, 0, null, out _));
            CheckStatus(Cl.EnqueueWriteBuffer(commandQueue, bufferCols, Bool.True, IntPtr.Zero, new IntPtr(intDataSize), cols, 0, null, out _));
            CheckStatus(Cl.EnqueueWriteBuffer(commandQueue, bufferKernelDim, Bool.True, IntPtr.Zero, new IntPtr(intDataSize), kernelDim, 0, null, out _));
            CheckStatus(Cl.EnqueueWriteBuffer(commandQueue, bufferIsHorizontal, Bool.True, IntPtr.Zero, new IntPtr(intDataSize), isHorizontal, 0, null, out _));


            // create the program
            var programSource = File.ReadAllText("blurKernel.cl");
            var program = Cl.CreateProgramWithSource(context, 1, new string[] { programSource }, null, out status);
            CheckStatus(status);

            // build the program
            status = Cl.BuildProgram(program, 1, new Device[] { device }, "", null, IntPtr.Zero);
            if (status != ErrorCode.Success)
            {
                var infoBuffer = Cl.GetProgramBuildInfo(program, device, ProgramBuildInfo.Log, out status);
                CheckStatus(status);
                Console.WriteLine("Build Error: " + infoBuffer);
                Console.ReadLine();
                System.Environment.Exit(1);
            }

            // create the vector addition kernel
            var kernel = Cl.CreateKernel(program, "blur", out status);
            CheckStatus(status);

            // set the kernel arguments
            CheckStatus(Cl.SetKernelArg(kernel, 0, bufferInputImage));
            CheckStatus(Cl.SetKernelArg(kernel, 1, bufferOutputImage));
            CheckStatus(Cl.SetKernelArg(kernel, 2, bufferKernel));
            CheckStatus(Cl.SetKernelArg(kernel, 3, bufferRows));
            CheckStatus(Cl.SetKernelArg(kernel, 4, bufferCols));
            CheckStatus(Cl.SetKernelArg(kernel, 5, bufferKernelDim));
            CheckStatus(Cl.SetKernelArg(kernel, 6, bufferIsHorizontal));

            // output device capabilities
            IntPtr paramSize;
            CheckStatus(Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkGroupSize, IntPtr.Zero, InfoBuffer.Empty, out paramSize));
            var maxWorkGroupSizeBuffer = new InfoBuffer(paramSize);
            CheckStatus(Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkGroupSize, paramSize, maxWorkGroupSizeBuffer, out _));
            var maxWorkGroupSize = maxWorkGroupSizeBuffer.CastTo<int>();
            Console.WriteLine("Device Capabilities: Max work items in single group: " + maxWorkGroupSize);

            CheckStatus(Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkItemDimensions, IntPtr.Zero, InfoBuffer.Empty, out paramSize));
            var dimensionInfoBuffer = new InfoBuffer(paramSize);
            CheckStatus(Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkItemDimensions, paramSize, dimensionInfoBuffer, out _));
            var maxWorkItemDimensions = dimensionInfoBuffer.CastTo<int>();
            Console.WriteLine("Device Capabilities: Max work item dimensions: " + maxWorkItemDimensions);

            CheckStatus(Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkItemSizes, IntPtr.Zero, InfoBuffer.Empty, out paramSize));
            var maxWorkItemSizesInfoBuffer = new InfoBuffer(paramSize);
            CheckStatus(Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkItemSizes, paramSize, maxWorkItemSizesInfoBuffer, out _));
            var maxWorkItemSizes = maxWorkItemSizesInfoBuffer.CastToArray<IntPtr>(maxWorkItemDimensions);
            Console.Write("Device Capabilities: Max work items in group per dimension:");
            for (var i = 0; i < maxWorkItemDimensions; ++i)
            {
                Console.Write(" " + i + ":" + maxWorkItemSizes[i]);
            }

            Console.WriteLine();

            // execute the kernel
            // ndrange capabilities only need to be checked when we specify a local work group size manually
            // in our case we provide NULL as local work group size, which means groups get formed automatically 
            Console.Write("MaxWorkGroupSize: ", DeviceInfo.MaxWorkGroupSize);

            if (maxWorkGroupSize < height || maxWorkGroupSize < width)
            {
                Console.Write("Picture is larger than the MaxWorkGroupSize of ", DeviceInfo.MaxWorkGroupSize);
                return null;
            }

            Console.Write("Started kernel with MaxWorkGroupSize of ", DeviceInfo.MaxWorkGroupSize);
            CheckStatus(Cl.EnqueueNDRangeKernel(commandQueue, kernel, 2, null, new IntPtr[] { new IntPtr(height), new IntPtr(width), }, new IntPtr[] { new IntPtr(1), new IntPtr(width), }, 0, null, out var _));

            isHorizontal = 0;
            CheckStatus(Cl.EnqueueWriteBuffer(commandQueue, bufferIsHorizontal, Bool.True, IntPtr.Zero, new IntPtr(intDataSize), isHorizontal, 0, null, out _));
            CheckStatus(Cl.EnqueueNDRangeKernel(commandQueue, kernel, 2, null, new IntPtr[] { new IntPtr(height), new IntPtr(width), }, new IntPtr[] { new IntPtr(height), new IntPtr(1), }, 0, null, out var _));

            // read the device output buffer to the host output array
            CheckStatus(Cl.EnqueueReadBuffer(commandQueue, bufferOutputImage, Bool.True, IntPtr.Zero, new IntPtr(imageDataSize), outputArray, 0, null, out var _));

            // convert output array to bitmap
            Bitmap outputImage = new Bitmap(width, height);
            BitmapData resultData = outputImage.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
            Marshal.Copy(outputArray, 0, resultData.Scan0, bytes);
            outputImage.UnlockBits(resultData);

            // release opencl objects
            CheckStatus(Cl.ReleaseKernel(kernel));
            CheckStatus(Cl.ReleaseProgram(program));
            CheckStatus(Cl.ReleaseMemObject(bufferInputImage));
            CheckStatus(Cl.ReleaseMemObject(bufferKernel));
            CheckStatus(Cl.ReleaseMemObject(bufferOutputImage));
            CheckStatus(Cl.ReleaseMemObject(bufferRows));
            CheckStatus(Cl.ReleaseMemObject(bufferCols));
            CheckStatus(Cl.ReleaseMemObject(bufferKernelDim));
            CheckStatus(Cl.ReleaseCommandQueue(commandQueue));
            CheckStatus(Cl.ReleaseContext(context));

            return outputImage;
        }

        private static void CheckStatus(ErrorCode err)
        {
            if (err != ErrorCode.Success)
            {
                Console.WriteLine("OpenCL Error: " + err);
                Console.ReadLine();
                System.Environment.Exit(1);
            }
        }
    }
}
