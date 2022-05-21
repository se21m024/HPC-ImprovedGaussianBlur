using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace ImprovedGaussianBlur
{
    public class SyncDemo
    {
        static void CheckStatus(ErrorCode err)
        {
            if (err != ErrorCode.Success)
            {
                Console.WriteLine("OpenCL Error: " + err.ToString());
                System.Environment.Exit(1);
            }
        }

        static void PrintVector(int[] vector, int elementSize, string label)
        {
            Console.WriteLine(label + ":");

            for (int i = 0; i < elementSize; ++i)
            {
                Console.Write(vector[i] + " ");
            }

            Console.WriteLine();
        }

        public static void DoStuff()
        {


            int[] vectorA = { 0, 1, 2, 3 };
            // input and output arrays
            const int elementSize = 4;
            //const int elementSize = vectorA.Length;

            Double lengthResult = vectorA.Length;

            const int dataSize = elementSize * sizeof(int);


            int[] vectorB = new int[(int)lengthResult];


            // used for checking error status of api calls
            ErrorCode status;

            // retrieve the number of platforms
            uint numPlatforms = 0;
            CheckStatus(Cl.GetPlatformIDs(0, null, out numPlatforms));

            if (numPlatforms == 0)
            {
                Console.WriteLine("Error: No OpenCL platform available!");
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
                System.Environment.Exit(1);
            }

            // select the device
            Device[] devices = new Device[numDevices];
            CheckStatus(Cl.GetDeviceIDs(platform, DeviceType.All, numDevices, devices, out numDevices));
            Device device = devices[0];

            // create context
            Context context = Cl.CreateContext(null, 1, new Device[] { device }, null, IntPtr.Zero, out status);
            CheckStatus(status);

            // create command queue
            CommandQueue commandQueue = Cl.CreateCommandQueue(context, device, 0, out status);
            CheckStatus(status);

            // allocate two input and one output buffer for the three vectors
            IMem<int> bufferA = Cl.CreateBuffer<int>(context, MemFlags.ReadOnly, dataSize, out status);
            CheckStatus(status);
            IMem<int> bufferB = Cl.CreateBuffer<int>(context, MemFlags.WriteOnly, dataSize, out status);
            CheckStatus(status);

            // write data from the input vectors to the buffers
            CheckStatus(Cl.EnqueueWriteBuffer(commandQueue, bufferA, Bool.True, IntPtr.Zero, new IntPtr(dataSize), vectorA, 0, null, out var _));
            //CheckStatus(Cl.EnqueueWriteBuffer(commandQueue, bufferB, Bool.True, IntPtr.Zero, new IntPtr(dataSize), vectorB, 0, null, out var _));

            // create the program
            string programSource = File.ReadAllText("syncKernel.cl");
            OpenCL.Net.Program program = Cl.CreateProgramWithSource(context, 1, new string[] { programSource }, null, out status);
            CheckStatus(status);

            // build the program
            status = Cl.BuildProgram(program, 1, new Device[] { device }, "", null, IntPtr.Zero);
            if (status != ErrorCode.Success)
            {
                InfoBuffer infoBuffer = Cl.GetProgramBuildInfo(program, device, ProgramBuildInfo.Log, out status);
                CheckStatus(status);
                Console.WriteLine("Build Error: " + infoBuffer);
                System.Environment.Exit(1);
            }

            // create the vector addition kernel
            OpenCL.Net.Kernel kernel = Cl.CreateKernel(program, "sync_demo_kernel", out status);
            CheckStatus(status);

            // set the kernel arguments
            CheckStatus(Cl.SetKernelArg(kernel, 0, bufferA));
            CheckStatus(Cl.SetKernelArg(kernel, 1, bufferB));

            // output device capabilities
            IntPtr paramSize;
            CheckStatus(Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkGroupSize, IntPtr.Zero, InfoBuffer.Empty, out paramSize));
            InfoBuffer maxWorkGroupSizeBuffer = new InfoBuffer(paramSize);
            CheckStatus(Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkGroupSize, paramSize, maxWorkGroupSizeBuffer, out paramSize));
            int maxWorkGroupSize = maxWorkGroupSizeBuffer.CastTo<int>();
            Console.WriteLine("Device Capabilities: Max work items in single group: " + maxWorkGroupSize);

            CheckStatus(Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkItemDimensions, IntPtr.Zero, InfoBuffer.Empty, out paramSize));
            InfoBuffer dimensionInfoBuffer = new InfoBuffer(paramSize);
            CheckStatus(Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkItemDimensions, paramSize, dimensionInfoBuffer, out paramSize));
            int maxWorkItemDimensions = dimensionInfoBuffer.CastTo<int>();
            Console.WriteLine("Device Capabilities: Max work item dimensions: " + maxWorkItemDimensions);

            CheckStatus(Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkItemSizes, IntPtr.Zero, InfoBuffer.Empty, out paramSize));
            InfoBuffer maxWorkItemSizesInfoBuffer = new InfoBuffer(paramSize);
            CheckStatus(Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkItemSizes, paramSize, maxWorkItemSizesInfoBuffer, out paramSize));
            IntPtr[] maxWorkItemSizes = maxWorkItemSizesInfoBuffer.CastToArray<IntPtr>(maxWorkItemDimensions);
            Console.Write("Device Capabilities: Max work items in group per dimension:");
            for (int i = 0; i < maxWorkItemDimensions; ++i)
                Console.Write(" " + i + ":" + maxWorkItemSizes[i]);
            Console.WriteLine();

            // execute the kernel
            // ndrange capabilities only need to be checked when we specify a local work group size manually
            // in our case we provide NULL as local work group size, which means groups get formed automatically
            CheckStatus(Cl.EnqueueNDRangeKernel(commandQueue, kernel, 1, null, new IntPtr[] { new IntPtr(elementSize) }, null, 0, null, out var _));

            // read the device output buffer to the host output array
            CheckStatus(Cl.EnqueueReadBuffer(commandQueue, bufferB, Bool.True, IntPtr.Zero, new IntPtr(dataSize), vectorB, 0, null, out var _));

            // output result
            PrintVector(vectorA, elementSize, "Input A");
            PrintVector(vectorB, elementSize, "Output B");

            // release opencl objects
            CheckStatus(Cl.ReleaseKernel(kernel));
            CheckStatus(Cl.ReleaseProgram(program));
            CheckStatus(Cl.ReleaseMemObject(bufferB));
            CheckStatus(Cl.ReleaseMemObject(bufferA));
            CheckStatus(Cl.ReleaseCommandQueue(commandQueue));
            CheckStatus(Cl.ReleaseContext(context));
        }
    }
}
