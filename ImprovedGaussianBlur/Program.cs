using System.Drawing;
using System;

namespace ImprovedGaussianBlur
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Choose which cKernel you want to use:\n");
            Console.WriteLine("Just press ENTER to use the default cKernel: \n0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006\n");
            Console.WriteLine("Or type in any characters and then press ENTER to use the custom cKernel: \n" +
                              "0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428\n");
            Console.Write("-->");

            var useCustomCKernel = !string.IsNullOrEmpty(Console.ReadLine());
            Console.WriteLine($"\nUsing {(useCustomCKernel ? "custom cKernel" : "default cKernel")}\n\n");

            using (var inputImage = new Bitmap("./InputImage.bmp"))
            {
                var outputImage = GaussianOpenCl.ApplyGaussianBlur(inputImage, useCustomCKernel);
                if (outputImage != null)
                {
                    outputImage.Save("./OutputImage.bmp");

                }
            }

            Console.WriteLine("Program finished.\nPress ENTER to exit the program.");
            Console.ReadLine();
        }
    }
}
