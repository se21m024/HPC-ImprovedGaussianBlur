using System.Drawing;
using System;

namespace ImprovedGaussianBlur
{
    class Program
    {
        static void Main(string[] args)
        {
            using (var inputImage = new Bitmap("./InputImage.bmp"))
            {
                var outputImage = GaussianOpenCl.ApplyGaussianBlur(inputImage);
                outputImage.Save("./OutputImage.bmp");
            }

            Console.WriteLine("Program finished.\nPress RETURN to exit the program.");
            Console.ReadLine();
        }
    }
}
