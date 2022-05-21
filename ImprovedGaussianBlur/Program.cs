using System.Drawing;
using System;

namespace ImprovedGaussianBlur
{
    class Program
    {
        static void Main(string[] args)
        {


            //Console.WriteLine("Start Sync Demo ----------------------");
            //SyncDemo.DoStuff();
            //Console.WriteLine("Finished Sync Demo ----------------------");

            using (var inputImage = new Bitmap("./InputImage.bmp"))
            {
                var outputImage = GaussianOpenCl.ApplyGaussianBlur(inputImage);
                if (outputImage != null)
                {
                    outputImage.Save("./OutputImage.bmp");

                }
            }

            Console.WriteLine("Program finished.\nPress RETURN to exit the program.");
            Console.ReadLine();
        }
    }
}
