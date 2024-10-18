using System;

public class NormalRandom : Random
{
    private int range_min = 0;
    private int range_max = 0;
    private int average = 0;
    private int step = 0;

    public NormalRandom(int min, int max)
        : base()
    {
        range_min = min;
        range_max = max;
        average = (range_max - range_min) / 2;
        step = average / 5;
    }

    double fi_func(double x)
    {
        return 1.0 / Math.Sqrt(2 * Math.PI) * Math.Pow(Math.E, -0.5 * x * x);
    }

    double ziggurat()
    {
        double max_y = fi_func(0);
        while (true)
        {
            double x = Sample() * (10.0) - 5.0;
            double y = Sample() * max_y;
            double y0 = fi_func(x);
            if (y <= y0)
            {
                return x;
            }
        }
    }

    double box_muller()
    {
        double fi = Sample();
        double r = Sample();
        double z0 = Math.Cos(2 * Math.PI * fi) * Math.Sqrt(-2 * Math.Log(r));
        return z0;
    }

    int normalizeResult(double x)
    {
        return Math.Clamp((int)(x * step + average), range_min, range_max);
    }

    public override int Next()
    {
        return normalizeResult(ziggurat());
    }
}
/*
// this is runner class for demo
public class NormnalRandomDemo
{
    static void Main()
    {

        NormalRandom randObj = new NormalRandom(0, 319);
        int arr_len = 80; // row length for output
        int[] array1 = new int[arr_len];

        for (int i = 0; i < arr_len; ++i)
            array1[i] = 0;

        for (int i = 0; i < 2000; ++i)
        {
            int r = randObj.Next();
            array1[r / 4] += 1;
        }
        int max = 0;

        for (int i = 0; i < arr_len; ++i)
            max = Math.Max(max, array1[i]);


        int height = 25; // lines of output
        for (int j = 0; j < height; ++j)
        {
            for (int i = 0; i < arr_len; ++i)
            {
                Console.Write(array1[i] > (height - j) * max / height ? "*" : " ");
            }
            Console.WriteLine("");
        }
    }
}
*/

/***
 * Result for 1000000 samples

                                      ****
                                     ******
                                    ********
                                   *********
                                   **********
                                  ************
                                 *************
                                 **************
                                ***************
                                ****************
                               *****************
                               ******************
                              *******************
                              ********************
                             *********************
                             **********************
                            ***********************
                           *************************
                           **************************
                          ****************************
                         ******************************
                        ********************************
                      ***********************************
                    ***************************************

 
 * Result for 10000 samples

                                      *
                                     ****
                                    *******
                                   ********
                                   **********
                                   **********
                                  ************
                                 *************
                                 **************
                                ***************
                                ***************
                               *****************
                               ******************
                              ********************
                             *********************
                             **********************
                             **********************
                            ***********************
                          **************************
                          ***************************
                         *****************************
                       *********************************
                       **********************************
                   *****************************************

 * Result for 2000 samples

                                       **
                                       **
                                     * **
                                   * * **
                                   * * **  *
                                   * ***** *
                                   * ***** * *
                                  ********** *
                                  ************
                                 *************
                                ***************
                                ****************
                                ****************
                              * ****************
                              * **************** *
                              ********************
                              ********************
                             ********************* *
                           * ***********************
                          ** ***********************
                          ****************************
                         ******************************
                     ** ********************************
                   * ***************************************
 */
