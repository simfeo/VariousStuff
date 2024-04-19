using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace Fake_MGP_Core
{
    public class Program
    {
        public static void Main(string[] args)
        {
            bool threadShouldLive = true;

            new Thread(() =>
            {
                Thread.CurrentThread.IsBackground = true;
                while (threadShouldLive)
                {
                    Thread.Sleep(1000);
                }
                Console.WriteLine("Exiting background thread");
            }).Start();

            CreateWebHostBuilder(args).Build().Run();
            threadShouldLive = false;
        }

        public static IWebHostBuilder CreateWebHostBuilder(string[] args) =>
            WebHost.CreateDefaultBuilder(args)
                .UseStartup<Startup>();
    }
}
