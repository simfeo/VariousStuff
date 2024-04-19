using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;

namespace Fake_MGP_Core.Controllers
{
    public class WebSocketStaticController : Controller
    {
        public IActionResult Index()
        {
            var currentRequestUrl = HttpContext.Request.Host.Value;
            ViewData["currentLink"] = $"ws://{currentRequestUrl}/ws";
            return View();
        }
    }
}